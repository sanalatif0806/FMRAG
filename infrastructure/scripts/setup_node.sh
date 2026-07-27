#!/usr/bin/env bash
# infrastructure/scripts/setup_node.sh
# ──────────────────────────────────────
# Provisions an Ubuntu 22.04 VMware VM as an FMRAG node.
# Run as root inside each VMware VM after OS installation.
#
# Usage:
#   sudo bash setup_node.sh --role central --ip 172.16.24.34
#   sudo bash setup_node.sh --role client  --ip 192.168.1.10
#
# Roles:
#   central  CIPYZ broker + FL server + MySQL + Apache + psdash
#   client   CIPYZ client daemon + FL training + MySQL + Apache + psdash
#
# After running this script on each VM:
#   1. Install VMware Tools inside the VM:
#        sudo apt-get install open-vm-tools
#   2. Install vmrun + ovftool on the HOST machine (not inside the VM)
#   3. Edit /etc/fmrag/env with correct IPs and passwords
#   4. Register this VM: mysql -u fmrag -p fmrag -e "INSERT INTO resource..."
#   5. Start services: sudo systemctl start fmrag-*

set -euo pipefail

ROLE=""
NODE_IP=""
CENTRAL_IP="${CENTRAL_IP:-172.16.24.34}"
DB_PASS="${DB_PASS:-FmragSecure2024!}"
PROJECT_ROOT="${PROJECT_ROOT:-/opt/fmrag-project}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --role)       ROLE="$2";       shift 2 ;;
    --ip)         NODE_IP="$2";    shift 2 ;;
    --central-ip) CENTRAL_IP="$2"; shift 2 ;;
    --db-pass)    DB_PASS="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$ROLE" || -z "$NODE_IP" ]] && {
  echo "Usage: $0 --role <central|client> --ip 100.105.131.115 [--central-ip <central_vm_ip>]"
  exit 1
}

# A central node's own IP IS the central IP — there's no scenario where
# they'd differ. This used to fall back to a hardcoded example default
# (172.16.24.34) whenever --central-ip was omitted on the central node,
# which silently wrote the WRONG value into /etc/fmrag/env. Fixed by
# deriving it automatically instead of relying on the caller to pass a
# redundant argument.
if [[ "$ROLE" == "central" ]]; then
  CENTRAL_IP="$NODE_IP"
elif [[ "$CENTRAL_IP" == "100.105.131.115" ]]; then
  # Client role, and --central-ip was never passed — this default is
  # almost certainly wrong for your network. Fail loudly instead of
  # silently writing a broken value, the way this used to.
  echo "ERROR: --central-ip was not provided for a client node."
  echo "       Pass the actual central/server VM's IP explicitly, e.g.:"
  echo "       $0 --role client --ip $NODE_IP --central-ip <server_ip>"
  exit 1
fi

echo "═══════════════════════════════════════════════"
echo " FMRAG node setup"
echo " Role    : $ROLE"
echo " IP      : $NODE_IP"
echo " Central : $CENTRAL_IP"
echo "═══════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3.10 python3.10-venv python3-pip \
  mysql-server mysql-client libmysqlclient-dev \
  apache2 php php-mysqli \
  sshpass openssh-client openssh-server \
  iputils-ping curl wget git \
  build-essential pkg-config libssl-dev \
  open-vm-tools          2>/dev/null || true

# VMware tools check (vmrun and ovftool run on the HOST, not inside the VM)
echo "NOTE: vmrun and ovftool must be installed on the HOST machine,"
echo "      not inside this VM. VMware Workstation on the host provides vmrun."

# ── 2. fmrag user and directories ────────────────────────────────────────────
id -u fmrag &>/dev/null || useradd -r -m -s /bin/bash fmrag

mkdir -p /var/fmrag/{vms,data,cache,checkpoints,logs} \
         /var/www/html/uploads \
         /etc/fmrag \
         /var/log/fmrag

chown -R fmrag:fmrag /var/fmrag /var/log/fmrag
chmod 755 /var/www/html/uploads

# ── 3. Environment config ─────────────────────────────────────────────────────
cat > /etc/fmrag/env << EOF
# FMRAG node environment
# Edit this file, then restart services: sudo systemctl restart fmrag-*

FMRAG_NODE_ROLE=${ROLE}
FMRAG_NODE_IP=${NODE_IP}
FMRAG_CENTRAL_IP=${CENTRAL_IP}

FMRAG_DB_HOST=localhost
FMRAG_DB_USER=fmrag
FMRAG_DB_PASS=${DB_PASS}
FMRAG_DB_NAME=fmrag

# VMware — paths for vmrun and ovftool on this machine
# (Central node does not need these; client nodes do)
FMRAG_VM_STORE=/var/fmrag/vms
FMRAG_OVA_UPLOAD=/var/www/html/uploads
VMRUN_BIN=/usr/lib/vmware/bin/vmrun
OVFTOOL_BIN=/usr/bin/ovftool
VMWARE_TYPE=ws
FMRAG_BRIDGE_IFACE=ens33
FMRAG_VM_USER=fmrag
FMRAG_VM_PASS=${DB_PASS}

# CPU mode — edge VMs run on CPU, not GPU
FMRAG_CPU_MODE=true
FMRAG_LOCAL_EPOCHS=1
FMRAG_BATCH_SIZE=4
FMRAG_GRAD_ACCUM=4
FMRAG_EWC_SAMPLES=50

FMRAG_POLL_INTERVAL=30
FMRAG_BROKER_POLL=30

FMRAG_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(24))")
PYTHONPATH=${PROJECT_ROOT}/FMRAG
EOF
chmod 640 /etc/fmrag/env
chown root:fmrag /etc/fmrag/env
echo "Environment written to /etc/fmrag/env"

# ── 4. Python virtual environment ─────────────────────────────────────────────
# python3.10 is native to Ubuntu 22.04 Jammy — no extra PPA/package needed,
# and matches the working venv already confirmed on real deployments.
python3.10 -m venv /var/fmrag/venv
source /var/fmrag/venv/bin/activate
pip install --upgrade pip -q

echo "Installing PyTorch CPU (this takes 3-5 minutes)..."
# numpy MUST be pinned <2 and installed BEFORE torch — torch 2.2.0 was
# built against NumPy 1.x's ABI and segfaults under NumPy 2.x. This was
# previously a bare, unpinned `numpy` later in the dependency list below,
# which silently upgraded to 2.x after torch was already installed and
# broke torch_geometric with a segfault. Fixed here instead of relying on
# a manual post-install `pip install "numpy<2" --force-reinstall` fix.
pip install -q "numpy==1.26.4"
pip install -q torch==2.2.0 \
  --index-url https://download.pytorch.org/whl/cpu

echo "Installing torch-geometric..."
pip install -q \
  torch-geometric==2.5.2 \
  torch-scatter==2.1.2 \
  torch-sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

echo "Installing remaining dependencies..."
pip install -q \
  transformers==4.40.0 \
  peft==0.10.0 \
  accelerate==0.29.3 \
  pyhealth==1.1.4 \
  flask flask-login \
  mysqlclient \
  gevent \
  psutil \
  requests \
  pyyaml \
  scikit-learn \
  pandas \
  networkx \
  python-dotenv \
  bcrypt \
  zerorpc \
  werkzeug \
  tqdm 2>/dev/null || true

# Optional: 8-bit quantisation to halve RAM usage (install if pip can reach PyPI)
pip install -q bitsandbytes 2>/dev/null || \
  echo "bitsandbytes not installed — model will use float32 (works fine, uses more RAM)"

deactivate
echo "Python environment ready"

# ── 5. MySQL setup ────────────────────────────────────────────────────────────
systemctl start mysql 2>/dev/null || true

mysql -u root << MYSQL
CREATE DATABASE IF NOT EXISTS fmrag
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'fmrag'@'localhost' IDENTIFIED BY '${DB_PASS}';
CREATE USER IF NOT EXISTS 'fmrag'@'%'         IDENTIFIED BY '${DB_PASS}';
GRANT ALL PRIVILEGES ON fmrag.* TO 'fmrag'@'localhost';
GRANT ALL PRIVILEGES ON fmrag.* TO 'fmrag'@'%';
FLUSH PRIVILEGES;
MYSQL

# Allow remote MySQL connections (broker needs to connect to client DBs)
sed -i "s/bind-address.*=.*/bind-address = 0.0.0.0/" /etc/mysql/mysql.conf.d/mysqld.cnf 2>/dev/null || true
systemctl restart mysql

# Run schema
mysql -u fmrag -p"${DB_PASS}" fmrag \
  < "${PROJECT_ROOT}/FMRAG/infrastructure/database/fmrag_schema.sql"
echo "Database fmrag initialised"

# ── 6. Apache + PHP frontend ──────────────────────────────────────────────────
if [[ -d "${PROJECT_ROOT}/Client_ver8/client/html" ]]; then
  cp -r ${PROJECT_ROOT}/Client_ver8/client/html/* /var/www/html/
  # Update credentials in db.php
  sed -i "s/100\.105\.131\.115/${CENTRAL_IP}/g" /var/www/html/db.php 2>/dev/null || true
  sed -i "s/123/${DB_PASS}/g"             /var/www/html/db.php 2>/dev/null || true
  chown -R www-data:www-data /var/www/html/
  echo "PHP frontend installed"
fi
a2enmod rewrite 2>/dev/null || true
systemctl restart apache2 2>/dev/null || true

# ── 7. Systemd services ───────────────────────────────────────────────────────

# psdash monitor — runs on all nodes
cat > /etc/systemd/system/fmrag-monitor.service << EOF
[Unit]
Description=FMRAG psdash system monitor
After=network.target mysql.service

[Service]
Type=simple
User=fmrag
EnvironmentFile=/etc/fmrag/env
WorkingDirectory=${PROJECT_ROOT}/FMRAG
ExecStart=/var/fmrag/venv/bin/python -m monitoring.psdash_app --bind 0.0.0.0 --port 5000
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/fmrag/monitor.log
StandardError=append:/var/log/fmrag/monitor.log

[Install]
WantedBy=multi-user.target
EOF

if [[ "$ROLE" == "central" ]]; then

  cat > /etc/systemd/system/fmrag-broker.service << EOF
[Unit]
Description=FMRAG CIPYZ broker (resource polling + VM offload decisions)
After=network.target mysql.service

[Service]
Type=simple
User=fmrag
EnvironmentFile=/etc/fmrag/env
WorkingDirectory=${PROJECT_ROOT}/FMRAG
ExecStart=/var/fmrag/venv/bin/python -m infrastructure.broker.broker_daemon
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/fmrag/broker.log
StandardError=append:/var/log/fmrag/broker.log

[Install]
WantedBy=multi-user.target
EOF

  cat > /etc/systemd/system/fmrag-fl-server.service << EOF
[Unit]
Description=FMRAG FL server (FedAvg aggregation, port 8080 (configurable via FMRAG_FL_PORT))
After=network.target mysql.service fmrag-broker.service

[Service]
Type=simple
User=fmrag
EnvironmentFile=/etc/fmrag/env
WorkingDirectory=${PROJECT_ROOT}/FMRAG
ExecStart=/var/fmrag/venv/bin/python -m server.fl_server
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/fmrag/fl_server.log
StandardError=append:/var/log/fmrag/fl_server.log

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable  fmrag-broker fmrag-fl-server fmrag-monitor
  systemctl start   fmrag-broker fmrag-fl-server fmrag-monitor
fi

if [[ "$ROLE" == "client" ]]; then

  # Resource reporter — pushes live psutil stats to local + central MySQL
  cat > /etc/systemd/system/fmrag-reporter.service << EOF
[Unit]
Description=FMRAG resource reporter (live stats to central)
After=network.target mysql.service

[Service]
Type=simple
User=ubuntu
EnvironmentFile=/etc/fmrag/env
WorkingDirectory=${PROJECT_ROOT}/FMRAG
ExecStart=/var/fmrag/venv/bin/python -m infrastructure.client_node.resource_reporter
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/fmrag/reporter.log
StandardError=append:/var/log/fmrag/reporter.log

[Install]
WantedBy=multi-user.target
EOF

  cat > /etc/systemd/system/fmrag-client.service << EOF
[Unit]
Description=FMRAG client daemon (CIPYZ VM management + FL training)
After=network.target mysql.service

[Service]
Type=simple
User=fmrag
EnvironmentFile=/etc/fmrag/env
WorkingDirectory=${PROJECT_ROOT}/FMRAG
ExecStart=/var/fmrag/venv/bin/python -m infrastructure.client_node.client_daemon \
  --fl-config /etc/fmrag/client_config.yaml
Restart=on-failure
RestartSec=30
StandardOutput=append:/var/log/fmrag/client.log
StandardError=append:/var/log/fmrag/client.log

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable  fmrag-reporter fmrag-client fmrag-monitor
  systemctl start   fmrag-reporter fmrag-client fmrag-monitor
fi

# ── 8. Register this node in MySQL ────────────────────────────────────────────
HOSTNAME=$(hostname)
mysql -u fmrag -p"${DB_PASS}" fmrag << MYSQL
INSERT IGNORE INTO resource (ip_address, cloudlet_name, status)
VALUES ('${NODE_IP}', '${HOSTNAME}', 'online');
MYSQL
echo "Node registered: ${NODE_IP} (${HOSTNAME})"

echo ""
echo "═══════════════════════════════════════════════"
echo " Setup complete: ${ROLE} @ ${NODE_IP}"
echo "═══════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Edit /etc/fmrag/env — verify IPs and passwords"
if [[ "$ROLE" == "client" ]]; then
  echo "  2. Edit /etc/fmrag/client_config.yaml — set data.root to your MIMIC path"
  echo "  3. Ensure vmrun is on PATH: which vmrun"
  echo "  4. Run GraphCare data pipeline once:"
  echo "       cd /opt/fmrag-project/GraphCare-main"
  echo "       source /var/fmrag/venv/bin/activate"
  echo "       python data_prepare.py --dataset mimic3 --task mortality"
fi
echo ""
echo "Then start services:"
echo "  sudo systemctl start fmrag-broker fmrag-fl-server   (central VM only)"
echo "  sudo systemctl start fmrag-client                    (client VMs)"
echo "  sudo systemctl start fmrag-monitor                   (all VMs)"
echo ""
echo "Logs: /var/log/fmrag/"
echo "Web:  http://${NODE_IP}/"
echo "Monitor: http://${NODE_IP}:5000/"
