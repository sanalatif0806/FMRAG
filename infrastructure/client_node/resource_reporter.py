"""
infrastructure/client_node/resource_reporter.py
-------------------------------------------------
Runs on each client VM. Every POLL_INTERVAL seconds:
  1. Collects real system stats using psutil
  2. Updates the local MySQL resource table
  3. Pushes the same update to the central MySQL
     (so the broker and web console see live client stats)

This replaces the hardcoded INSERT into resource table.
The broker uses these values for offload decisions:
  cpu_cores    → vm_cpu eligibility filter
  disk_free    → vm_storage eligibility filter
  memory_free  → vm_memory eligibility filter
  resource_index → max(resource_index) destination selection

Runs as a background thread inside client_daemon.py
OR as a standalone systemd service.
"""

from __future__ import annotations

import logging
import os
import socket
import time

import psutil

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
NODE_IP       = os.environ.get("FMRAG_NODE_IP",      "")
NODE_NAME     = os.environ.get("FMRAG_NODE_NAME",    socket.gethostname())
CENTRAL_IP    = os.environ.get("FMRAG_CENTRAL_IP",   "100.105.131.115")
DB_HOST_LOCAL = os.environ.get("FMRAG_DB_HOST",      "localhost")
DB_HOST_CENTRAL = CENTRAL_IP
DB_USER       = os.environ.get("FMRAG_DB_USER",      "fmrag")
DB_PASS       = os.environ.get("FMRAG_DB_PASS",      "")
DB_NAME       = os.environ.get("FMRAG_DB_NAME",      "fmrag")
POLL_INTERVAL = int(os.environ.get("FMRAG_POLL_INTERVAL", "30"))

# Resource thresholds (same as broker_daemon.py)
LEVEL_NORMAL  = 0.60
LEVEL_HIGH    = 0.40


# ── Stats collection ──────────────────────────────────────────────────────────

def collect_stats() -> dict:
    """Collect real system stats using psutil."""

    # CPU
    cpu_percent  = psutil.cpu_percent(interval=1)
    cpu_cores    = psutil.cpu_count(logical=True) or 1
    cpu_avload   = cpu_percent / 100.0

    # Memory
    mem          = psutil.virtual_memory()
    memory_total = round(mem.total / (1024 ** 3), 2)      # GB
    memory_free  = round(mem.available / (1024 ** 3), 2)  # GB
    memory_used  = round(mem.used / (1024 ** 3), 2)       # GB

    # Disk
    disk         = psutil.disk_usage("/")
    disk_total   = round(disk.total / (1024 ** 3), 2)     # GB
    disk_free    = round(disk.free / (1024 ** 3), 2)      # GB

    # Network
    net          = psutil.net_io_counters()
    bytes_sent   = net.bytes_sent
    bytes_recv   = net.bytes_recv

    # Resource index (same formula as broker_daemon.py)
    cpu_score    = max(0.0, 1.0 - cpu_avload)
    mem_score    = memory_free / memory_total if memory_total > 0 else 0.0
    resource_index = round(0.5 * cpu_score + 0.5 * mem_score, 4)

    resource_level = (
        "normal"   if resource_index >= LEVEL_NORMAL else
        "high"     if resource_index >= LEVEL_HIGH   else
        "critical"
    )

    return {
        "ip_address":      NODE_IP or _get_local_ip(),
        "cloudlet_name":   NODE_NAME,
        "status":          "online",
        "cpu_cores":       cpu_cores,
        "cpu_avload":      round(cpu_avload, 4),
        "memory_total":    memory_total,
        "memory_free":     memory_free,
        "memory_used":     memory_used,
        "disk_total":      disk_total,
        "disk_free":       disk_free,
        "bytes_sent":      bytes_sent,
        "bytes_recv":      bytes_recv,
        "resource_index":  resource_index,
        "resource_level":  resource_level,
    }


def _get_local_ip() -> str:
    """Get the primary network interface IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── MySQL upsert ──────────────────────────────────────────────────────────────

def upsert_resource(host: str, stats: dict, label: str = "local"):
    """Upsert resource row into MySQL at the given host."""
    try:
        import MySQLdb
        db  = MySQLdb.connect(
            host=host, user=DB_USER, passwd=DB_PASS, db=DB_NAME,
            connect_timeout=5,
        )
        cur = db.cursor()

        cur.execute("""
            INSERT INTO resource
              (ip_address, cloudlet_name, status,
               cpu_cores, cpu_avload,
               memory_total, memory_free, memory_used,
               disk_total, disk_free,
               bytes_sent, bytes_recv)
            VALUES
              (%(ip_address)s, %(cloudlet_name)s, %(status)s,
               %(cpu_cores)s, %(cpu_avload)s,
               %(memory_total)s, %(memory_free)s, %(memory_used)s,
               %(disk_total)s, %(disk_free)s,
               %(bytes_sent)s, %(bytes_recv)s)
            ON DUPLICATE KEY UPDATE
              cloudlet_name = VALUES(cloudlet_name),
              status        = VALUES(status),
              cpu_cores     = VALUES(cpu_cores),
              cpu_avload    = VALUES(cpu_avload),
              memory_total  = VALUES(memory_total),
              memory_free   = VALUES(memory_free),
              memory_used   = VALUES(memory_used),
              disk_total    = VALUES(disk_total),
              disk_free     = VALUES(disk_free),
              bytes_sent    = VALUES(bytes_sent),
              bytes_recv    = VALUES(bytes_recv)
        """, stats)
        db.commit()

        # Also update decision_parameters
        cur.execute("""
            INSERT INTO decision_parameters
              (cloudlet_ip, resource_index, resource_level)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE
              resource_index = VALUES(resource_index),
              resource_level = VALUES(resource_level)
        """, (stats["ip_address"],
              stats["resource_index"],
              stats["resource_level"]))
        db.commit()
        db.close()

        log.debug("[%s] resource updated: index=%.4f level=%s cpu=%.1f%% mem_free=%.1fGB",
                  label,
                  stats["resource_index"],
                  stats["resource_level"],
                  stats["cpu_avload"] * 100,
                  stats["memory_free"])

    except Exception as e:
        log.warning("[%s] resource update failed (%s): %s", label, host, e)


# ── Main reporter loop ────────────────────────────────────────────────────────

def run():
    """
    Main loop — collect stats and push to local + central MySQL.
    Run as a background thread or standalone process.
    """
    log.info("Resource reporter starting")
    log.info("  Node IP    : %s", NODE_IP or _get_local_ip())
    log.info("  Node name  : %s", NODE_NAME)
    log.info("  Central IP : %s", CENTRAL_IP)
    log.info("  Poll       : %ds", POLL_INTERVAL)

    while True:
        try:
            stats = collect_stats()

            # Push to local MySQL
            upsert_resource(DB_HOST_LOCAL, stats, label="local")

            # Push to central MySQL (broker + web console read from here)
            if CENTRAL_IP and CENTRAL_IP != DB_HOST_LOCAL:
                upsert_resource(DB_HOST_CENTRAL, stats, label="central")

            log.info(
                "Stats: cpu=%.1f%% mem_free=%.1f/%.1fGB disk_free=%.1fGB "
                "index=%.4f level=%s",
                stats["cpu_avload"] * 100,
                stats["memory_free"], stats["memory_total"],
                stats["disk_free"],
                stats["resource_index"],
                stats["resource_level"],
            )

        except Exception as e:
            log.error("Resource reporter error: %s", e, exc_info=True)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [REPORTER] %(message)s",
    )
    run()
