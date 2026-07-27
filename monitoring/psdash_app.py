"""
monitoring/psdash_app.py
-------------------------
Python 3 rewrite of Client_ver8/client/psdash/run.py + web.py

Removes Python 2-only dependencies:
  urllib2           → urllib.request
  mysql.fabric      → removed (was only used for timestamp util)
  from __future__   → not needed in Python 3

Keeps:
  Flask + Blueprint  (psdash web UI)
  gevent WSGIServer  (async serving)
  zerorpc            (remote node RPC)
  psutil             (system metrics)
  MySQLdb            (local DB for resource/status tracking)

Run:
    python -m monitoring.psdash_app --bind 0.0.0.0 --port 5000
"""

from __future__ import annotations
import argparse
import logging
import os
import socket
import threading
import time
from datetime import datetime, timedelta

import gevent
from gevent import monkey
monkey.patch_all()

from gevent.pywsgi import WSGIServer
import psutil
import MySQLdb
import zerorpc
from flask import (
    Flask, Blueprint, render_template, request,
    session, jsonify, g, current_app,
)
from werkzeug.local import LocalProxy

log = logging.getLogger("fmrag.monitor")

# ── Config ────────────────────────────────────────────────────────────────────
DB_HOST  = os.environ.get("FMRAG_DB_HOST",  "localhost")
DB_USER  = os.environ.get("FMRAG_DB_USER",  "fmrag")
DB_PASS  = os.environ.get("FMRAG_DB_PASS",  "FmragSecure2024!")
DB_NAME  = os.environ.get("FMRAG_DB_NAME",  "fmrag")
NODE_IP  = os.environ.get("FMRAG_NODE_IP",  socket.gethostbyname(socket.gethostname()))

webapp = Blueprint("psdash", __name__,
                   template_folder="templates",
                   static_folder="static")


# ── DB helper ─────────────────────────────────────────────────────────────────

def get_db():
    return MySQLdb.connect(host=DB_HOST, user=DB_USER,
                           passwd=DB_PASS, db=DB_NAME)


def db_query(sql: str, params=None) -> list:
    db  = get_db()
    cur = db.cursor()
    cur.execute(sql, params or ())
    rows = cur.fetchall()
    db.close()
    return rows


# ── System metrics ────────────────────────────────────────────────────────────

def get_sysinfo() -> dict:
    cpu    = psutil.cpu_percent(interval=0.5)
    mem    = psutil.virtual_memory()
    disk   = psutil.disk_usage("/")
    boot   = psutil.boot_time()
    uptime = int(time.time() - boot)
    return {
        "hostname":     socket.gethostname(),
        "os":           f"{os.uname().sysname} {os.uname().release}",
        "uptime":       uptime,
        "cpu_percent":  cpu,
        "cpu_cores":    psutil.cpu_count(logical=False),
        "mem_total_gb": round(mem.total / 1e9, 1),
        "mem_used_gb":  round(mem.used  / 1e9, 1),
        "mem_percent":  mem.percent,
        "disk_total_gb":round(disk.total / 1e9, 1),
        "disk_used_gb": round(disk.used  / 1e9, 1),
        "disk_percent": disk.percent,
        "node_ip":      NODE_IP,
    }


def push_resource_stats():
    """Write current host metrics to the resource table every 30s."""
    while True:
        try:
            info = get_sysinfo()
            db = get_db()
            cur = db.cursor()
            cur.execute("""
                UPDATE resource SET
                    cpu_cores=%s, cpu_avload=%s,
                    memory_total=%s, memory_free=%s,
                    disk_total=%s, disk_free=%s,
                    last_updated=NOW()
                WHERE ip_address=%s
            """, (
                info["cpu_cores"],
                info["cpu_percent"] / 100.0,
                info["mem_total_gb"],
                info["mem_total_gb"] - info["mem_used_gb"],
                info["disk_total_gb"],
                info["disk_total_gb"] - info["disk_used_gb"],
                NODE_IP,
            ))
            db.commit()
            db.close()
        except Exception as e:
            log.warning("Resource push failed: %s", e)
        time.sleep(30)


# ── Web routes ────────────────────────────────────────────────────────────────

@webapp.route("/")
def index():
    info = get_sysinfo()
    uptime_str = str(timedelta(seconds=info["uptime"])).split(".")[0]
    return render_template("index.html", sysinfo=info, uptime=uptime_str)


@webapp.route("/api/sysinfo")
def api_sysinfo():
    return jsonify(get_sysinfo())


@webapp.route("/api/processes")
def api_processes():
    procs = []
    for p in psutil.process_iter(
        ["pid", "name", "username", "cpu_percent", "memory_percent", "status"]
    ):
        try:
            procs.append(p.info)
        except psutil.NoSuchProcess:
            pass
    return jsonify(sorted(procs, key=lambda x: x["cpu_percent"] or 0, reverse=True)[:50])


@webapp.route("/api/fl_rounds")
def api_fl_rounds():
    rows = db_query(
        "SELECT round_number, started_at, completed_at, n_clients, "
        "global_auroc, comm_cost_mb FROM fl_round ORDER BY round_number DESC LIMIT 20"
    )
    keys = ["round", "started", "completed", "clients", "auroc", "comm_mb"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@webapp.route("/api/resource")
def api_resource():
    rows = db_query(
        "SELECT ip_address, cloudlet_name, status, cpu_avload, "
        "memory_free, memory_total, last_updated FROM resource"
    )
    keys = ["ip", "name", "status", "cpu_load", "mem_free", "mem_total", "updated"]
    return jsonify([dict(zip(keys, r)) for r in rows])


@webapp.route("/api/hypotheses")
def api_hypotheses():
    rows = db_query(
        "SELECT drug_name, drug_class, score, contraindicated, "
        "narrative, generated_at FROM drug_hypothesis "
        "ORDER BY generated_at DESC LIMIT 50"
    )
    keys = ["drug", "class", "score", "contraindicated", "narrative", "at"]
    return jsonify([dict(zip(keys, r)) for r in rows])


# ── Minimal HTML template (inline, no template files required) ───────────────

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FMRAG Monitor — {{ sysinfo.hostname }}</title>
<style>
  body{font-family:monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:20px}
  h1{color:#58a6ff;margin:0 0 4px}
  .sub{color:#8b949e;font-size:12px;margin-bottom:20px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:24px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
  .card h3{margin:0 0 8px;color:#79c0ff;font-size:13px;text-transform:uppercase}
  .val{font-size:28px;font-weight:bold;color:#e6edf3}
  .bar-bg{background:#21262d;border-radius:4px;height:6px;margin-top:8px}
  .bar{height:6px;border-radius:4px;background:#1f6feb;transition:width .4s}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;padding:6px 10px;color:#8b949e;border-bottom:1px solid #21262d}
  td{padding:6px 10px;border-bottom:1px solid #161b22}
  tr:hover td{background:#161b22}
</style>
</head>
<body>
<h1>FMRAG Monitor</h1>
<div class="sub">{{ sysinfo.hostname }} · {{ sysinfo.node_ip }} · {{ sysinfo.os }} · up {{ uptime }}</div>

<div class="grid">
  <div class="card">
    <h3>CPU</h3>
    <div class="val">{{ sysinfo.cpu_percent }}%</div>
    <div class="bar-bg"><div class="bar" style="width:{{ sysinfo.cpu_percent }}%"></div></div>
    <div style="font-size:12px;color:#8b949e;margin-top:4px">{{ sysinfo.cpu_cores }} cores</div>
  </div>
  <div class="card">
    <h3>Memory</h3>
    <div class="val">{{ sysinfo.mem_percent }}%</div>
    <div class="bar-bg"><div class="bar" style="width:{{ sysinfo.mem_percent }}%"></div></div>
    <div style="font-size:12px;color:#8b949e;margin-top:4px">{{ sysinfo.mem_used_gb }} / {{ sysinfo.mem_total_gb }} GB</div>
  </div>
  <div class="card">
    <h3>Disk</h3>
    <div class="val">{{ sysinfo.disk_percent }}%</div>
    <div class="bar-bg"><div class="bar" style="width:{{ sysinfo.disk_percent }}%"></div></div>
    <div style="font-size:12px;color:#8b949e;margin-top:4px">{{ sysinfo.disk_used_gb }} / {{ sysinfo.disk_total_gb }} GB</div>
  </div>
</div>

<h2 style="color:#58a6ff;font-size:14px">FL Rounds</h2>
<div id="fl-rounds"><table>
<thead><tr><th>Round</th><th>Started</th><th>Clients</th><th>AUROC</th><th>Comm (MB)</th></tr></thead>
<tbody id="fl-tbody"></tbody>
</table></div>

<h2 style="color:#58a6ff;font-size:14px;margin-top:20px">Cloudlet Network</h2>
<div id="cloudlets"><table>
<thead><tr><th>IP</th><th>Name</th><th>Status</th><th>CPU Load</th><th>Mem Free</th><th>Updated</th></tr></thead>
<tbody id="cloud-tbody"></tbody>
</table></div>

<script>
async function refresh() {
  const [rounds, res] = await Promise.all([
    fetch('/api/fl_rounds').then(r=>r.json()),
    fetch('/api/resource').then(r=>r.json()),
  ]);
  document.getElementById('fl-tbody').innerHTML = rounds.map(r =>
    `<tr><td>${r.round}</td><td>${r.started}</td><td>${r.clients}</td>
     <td>${r.auroc ? r.auroc.toFixed(4) : '—'}</td>
     <td>${r.comm_mb ? r.comm_mb.toFixed(2) : '—'}</td></tr>`
  ).join('');
  document.getElementById('cloud-tbody').innerHTML = res.map(r =>
    `<tr><td>${r.ip}</td><td>${r.name}</td>
     <td style="color:${r.status==='online'?'#3fb950':'#f85149'}">${r.status}</td>
     <td>${(r.cpu_load*100).toFixed(1)}%</td>
     <td>${r.mem_free ? r.mem_free.toFixed(1)+' GB' : '—'}</td>
     <td>${r.updated}</td></tr>`
  ).join('');
}
refresh();
setInterval(refresh, 15000);
</script>
</body>
</html>"""


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("FMRAG_SECRET_KEY", "fmrag-dev-secret-change-me")
    app.jinja_env.globals["now"] = datetime.now

    # Serve inline template without template directory
    @app.route("/")
    def index():
        from flask import render_template_string
        info = get_sysinfo()
        uptime_str = str(timedelta(seconds=info["uptime"])).split(".")[0]
        return render_template_string(INDEX_HTML, sysinfo=info, uptime=uptime_str)

    app.register_blueprint(webapp, url_prefix="/dash")

    # Register API routes directly
    app.add_url_rule("/api/sysinfo",    view_func=api_sysinfo)
    app.add_url_rule("/api/processes",  view_func=api_processes)
    app.add_url_rule("/api/fl_rounds",  view_func=api_fl_rounds)
    app.add_url_rule("/api/resource",   view_func=api_resource)
    app.add_url_rule("/api/hypotheses", view_func=api_hypotheses)

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [MONITOR] %(message)s",
    )
    parser = argparse.ArgumentParser(description="FMRAG monitoring dashboard")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    # Background thread: push resource stats to DB
    t = threading.Thread(target=push_resource_stats, daemon=True)
    t.start()

    app    = create_app()
    server = WSGIServer((args.bind, args.port), app)
    log.info("FMRAG monitor running on http://%s:%d/", args.bind, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
