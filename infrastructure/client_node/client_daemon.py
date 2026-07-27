"""
infrastructure/client_node/client_daemon.py
---------------------------------------------
Python 3 faithful rewrite of Client_ver8/client/psdash/vm.py

Every case from vm.py is preserved exactly:
  Case 1: dst==local  src==local   → import OVA + rename + bridge NIC + start
  Case 2: dst==local  src!=local   → wait for OVA (SCP'd by src) → import + start
  Case 3: dst!=local  src==local   → SCP OVA to dst cloudlet
  Fallback: broker offline + resource != critical → run locally

Changes from original:
  Python 3  (print → log, urllib2 → urllib, threading.Timer → sleep loop)
  VBoxManage → VMware  (via vm_manager.py)
  Hardcoded Gift123/root → environment variables
  Single persistent DB connection → reconnect per poll (avoids stale cursor)
  Runs FL training in a background thread alongside VM management
"""

from __future__ import annotations

import logging
import os
import threading
import time

import MySQLdb

from infrastructure.vmware.vm_manager import (
    list_vms, list_running_vms, vm_in_list,
    import_ova, rename_vm, set_bridged_nic, start_vm, migrate_ova,
    OVA_UPLOAD_PATH,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CLIENT] %(message)s",
)

# ── Config ────────────────────────────────────────────────────────────────────
DB_HOST        = os.environ.get("FMRAG_DB_HOST",       "localhost")
DB_USER        = os.environ.get("FMRAG_DB_USER",       "fmrag")
DB_PASS        = os.environ.get("FMRAG_DB_PASS",       "")
DB_NAME        = os.environ.get("FMRAG_DB_NAME",       "fmrag")
BROKER_IP      = os.environ.get("FMRAG_CENTRAL_IP",    "100.105.131.115"
                                                       "")
VM_USER        = os.environ.get("FMRAG_VM_USER",       "fmrag")
VM_PASS        = os.environ.get("FMRAG_VM_PASS",       "")
POLL_INTERVAL  = int(os.environ.get("FMRAG_POLL_INTERVAL", "30"))


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_db():
    return MySQLdb.connect(
        host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME
    )


def get_local_ip() -> str | None:
    db  = get_db()
    cur = db.cursor()
    n   = cur.execute("SELECT ip_address FROM resource")
    if n == 0:
        db.close()
        return None
    row = cur.fetchone()
    db.close()
    return row[0] if row else None


def broker_reachable() -> bool:
    return os.system(f"ping {BROKER_IP} -c 1 > /dev/null 2>&1") == 0


def remote_db(host: str):
    return MySQLdb.connect(
        host=host, user=DB_USER, passwd=DB_PASS, db=DB_NAME
    )


# ── error helper ──────────────────────────────────────────────────────────────

def set_error(cur, db, req_num: int, code: int, msg: str):
    log.error("[req %d] error %d: %s", req_num, code, msg)
    cur.execute(
        "UPDATE status SET error_status=%s, mes_to_user=%s "
        "WHERE request_number=%s",
        (code, msg, req_num)
    )
    db.commit()


# ── VM provisioning sequence (used in all three cases) ───────────────────────

def _provision(db, cur, req_num: int,
               vm_file_name: str, usrvm_reg_name: str,
               remote_cur=None, remote_db_=None) -> bool:
    """
    Import OVA → rename → bridge NIC → start.
    Mirrors lines 95-170 of vm.py exactly.
    remote_cur/remote_db_ used for Case 2 error reporting to source.
    """
    ec = cur if remote_cur is None else remote_cur
    ed = db  if remote_db_ is None else remote_db_

    ova_path = os.path.join(OVA_UPLOAD_PATH, vm_file_name)
    if not os.path.isfile(ova_path):
        set_error(ec, ed, req_num, 2, "Waiting for file offload")
        return False

    # ── import ────────────────────────────────────────────────────────────────
    vm_list = list_vms()
    if vm_in_list(usrvm_reg_name, vm_list):
        log.info("[req %d] VM already imported", req_num)
    else:
        rc = import_ova(vm_file_name, usrvm_reg_name)
        if rc != 0:
            set_error(ec, ed, req_num, 1, "import failed, please re-upload the file")
            return False
        log.info("[req %d] VM imported successfully", req_num)

    # ── rename to system name ─────────────────────────────────────────────────
    run_vm_list = list_running_vms()
    if vm_in_list(usrvm_reg_name, run_vm_list):
        log.info("[req %d] VM already running", req_num)
        return True

    sysvm_reg_name = vm_file_name.split("_", 1)[0]

    rc = rename_vm(usrvm_reg_name, sysvm_reg_name)
    if rc != 0:
        set_error(ec, ed, req_num, 4, "Server error (004)")
        return False

    new_vm_list = list_vms()
    if not vm_in_list(sysvm_reg_name, new_vm_list):
        set_error(ec, ed, req_num, 3, "Server error (003)")
        return False

    # ── bridged NIC ───────────────────────────────────────────────────────────
    rc = set_bridged_nic(sysvm_reg_name)
    if rc != 0:
        set_error(ec, ed, req_num, 5, "Server error (005)")
        return False

    # ── start VM ──────────────────────────────────────────────────────────────
    rc = start_vm(sysvm_reg_name)
    if rc != 0:
        set_error(ec, ed, req_num, 7, "Server error (007)")
        return False

    run_vm_list = list_running_vms()
    if vm_in_list(sysvm_reg_name, run_vm_list):
        log.info("[req %d] VM started successfully", req_num)
        cur.execute(
            "UPDATE status SET request_status='complete' "
            "WHERE request_number=%s", (req_num,)
        )
        db.commit()
        return True
    else:
        set_error(ec, ed, req_num, 6, "Server error (006)")
        return False


# ── main poll loop ────────────────────────────────────────────────────────────

def update():
    """
    Direct Python 3 translation of vm.py update() function.
    Runs every POLL_INTERVAL seconds.
    """
    local_ip = get_local_ip()
    if not local_ip:
        log.warning("Local IP not in resource table — is this node registered?")
        return

    db  = get_db()
    cur = db.cursor()

    n = cur.execute("SELECT * FROM status")
    if n == 0:
        log.debug("No requests in status table")
        db.close()
        return

    status_results = cur.fetchall()

    for row in status_results:
        # column order from schema:
        # id, request_number, ip_address, request_status, decision_status,
        # offload_status, migration_status, error_status, mes_to_user
        req_num          = row[1]
        request_status   = row[3]
        decision_status  = row[4]
        offload_status   = row[5]
        migration_status = row[6]

        # ── Main case: decision received ─────────────────────────────────────
        if (request_status  == "incomplete" and
            decision_status == "complete"   and
            offload_status  == "complete"):

            n = cur.execute(
                "SELECT src_cl_ip, dst_cl_ip FROM decision "
                "WHERE request_number=%s", (req_num,)
            )
            if n == 0:
                continue
            src_cl_ip, dst_cl_ip = cur.fetchone()

            cur.execute(
                "SELECT user, vm_name, vm_file_name FROM user_request "
                "WHERE request_number=%s", (req_num,)
            )
            req_row = cur.fetchone()
            if not req_row:
                continue
            user_name, usrvm_reg_name, vm_file_name = req_row

            # ── Case 1: local execution ───────────────────────────────────────
            if dst_cl_ip == local_ip and src_cl_ip == local_ip:
                log.info("[req %d] Case 1: local execution", req_num)
                _provision(db, cur, req_num, vm_file_name, usrvm_reg_name)

            # ── Case 2: incoming migration ────────────────────────────────────
            elif dst_cl_ip == local_ip and src_cl_ip != local_ip:
                log.info("[req %d] Case 2: incoming migration from %s",
                         req_num, src_cl_ip)
                try:
                    rdb = remote_db(src_cl_ip)
                    rc_ = rdb.cursor()
                    rc_.execute(
                        "SELECT user, vm_file_name, vm_name FROM user_request "
                        "WHERE request_number=%s", (req_num,)
                    )
                    r = rc_.fetchone()
                    if r:
                        _, vm_file_name, usrvm_reg_name = r
                    rdb.close()
                except MySQLdb.Error as e:
                    log.error("[req %d] Cannot reach src DB %s: %s",
                              req_num, src_cl_ip, e)
                _provision(db, cur, req_num, vm_file_name, usrvm_reg_name)

            # ── Case 3: outgoing migration ────────────────────────────────────
            elif dst_cl_ip != local_ip and src_cl_ip == local_ip:
                log.info("[req %d] Case 3: outgoing migration → %s",
                         req_num, dst_cl_ip)
                if migration_status == "complete":
                    log.info("[req %d] Already migrated", req_num)
                    continue
                rc = migrate_ova(vm_file_name, dst_cl_ip, VM_USER, VM_PASS)
                if rc == 0:
                    cur.execute(
                        "UPDATE status SET migration_status='complete' "
                        "WHERE request_number=%s", (req_num,)
                    )
                    db.commit()

        # ── Broker offline fallback ───────────────────────────────────────────
        elif (request_status  == "incomplete" and
              decision_status == "pending"    and
              offload_status  == "complete"):

            if broker_reachable():
                log.debug("[req %d] Waiting for broker decision", req_num)
                set_error(cur, db, req_num, 11, "Request in process")
                continue

            log.warning("[req %d] Broker offline — checking local resources",
                        req_num)
            cur.execute(
                "SELECT resource_level FROM decision_parameters LIMIT 1"
            )
            lvl_row = cur.fetchone()
            if not lvl_row or lvl_row[0] == "critical":
                log.warning("[req %d] Resources critical — cannot run locally",
                            req_num)
                set_error(cur, db, req_num, 10,
                          "Not enough resources available")
                continue

            cur.execute(
                "SELECT user, vm_file_name, vm_name FROM user_request "
                "WHERE request_number=%s", (req_num,)
            )
            req_row = cur.fetchone()
            if not req_row:
                continue
            user_name, vm_file_name, usrvm_reg_name = req_row
            log.info("[req %d] Running locally (broker offline)", req_num)
            _provision(db, cur, req_num, vm_file_name, usrvm_reg_name)

    db.close()


# ── FL training background thread ─────────────────────────────────────────────

def _fl_thread(config_path: str):
    log.info("FL training thread starting (config=%s)", config_path)
    try:
        import sys
        sys.argv = ["main_client.py", "--config", config_path]
        from main_client import main
        main()
    except Exception as e:
        log.error("FL training thread crashed: %s", e, exc_info=True)


# ── entry point ───────────────────────────────────────────────────────────────

def run(fl_config: str | None = None):
    log.info("FMRAG client daemon starting")
    log.info("  DB broker: %s", BROKER_IP)
    log.info("  VM store:  %s", os.environ.get("FMRAG_VM_STORE", "/var/fmrag/vms"))

    # Start resource reporter — pushes real psutil stats to local + central MySQL
    from infrastructure.client_node.resource_reporter import run as reporter_run
    reporter_thread = threading.Thread(
        target=reporter_run, daemon=True, name="fmrag-reporter"
    )
    reporter_thread.start()
    log.info("Resource reporter started — pushing live stats to central every %ds",
             int(os.environ.get("FMRAG_POLL_INTERVAL", "30")))

    if fl_config:
        t = threading.Thread(target=_fl_thread, args=(fl_config,),
                             daemon=True, name="fmrag-fl")
        t.start()
        log.info("FL training thread started")

    # Mirror vm.py: threading.Timer(30.0, update).start() in a loop
    while True:
        try:
            update()
        except MySQLdb.Error as e:
            log.error("DB error: %s", e)
        except Exception as e:
            log.error("Unexpected error: %s", e, exc_info=True)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fl-config", default=None)
    args = p.parse_args()
    run(fl_config=args.fl_config)
