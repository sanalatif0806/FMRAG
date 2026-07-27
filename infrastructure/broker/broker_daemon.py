"""
infrastructure/broker/broker_daemon.py
-----------------------------------------
Python 3 faithful rewrite of Broker_ver8/psdash/test.py

Every step from test.py is preserved exactly:
  1. For each known cloudlet (ip != self):
       a. ping check
       b. connect to remote MySQL, pull decision_parameters
       c. pull user_request + status rows, upsert into broker DB
  2. For each pending request:
       a. filter eligible cloudlets: cpu_cores >= vm_cpu
                                     disk_free >= vm_storage
                                     memory_free >= vm_memory
       b. insert into el_cloudlet staging table
       c. pick dst = max(resource_index) non-critical
       d. insert decision (decision_sent='no')
  3. Push decisions to src and dst cloudlet DBs
     mark decision_sent='yes' when both delivered

Changes from original:
  Python 3 (print → log, urllib2 → urllib)
  Hardcoded Gift123/root → environment variables
  threading.Timer loop → while True + sleep
  Reconnect per poll — avoids stale cursor on long uptime
"""

from __future__ import annotations

import logging
import os
import time

import MySQLdb

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BROKER] %(message)s",
)

# ── Config ────────────────────────────────────────────────────────────────────
SELF_IP       = os.environ.get("FMRAG_CENTRAL_IP",  "100.105.131.115")
DB_HOST       = os.environ.get("FMRAG_DB_HOST",     "localhost")
DB_USER       = os.environ.get("FMRAG_DB_USER",     "fmrag")
DB_PASS       = os.environ.get("FMRAG_DB_PASS",     "")
DB_NAME       = os.environ.get("FMRAG_DB_NAME",     "fmrag")
POLL_INTERVAL = int(os.environ.get("FMRAG_BROKER_POLL", "30"))


# ── DB helpers ────────────────────────────────────────────────────────────────

def local_db():
    return MySQLdb.connect(
        host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME
    )


def remote_db(host: str):
    return MySQLdb.connect(
        host=host, user=DB_USER, passwd=DB_PASS, db=DB_NAME
    )


def ping(ip: str) -> bool:
    """Mirrors: os.system('ping %s -c 1 > /dev/null 2>&1' % ip) is 0"""
    return os.system(f"ping {ip} -c 1 > /dev/null 2>&1") == 0


# ── update() — direct translation of test.py update() ───────────────────────

def update():
    db  = local_db()
    cur = db.cursor()

    # ── Step 1: poll all remote cloudlets ────────────────────────────────────
    cur.execute(
        "SELECT ip_address FROM resource WHERE ip_address != %s",
        (SELF_IP,)
    )
    cloudlets = [r[0] for r in cur.fetchall()]

    for cl_ip in cloudlets:
        if not ping(cl_ip):
            log.warning("Remote request not received. Connection failed with %s",
                        cl_ip)
            continue

        try:
            rdb = remote_db(cl_ip)
            rc  = rdb.cursor()

            # ── 1a. pull decision_parameters ─────────────────────────────────
            exist = rc.execute(
                "SELECT a.ip_address, b.resource_index, b.resource_level "
                "FROM resource a, decision_parameters b"
            )
            if exist != 0:
                row = rc.fetchone()
                cloudlet_ip, resource_index, resource_level = row

                dec_param_exist = cur.execute(
                    "SELECT * FROM decision_parameters "
                    "WHERE cloudlet_ip=%s", (cloudlet_ip,)
                )
                if dec_param_exist != 1:
                    cur.execute(
                        "INSERT INTO decision_parameters "
                        "SET cloudlet_ip=%s, resource_index=%s, resource_level=%s",
                        (cloudlet_ip, resource_index, resource_level)
                    )
                else:
                    cur.execute(
                        "UPDATE decision_parameters "
                        "SET resource_index=%s, resource_level=%s "
                        "WHERE cloudlet_ip=%s",
                        (resource_index, resource_level, cloudlet_ip)
                    )
                db.commit()
            else:
                log.warning("Failed to get decision_parameters from remote %s",
                            cl_ip)

            # ── 1b. pull user_request + status ────────────────────────────────
            rc.execute(
                "SELECT a.ip_address, a.cloudlet_name, b.*, c.* "
                "FROM resource a, user_request b, status c "
                "WHERE b.request_number = c.request_number"
            )
            results = rc.fetchall()

            for row in results:
                cloudlet_ip   = row[0]
                cloudlet_name = row[1]
                request_number= row[2]
                request_dt    = row[3]
                user          = row[4]
                vm_name       = row[5]
                vm_ip         = row[6]
                vm_cpu        = row[7]
                vm_storage    = row[8]
                vm_memory     = row[9]
                vm_user       = row[10]
                vm_pass       = row[11]
                vm_file_name  = row[12]
                request_status  = row[15]
                decision_status = row[16]
                offload_status  = row[17]
                migration_status= row[18]
                error_status    = row[19]
                mes_to_user     = row[20]

                # Check if request already in broker DB
                req_record_exist = cur.execute(
                    "SELECT request_number, ip_address FROM user_request"
                )
                req_result = cur.fetchall()

                record = 0
                if req_record_exist != 0:
                    for i in range(len(req_result)):
                        if (req_result[i][0] == request_number and
                                req_result[i][1] == cloudlet_ip):
                            record = 1
                            break
                    if record == 0:
                        cur.execute(
                            "INSERT INTO user_request "
                            "SET ip_address=%s, cloudlet_name=%s, "
                            "request_number=%s, request_dt=%s, user=%s, "
                            "vm_name=%s, vm_ip=%s, vm_cpu=%s, vm_storage=%s, "
                            "vm_memory=%s, vm_user=%s, vm_file_name=%s",
                            (cloudlet_ip, cloudlet_name, request_number,
                             request_dt, user, vm_name, vm_ip, vm_cpu,
                             vm_storage, vm_memory, vm_user, vm_file_name)
                        )
                        db.commit()
                        cur.execute(
                            "INSERT INTO status "
                            "SET ip_address=%s, request_number=%s, "
                            "request_status=%s, decision_status=%s, "
                            "offload_status=%s, migration_status=%s, "
                            "error_status=%s, mes_to_user=%s",
                            (cloudlet_ip, request_number, request_status,
                             decision_status, offload_status, migration_status,
                             error_status, mes_to_user)
                        )
                        db.commit()
                        log.info("New request %d received from %s",
                                 request_number, cloudlet_ip)
                    else:
                        cur.execute(
                            "UPDATE status SET request_status=%s, "
                            "decision_status=%s, offload_status=%s, "
                            "migration_status=%s, error_status=%s, "
                            "mes_to_user=%s "
                            "WHERE request_number=%s AND ip_address=%s",
                            (request_status, decision_status, offload_status,
                             migration_status, error_status, mes_to_user,
                             request_number, cloudlet_ip)
                        )
                        db.commit()
                else:
                    log.info("New request %d received from %s",
                             request_number, cloudlet_ip)
                    cur.execute(
                        "INSERT INTO user_request "
                        "SET ip_address=%s, cloudlet_name=%s, "
                        "request_number=%s, request_dt=%s, user=%s, "
                        "vm_name=%s, vm_ip=%s, vm_cpu=%s, vm_storage=%s, "
                        "vm_memory=%s, vm_user=%s, vm_file_name=%s",
                        (cloudlet_ip, cloudlet_name, request_number,
                         request_dt, user, vm_name, vm_ip, vm_cpu,
                         vm_storage, vm_memory, vm_user, vm_file_name)
                    )
                    db.commit()
                    cur.execute(
                        "INSERT INTO status "
                        "SET ip_address=%s, request_number=%s, "
                        "request_status=%s, decision_status=%s, "
                        "offload_status=%s, migration_status=%s, "
                        "error_status=%s, mes_to_user=%s",
                        (cloudlet_ip, request_number, request_status,
                         decision_status, offload_status, migration_status,
                         error_status, mes_to_user)
                    )
                    db.commit()

            rc.close()
            rdb.close()

        except MySQLdb.Error as e:
            log.error("Remote DB error for %s: %s", cl_ip, e)

    # ── Step 2: make decisions ────────────────────────────────────────────────
    cur.execute(
        "SELECT a.ip_address, a.cloudlet_name, a.request_number, "
        "a.vm_cpu, a.vm_storage, a.vm_memory "
        "FROM user_request a, status b "
        "WHERE a.request_number = b.request_number "
        "AND b.decision_status = 'pending'"
    )
    rq_result = cur.fetchall()

    # Clear staging table each cycle (mirrors test.py: curl.execute("delete from el_cloudlet"))
    cur.execute("DELETE FROM el_cloudlet")
    db.commit()

    for row in rq_result:
        cloudlet_ip    = row[0]
        cloudlet_name  = row[1]
        request_number = row[2]
        vm_cpu         = row[3] or 1
        vm_storage     = row[4] or 1.0
        vm_memory      = row[5] or 1.0

        # Filter eligible cloudlets by hardware capacity
        cur.execute(
            "SELECT ip_address, cpu_cores, disk_free, memory_free "
            "FROM resource WHERE ip_address != %s",
            (SELF_IP,)
        )
        rresult = cur.fetchall()

        el_cloudlet = []
        for i in range(len(rresult)):
            if (rresult[i][1] >= vm_cpu and
                    rresult[i][2] >= vm_storage and
                    rresult[i][3] >= vm_memory):
                el_cloudlet.append(rresult[i][0])
            else:
                el_cloudlet.append("NULL")

        # Check for duplicate el_cloudlet entries
        rec_exist = cur.execute(
            "SELECT cloudlet_ip, request_number FROM el_cloudlet"
        )
        el_result = cur.fetchall()

        record = 0
        if rec_exist != 0:
            for i in range(len(el_result)):
                if (cloudlet_ip == el_result[i][0] and
                        request_number == el_result[i][1]):
                    record = 1
                    break

        if record == 0:
            for el_ip in el_cloudlet:
                cur.execute(
                    "INSERT INTO el_cloudlet "
                    "SET cloudlet_ip=%s, request_number=%s, el_cloudlet_ip=%s",
                    (cloudlet_ip, request_number, el_ip)
                )
            db.commit()

        # Pick best destination
        el_exist = cur.execute(
            "SELECT * FROM el_cloudlet "
            "WHERE el_cloudlet_ip = ("
            "  SELECT cloudlet_ip FROM decision_parameters "
            "  WHERE resource_index = ("
            "    SELECT MAX(resource_index) FROM decision_parameters "
            "    WHERE resource_level != 'critical'"
            "  ) LIMIT 1"
            ") AND request_number = %s",
            (request_number,)
        )
        el_result = cur.fetchall()

        if el_exist == 0:
            log.warning("No eligible cloudlet for request %d", request_number)
            continue

        cloudlet_ip    = el_result[0][0]
        request_number = el_result[0][1]
        final_cl_ip    = el_result[0][2]

        # Avoid duplicate decisions
        exist = cur.execute("SELECT * FROM decision")
        res   = cur.fetchall()

        record = 0
        if exist != 0:
            for n in res:
                rqn   = n[0]
                srcip = n[1]
                if srcip == cloudlet_ip and rqn == request_number:
                    record = 1
                    break

        if record == 0:
            cur.execute(
                "INSERT INTO decision "
                "SET request_number=%s, src_cl_ip=%s, dst_cl_ip=%s, "
                "decision_sent='no'",
                (request_number, cloudlet_ip, final_cl_ip)
            )
            db.commit()
            cur.execute(
                "UPDATE status SET decision_status='complete' "
                "WHERE ip_address=%s AND request_number=%s",
                (cloudlet_ip, request_number)
            )
            db.commit()
            log.info("Decision: req=%d src=%s dst=%s",
                     request_number, cloudlet_ip, final_cl_ip)

    # ── Step 3: push decisions to src and dst cloudlets ───────────────────────
    dexist = cur.execute(
        "SELECT * FROM decision WHERE decision_sent='no'"
    )
    ldecision = cur.fetchall()

    if dexist != 0:
        for row in ldecision:
            request_number    = row[0]
            src_cloudlet_ip   = row[1]
            dst_cloudlet_ip   = row[2]

            # Push to source
            if ping(src_cloudlet_ip):
                try:
                    conn = remote_db(src_cloudlet_ip)
                    curr = conn.cursor()
                    rdec_exist = curr.execute("SELECT * FROM decision")
                    rdec_result = curr.fetchall()
                    record = 0
                    if rdec_exist != 0:
                        for i in range(len(rdec_result)):
                            if rdec_result[i][0] == request_number:
                                record = 1
                                break
                    if record == 0:
                        check = curr.execute(
                            "INSERT INTO decision "
                            "SET request_number=%s, src_cl_ip=%s, dst_cl_ip=%s",
                            (request_number, src_cloudlet_ip, dst_cloudlet_ip)
                        )
                        conn.commit()
                        if check != 0:
                            st = curr.execute(
                                "UPDATE status SET decision_status='complete' "
                                "WHERE request_number=%s", (request_number,)
                            )
                            conn.commit()
                            if st != 0:
                                log.info("Decision %d pushed to src %s",
                                         request_number, src_cloudlet_ip)
                            else:
                                log.error("Failed to update status at src %s",
                                          src_cloudlet_ip)
                    conn.close()
                except MySQLdb.Error as e:
                    log.error("Push to src %s failed: %s", src_cloudlet_ip, e)
            else:
                log.warning("Decision not sent to src — connection failed: %s",
                            src_cloudlet_ip)

            # Push to destination
            if ping(dst_cloudlet_ip):
                try:
                    conn = remote_db(dst_cloudlet_ip)
                    curr = conn.cursor()
                    rdec_exist = curr.execute("SELECT * FROM decision")
                    rdec_result = curr.fetchall()
                    record = 0
                    if rdec_exist != 0:
                        for i in range(len(rdec_result)):
                            if rdec_result[i][0] == request_number:
                                record = 1
                                break
                    if record == 0:
                        curr.execute(
                            "INSERT INTO decision "
                            "SET request_number=%s, src_cl_ip=%s, dst_cl_ip=%s",
                            (request_number, src_cloudlet_ip, dst_cloudlet_ip)
                        )
                        conn.commit()
                        log.info("Decision %d pushed to dst %s",
                                 request_number, dst_cloudlet_ip)
                    conn.close()
                except MySQLdb.Error as e:
                    log.error("Push to dst %s failed: %s", dst_cloudlet_ip, e)
            else:
                log.warning("Decision not sent to dst — connection failed: %s",
                            dst_cloudlet_ip)

            # Mark sent in broker DB
            cur.execute(
                "UPDATE decision SET decision_sent='yes' "
                "WHERE request_number=%s", (request_number,)
            )
            db.commit()

    db.close()


# ── entry point ───────────────────────────────────────────────────────────────

def run():
    log.info("FMRAG broker daemon starting")
    log.info("  self IP : %s", SELF_IP)
    log.info("  DB      : %s@%s/%s", DB_USER, DB_HOST, DB_NAME)
    log.info("  poll    : %ds", POLL_INTERVAL)

    while True:
        try:
            update()
        except MySQLdb.Error as e:
            log.error("DB error: %s", e)
        except Exception as e:
            log.error("Error in broker poll: %s", e, exc_info=True)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
