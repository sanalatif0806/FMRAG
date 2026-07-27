"""
infrastructure/scripts/migrate_db.py
--------------------------------------
One-time migration: copies data from the old Client_ver8 MySQL schema
(database: test, tables: user_request / status / decision / resource)
into the new FMRAG schema (database: fmrag).

Run ONCE on the broker machine after setup_node.sh has created the
new fmrag database.

Usage:
    python -m infrastructure.scripts.migrate_db \
        --old-host localhost \
        --old-db   test \
        --old-user root \
        --old-pass Gift123 \
        --new-host localhost \
        --new-db   fmrag \
        --new-user fmrag \
        --new-pass FmragSecure2024!
"""

import argparse
import logging
import MySQLdb

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [MIGRATE] %(message)s")


def conn(host, user, passwd, db):
    return MySQLdb.connect(host=host, user=user, passwd=passwd, db=db)


def migrate(old_cfg: dict, new_cfg: dict):
    old = conn(**old_cfg)
    new = conn(**new_cfg)
    oc  = old.cursor()
    nc  = new.cursor()

    # ── 1. resource ──────────────────────────────────────────────────────────
    log.info("Migrating resource table...")
    oc.execute("SELECT ip_address, cloudlet_name, last_updated, "
               "cpu_cores, cpu_avload, memory_total, memory_free, "
               "disk_total, disk_free FROM resource")
    for row in oc.fetchall():
        nc.execute("""
            INSERT IGNORE INTO resource
              (ip_address, cloudlet_name, last_updated,
               cpu_cores, cpu_avload, memory_total, memory_free,
               disk_total, disk_free)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, row)
    log.info("  resource: %d rows", oc.rowcount)

    # ── 2. user_request ───────────────────────────────────────────────────────
    log.info("Migrating user_request table...")
    oc.execute("SELECT request_number, request_dt, user, vm_name, vm_ip, "
               "vm_cpu, vm_storage, vm_memory, vm_user, vm_pass, "
               "vm_file_name FROM user_request")
    for row in oc.fetchall():
        # Old schema has no ip_address in user_request; use placeholder
        nc.execute("""
            INSERT IGNORE INTO user_request
              (request_number, request_dt, ip_address, user, vm_name,
               vm_ip, vm_cpu, vm_storage, vm_memory, vm_user,
               vm_file_name)
            VALUES (%s,%s,'0.0.0.0',%s,%s,%s,%s,%s,%s,%s,%s)
        """, row[:1] + row[1:2] + row[2:])
    log.info("  user_request rows attempted: %d", oc.rowcount)

    # ── 3. status ─────────────────────────────────────────────────────────────
    log.info("Migrating status table...")
    oc.execute("SELECT request_number, request_status, decision_status, "
               "offload_status, migration_status, error_status, "
               "mes_to_user FROM status")
    for row in oc.fetchall():
        nc.execute("""
            INSERT IGNORE INTO status
              (request_number, request_status, decision_status,
               offload_status, migration_status, error_status, mes_to_user)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, row)
    log.info("  status rows attempted: %d", oc.rowcount)

    # ── 4. decision ───────────────────────────────────────────────────────────
    log.info("Migrating decision table...")
    oc.execute("SELECT request_number, src_cl_ip, dst_cl_ip "
               "FROM decision")
    for row in oc.fetchall():
        nc.execute("""
            INSERT IGNORE INTO decision
              (request_number, src_cl_ip, dst_cl_ip, decision_sent)
            VALUES (%s,%s,%s,'yes')
        """, row)
    log.info("  decision rows attempted: %d", oc.rowcount)

    new.commit()
    old.close()
    new.close()
    log.info("Migration complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old-host", default="localhost")
    p.add_argument("--old-db",   default="test")
    p.add_argument("--old-user", default="root")
    p.add_argument("--old-pass", default="Gift123")
    p.add_argument("--new-host", default="localhost")
    p.add_argument("--new-db",   default="fmrag")
    p.add_argument("--new-user", default="fmrag")
    p.add_argument("--new-pass", default="FmragSecure2024!")
    args = p.parse_args()

    migrate(
        old_cfg=dict(host=args.old_host, user=args.old_user,
                     passwd=args.old_pass, db=args.old_db),
        new_cfg=dict(host=args.new_host, user=args.new_user,
                     passwd=args.new_pass, db=args.new_db),
    )


if __name__ == "__main__":
    main()
