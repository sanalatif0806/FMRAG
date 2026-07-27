"""
FMRAG Asynchronous FL Server (FedAsync-style)
---------------------------------------------
Replaces the synchronous round-based loop with continuous, asynchronous
aggregation that supports ANY number of clients with no fixed cohort.

Key differences from the synchronous server (kept as fl_server_sync_backup.py):
  - The server NEVER waits for a fixed number of clients. It accepts
    connections continuously; each client is handled in its own thread.
  - The global model is updated the MOMENT any single client's delta
    arrives — not after a batch of N.
  - Updates are staleness-weighted (FedAsync, Xie et al. 2019):
        global <- (1 - a_eff) * global + a_eff * client_model
        a_eff  = base_alpha * staleness_fn(staleness)
    where staleness = current_global_version - version_the_client_pulled.
    Stale updates (client trained on an old global) are down-weighted so
    they don't drag the model backward.
  - Clients pull the current global (tagged with a version number), train,
    and push back their NEW ABSOLUTE weights (global_pulled + local_delta).
    The server merges that into whatever the global is NOW.

Protocol (unchanged wire format, one field added):
  server -> client : {"lora", "kgc", "version"}     (version is new)
  client -> server : {"lora", "n_samples", "kgc", "kgc_eval",
                      "hypothesis_stats", "base_version"}   (base_version new)

Backwards-compatible: a client that doesn't send base_version is treated as
staleness 0 (no down-weighting), so older clients still work.

Config / env:
  FMRAG_FL_PORT            (default 8080)
  FMRAG_ASYNC_ALPHA        base mixing rate a (default 0.6)
  FMRAG_ASYNC_STALENESS    staleness function: "poly" | "hinge" | "const"
                           (default "poly", FedAsync's recommended form)
  FMRAG_ASYNC_STALENESS_B  parameter b for the staleness function (default 0.5)
  FMRAG_MAX_UPDATES        stop after this many total updates (default 500;
                           replaces the old fixed MAX_ROUNDS notion)
  FMRAG_MIN_UPDATES_LOG    log a checkpoint every N updates (default 1)
"""

import socket
import threading
import pickle
import struct
import logging
import copy
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")
log = logging.getLogger(__name__)

HOST = ""
PORT = int(os.environ.get("FMRAG_FL_PORT", "8080"))

# Async hyperparameters
BASE_ALPHA      = float(os.environ.get("FMRAG_ASYNC_ALPHA", "0.6"))
STALENESS_FN    = os.environ.get("FMRAG_ASYNC_STALENESS", "poly")
STALENESS_B     = float(os.environ.get("FMRAG_ASYNC_STALENESS_B", "0.5"))
MAX_UPDATES     = int(os.environ.get("FMRAG_MAX_UPDATES", "500"))


# ── weight I/O helpers (unchanged) ────────────────────────────────────────────

def send_weights(conn: socket.socket, state_dict: dict):
    payload = pickle.dumps(state_dict)
    conn.sendall(struct.pack(">I", len(payload)))
    conn.sendall(payload)


def recv_weights(conn: socket.socket) -> dict:
    raw_len = _recv_exact(conn, 4)
    msg_len = struct.unpack(">I", raw_len)[0]
    return pickle.loads(_recv_exact(conn, msg_len))


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed mid-receive")
        buf += chunk
    return buf


# ── staleness weighting (FedAsync) ────────────────────────────────────────────

def staleness_weight(staleness: int) -> float:
    """
    Returns a multiplier in (0, 1] that shrinks as staleness grows.
    staleness = how many global updates happened since this client pulled.

    "poly"  : 1 / (1 + b*staleness)         (FedAsync polynomial, default)
    "hinge" : 1 if staleness<=b else 1/(b*(staleness-b)+1)
    "const" : 1 (no staleness handling — pure incremental merge)
    """
    s = max(0, staleness)
    if STALENESS_FN == "const":
        return 1.0
    if STALENESS_FN == "hinge":
        if s <= STALENESS_B:
            return 1.0
        return 1.0 / (STALENESS_B * (s - STALENESS_B) + 1.0)
    # default: polynomial
    return 1.0 / (1.0 + STALENESS_B * s)


# ── global model state (thread-safe) ──────────────────────────────────────────

class GlobalModel:
    """
    Holds the global LoRA (+ optional KGC) state, a monotonically increasing
    version counter, and a lock. Every merge bumps the version. Clients record
    which version they pulled so staleness can be computed on push.
    """

    def __init__(self, lora: dict, kgc: dict | None):
        self.lora = lora
        self.kgc = kgc
        self.version = 0
        self.lock = threading.Lock()
        self.updates_done = 0

    def snapshot(self):
        """Thread-safe copy of the current global + its version."""
        with self.lock:
            return (copy.deepcopy(self.lora),
                    copy.deepcopy(self.kgc) if self.kgc is not None else None,
                    self.version)

    def merge(self, client_lora: dict, client_kgc: dict | None,
              base_version: int, n_samples: int) -> tuple[int, float]:
        """
        FedAsync merge of one client's ABSOLUTE weights into the live global.
        Returns (new_version, effective_alpha).
        """
        with self.lock:
            staleness = self.version - base_version
            a_eff = BASE_ALPHA * staleness_weight(staleness)
            # clamp for safety
            a_eff = max(0.0, min(1.0, a_eff))

            for k in self.lora.keys():
                if k in client_lora:
                    self.lora[k] = (1.0 - a_eff) * self.lora[k] + a_eff * client_lora[k]

            if self.kgc is not None and client_kgc is not None:
                for k in self.kgc.keys():
                    if k in client_kgc:
                        self.kgc[k] = (1.0 - a_eff) * self.kgc[k] + a_eff * client_kgc[k]

            self.version += 1
            self.updates_done += 1
            return self.version, a_eff


# ── per-client thread ─────────────────────────────────────────────────────────

class AsyncClientHandler(threading.Thread):
    """
    One thread per connected client. Pulls current global, sends it, waits for
    the client's trained absolute weights, and merges them immediately.
    A single client connection serves ONE pull+train+push cycle; the client
    reconnects for its next cycle (keeps the protocol simple and stateless
    server-side, while still being fully asynchronous across clients).
    """

    def __init__(self, conn, addr, gmodel: GlobalModel, on_update):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.gmodel = gmodel
        self.on_update = on_update    # callback(version, a_eff, payload) for logging

    def run(self):
        log.info("Client connected: %s", self.addr)
        try:
            lora, kgc, version = self.gmodel.snapshot()
            send_weights(self.conn, {"lora": lora, "kgc": kgc, "version": version})
            log.info("Sent global v%d to %s  (KGC: %s)",
                     version, self.addr, "on" if kgc is not None else "off")

            payload = recv_weights(self.conn)
            client_lora = payload["lora"]
            n           = payload.get("n_samples", 1)
            client_kgc  = payload.get("kgc")
            base_ver    = payload.get("base_version", version)  # back-compat

            new_version, a_eff = self.gmodel.merge(
                client_lora, client_kgc, base_ver, n
            )
            staleness = new_version - 1 - base_ver
            log.info("Merged update from %s  (n=%d, base=v%d, staleness=%d, "
                     "a_eff=%.3f) -> global now v%d",
                     self.addr, n, base_ver, staleness, a_eff, new_version)

            self.on_update(new_version, a_eff, payload)
        except Exception as e:
            log.error("Error with client %s: %s", self.addr, e)
        finally:
            self.conn.close()


# ── initial state builders (unchanged from sync server) ───────────────────────

def load_initial_lora() -> dict:
    import torch
    from collections import OrderedDict
    lora_keys = []
    for layer in [10, 11]:
        for proj in ["query", "value"]:
            lora_keys.append(
                f"base_model.model.bert.encoder.layer.{layer}"
                f".attention.self.{proj}.lora_A.weight")
            lora_keys.append(
                f"base_model.model.bert.encoder.layer.{layer}"
                f".attention.self.{proj}.lora_B.weight")
    state = OrderedDict()
    for k in lora_keys:
        state[k] = torch.zeros(8, 768) if "lora_A" in k else torch.zeros(768, 8)
    return state


def load_initial_kgc(num_nodes: int, num_rels: int, dim: int = 64) -> dict:
    import torch
    return {
        "entity_emb.weight":   torch.zeros(num_nodes, dim),
        "relation_emb.weight": torch.zeros(num_rels, dim),
    }


# ── async server main loop ────────────────────────────────────────────────────

def run(num_nodes: int = 0, num_rels: int = 0):
    global_lora = load_initial_lora()
    global_kgc  = load_initial_kgc(num_nodes, num_rels) if num_nodes > 0 else None
    gmodel = GlobalModel(global_lora, global_kgc)

    # running aggregate of eval / hypothesis stats across updates
    stats_lock = threading.Lock()
    agg = {"kgc_evals": [], "hyp_stats": []}

    def on_update(version, a_eff, payload):
        kgc_eval  = payload.get("kgc_eval")
        hyp_stats = payload.get("hypothesis_stats")
        with stats_lock:
            if kgc_eval is not None:
                agg["kgc_evals"].append(kgc_eval)
                log.info("  v%d KGC eval — MRR=%.3f Hits@1=%.3f Hits@10=%.3f",
                         version, kgc_eval.get("MRR", 0),
                         kgc_eval.get("Hits@1", 0), kgc_eval.get("Hits@10", 0))
            if hyp_stats is not None:
                agg["hyp_stats"].append(hyp_stats)
                log.info("  v%d hypotheses — %d patients, avg/patient=%.2f, "
                         "avg top score=%.3f", version,
                         hyp_stats.get("patients_processed", 0),
                         hyp_stats.get("avg_hypotheses_per_patient", 0),
                         hyp_stats.get("avg_top_score", 0))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(32)
    srv.settimeout(1.0)   # so we can periodically check the stop condition

    log.info("ASYNC FL server listening on port %d  (KGC: %s | alpha=%.2f | "
             "staleness=%s b=%.2f | max_updates=%s)",
             PORT, "on" if global_kgc is not None else "off",
             BASE_ALPHA, STALENESS_FN, STALENESS_B,
             "UNLIMITED" if MAX_UPDATES == 0 else MAX_UPDATES)
    log.info("Accepting ANY number of clients asynchronously — no fixed cohort. "
             "%s", "Runs INDEFINITELY (Ctrl+C to stop)." if MAX_UPDATES == 0
             else "Stops after %d updates." % MAX_UPDATES)

    threads = []
    try:
        # MAX_UPDATES == 0 means run INDEFINITELY (server always listening,
        # never self-terminates — clients join/leave freely, one client
        # finishing never stops the server or affects the others).
        while MAX_UPDATES == 0 or gmodel.updates_done < MAX_UPDATES:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            h = AsyncClientHandler(conn, addr, gmodel, on_update)
            h.start()
            threads.append(h)
            # reap finished threads
            threads = [t for t in threads if t.is_alive()]
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")
    finally:
        # let in-flight merges finish
        for t in threads:
            t.join(timeout=5.0)
        srv.close()

    log.info("Async training stopped after %d global updates (final version v%d)",
             gmodel.updates_done, gmodel.version)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="FMRAG Async FL Server")
    parser.add_argument("--config", default="config/central_config.yaml")
    args = parser.parse_args()

    num_nodes, num_rels = 0, 0
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        kgc_cfg = cfg.get("kgc", {})
        if kgc_cfg.get("enabled", False):
            num_nodes = kgc_cfg.get("num_nodes", 0)
            num_rels  = kgc_cfg.get("num_rels", 0)
        if num_nodes == 0:
            log.warning("kgc.enabled=true but num_nodes=0 — KGC aggregation OFF")
    except FileNotFoundError:
        log.warning("Config %s not found — KGC aggregation OFF", args.config)

    run(num_nodes=num_nodes, num_rels=num_rels)
