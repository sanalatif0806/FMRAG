"""
FMRAG FL Server
---------------
Replaces legacy Server.py (Python 2, SCP-based model push).
- Holds the global BioBERT + LoRA adapter state
- Each round: broadcasts LoRA A/B deltas to all clients
- Receives trained LoRA deltas back, runs FedAvg
- Backbone weights never move — only LoRA matrices (~400 KB/round)
"""

import socket
import threading
import pickle
import struct
import logging
import copy
import os
from collections import OrderedDict
from queue import Queue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SERVER] %(message)s")
log = logging.getLogger(__name__)

HOST = ""
# Was hardcoded to 9996 with no override — fl_client.py already supports
# FMRAG_FL_PORT, but the server had no matching env-configurable option.
PORT = int(os.environ.get("FMRAG_FL_PORT", "8080"))
MIN_CLIENTS = int(os.environ.get("FMRAG_MIN_CLIENTS", "1"))  # reads env now
MAX_ROUNDS   = 50


# ── weight I/O helpers ────────────────────────────────────────────────────────

def send_weights(conn: socket.socket, state_dict: dict):
    """Pickle + length-prefix a state_dict over an open socket."""
    payload = pickle.dumps(state_dict)
    conn.sendall(struct.pack(">I", len(payload)))
    conn.sendall(payload)


def recv_weights(conn: socket.socket) -> dict:
    """Receive a length-prefixed pickle from the client."""
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


# ── FedAvg ────────────────────────────────────────────────────────────────────

def fedavg(global_state: dict, client_deltas: list[dict], client_sizes: list[int]) -> dict:
    """
    Weighted FedAvg over LoRA adapter deltas.
    client_sizes: number of local training samples per client (for weighted avg).
    Only keys present in the first delta are aggregated (LoRA + BAT-GNN head).
    Backbone keys are never touched.
    """
    total = sum(client_sizes)
    averaged = copy.deepcopy(global_state)

    keys = list(client_deltas[0].keys())
    for k in keys:
        stacked = sum(
            delta[k] * (n / total)
            for delta, n in zip(client_deltas, client_sizes)
        )
        averaged[k] = stacked

    return averaged


# ── per-client thread ─────────────────────────────────────────────────────────

class ClientHandler(threading.Thread):
    def __init__(self, conn, addr, round_q: Queue, global_lora: dict, global_kgc: dict | None):
        super().__init__(daemon=True)
        self.conn = conn
        self.addr = addr
        self.round_q = round_q          # queue to push (lora_delta, kgc_delta, kgc_eval, n_samples) into
        self.global_lora = global_lora  # shared reference, read-only during round
        self.global_kgc = global_kgc    # shared reference, read-only during round; None if KGC disabled

    def run(self):
        log.info("Client connected: %s", self.addr)
        try:
            # 1. Send current global LoRA (+ KGC, if enabled) state to client
            send_weights(self.conn, {"lora": self.global_lora, "kgc": self.global_kgc})
            log.info("Sent global state to %s  (KGC: %s)",
                     self.addr, "on" if self.global_kgc is not None else "off")

            # 2. Wait for trained delta(s) + sample count + KGC eval back
            payload  = recv_weights(self.conn)   # {"lora": dict, "n_samples": int, "kgc": dict|None, "kgc_eval": dict|None, "hypothesis_stats": dict|None}
            lora_d   = payload["lora"]
            n        = payload["n_samples"]
            kgc_d    = payload.get("kgc")
            kgc_eval = payload.get("kgc_eval")
            hyp_stats = payload.get("hypothesis_stats")
            log.info("Received delta from %s  (n=%d)%s", self.addr, n,
                     "  + KGC delta" if kgc_d is not None else "")

            self.round_q.put((lora_d, kgc_d, kgc_eval, hyp_stats, n))
        except Exception as e:
            log.error("Error with client %s: %s", self.addr, e)
        finally:
            self.conn.close()


# ── main server loop ──────────────────────────────────────────────────────────

def load_initial_lora() -> dict:
    """
    Returns an empty LoRA state dict.
    In production: load from a checkpoint or initialise via model_setup.py.
    Keys follow the PEFT naming convention:
      base_model.model.bert.encoder.layer.{10|11}.attention.self.{query|value}.lora_{A|B}.weight
    """
    import torch
    lora_keys = []
    for layer in [10, 11]:
        for proj in ["query", "value"]:
            lora_keys.append(
                f"base_model.model.bert.encoder.layer.{layer}"
                f".attention.self.{proj}.lora_A.weight"
            )
            lora_keys.append(
                f"base_model.model.bert.encoder.layer.{layer}"
                f".attention.self.{proj}.lora_B.weight"
            )
    # rank=8, hidden=768
    state = OrderedDict()
    for k in lora_keys:
        if "lora_A" in k:
            state[k] = torch.zeros(8, 768)
        else:
            state[k] = torch.zeros(768, 8)
    return state


def load_initial_kgc(num_nodes: int, num_rels: int, dim: int = 64) -> dict:
    """
    Zero-initialised global KGC embedding table, same shape convention as
    KGCompletionModel in client/kgc/kg_completion.py. Pass num_nodes=0 (or
    set FMRAG_KGC_ENABLED=false on every client) to run the ablation with
    KGC disabled entirely — clients then send kgc=None and this is unused.
    """
    import torch
    return {
        "entity_emb.weight":   torch.zeros(num_nodes, dim),
        "relation_emb.weight": torch.zeros(num_rels, dim),
    }


def run(num_nodes: int = 0, num_rels: int = 0):
    """
    num_nodes/num_rels > 0 enables federated KGC aggregation alongside LoRA.
    Leave at 0 to run the original LoRA-only FL loop (KGC ablation: off).
    """
    global_lora = load_initial_lora()
    global_kgc  = load_initial_kgc(num_nodes, num_rels) if num_nodes > 0 else None

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(10)
    log.info("FL server listening on port %d  (KGC aggregation: %s)",
             PORT, "on" if global_kgc is not None else "off")

    for fl_round in range(1, MAX_ROUNDS + 1):
        log.info("── Round %d ── waiting for %d clients", fl_round, MIN_CLIENTS)

        round_q: Queue = Queue()
        handlers = []

        # Accept MIN_CLIENTS connections for this round
        for _ in range(MIN_CLIENTS):
            conn, addr = srv.accept()
            h = ClientHandler(conn, addr, round_q, global_lora, global_kgc)
            h.start()
            handlers.append(h)

        for h in handlers:
            h.join()

        # Collect deltas
        lora_deltas, kgc_deltas, kgc_evals, hyp_stats_list, sizes = [], [], [], [], []
        while not round_q.empty():
            lora_d, kgc_d, kgc_eval, hyp_stats, n = round_q.get()
            lora_deltas.append(lora_d)
            sizes.append(n)
            if kgc_d is not None:
                kgc_deltas.append(kgc_d)
            if kgc_eval is not None:
                kgc_evals.append(kgc_eval)
            if hyp_stats is not None:
                hyp_stats_list.append(hyp_stats)

        if not lora_deltas:
            log.warning("No deltas received — skipping aggregation")
            continue

        global_lora = fedavg(global_lora, lora_deltas, sizes)

        if global_kgc is not None and kgc_deltas:
            # weight by the same client sizes (only over clients that sent a kgc delta)
            kgc_sizes = sizes[:len(kgc_deltas)]
            global_kgc = fedavg(global_kgc, kgc_deltas, kgc_sizes)

        log.info("Round %d complete — aggregated %d client(s)", fl_round, len(lora_deltas))

        if kgc_evals:
            avg_mrr = sum(m["MRR"] for m in kgc_evals) / len(kgc_evals)
            avg_h1  = sum(m["Hits@1"] for m in kgc_evals) / len(kgc_evals)
            avg_h10 = sum(m["Hits@10"] for m in kgc_evals) / len(kgc_evals)
            log.info("Round %d KGC eval (avg over %d clients) — MRR=%.3f Hits@1=%.3f Hits@10=%.3f",
                      fl_round, len(kgc_evals), avg_mrr, avg_h1, avg_h10)

        if hyp_stats_list:
            # Aggregate counts/scores only — never narrative text or
            # patient identifiers, which never leave the client at all.
            total_patients = sum(s["patients_processed"] for s in hyp_stats_list)
            avg_per_patient = sum(s["avg_hypotheses_per_patient"] for s in hyp_stats_list) / len(hyp_stats_list)
            avg_top_score   = sum(s["avg_top_score"] for s in hyp_stats_list) / len(hyp_stats_list)
            log.info("Round %d hypothesis generation (across %d clients, %d patients) — "
                      "avg hypotheses/patient=%.2f avg top score=%.3f",
                      fl_round, len(hyp_stats_list), total_patients,
                      avg_per_patient, avg_top_score)

    srv.close()
    log.info("Training complete after %d rounds", MAX_ROUNDS)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="FMRAG FL Server")
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
        # One-flag dataset switch (mirror of main_client): FMRAG_DATASET sets
        # the KGC vocab size so the server's global KGC model matches clients.
        _ds = os.environ.get("FMRAG_DATASET", "").lower()
        if _ds:
            if _ds in ("mimic", "mimic-iii", "mimic3-demo"):
                _ds = "mimic3"
            _nn = {"synthea": 260, "mimic3": 1335}.get(_ds)
            if _nn and kgc_cfg.get("enabled", False):
                num_nodes = int(os.environ.get("FMRAG_NUM_NODES", _nn))
                log.info("FMRAG_DATASET=%s -> server KGC num_nodes=%d",
                         _ds, num_nodes)
        if num_nodes == 0 and kgc_cfg.get("enabled", False):
            log.warning(
                "kgc.enabled=true but num_nodes=0 in %s — KGC aggregation "
                "will NOT run this session. Set kgc.num_nodes/num_rels to "
                "match graphcare.num_nodes/num_rels in client_config.yaml.",
                args.config,
            )
    except FileNotFoundError:
        log.warning("Config %s not found — starting with KGC aggregation OFF "
                     "(LoRA-only FL, same as before this addition)", args.config)

    run(num_nodes=num_nodes, num_rels=num_rels)
