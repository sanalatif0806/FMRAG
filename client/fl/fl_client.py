"""
client/fl/fl_client.py
-----------------------
FL round: receive global LoRA weights → local training → send delta.

CPU optimisations:
  - BATCH_SIZE 4  (vs 16 on GPU) — fits in RAM without swapping
  - LOCAL_EPOCHS 1  (vs 3 on GPU) — meaningful update without hours of waiting
  - gradient accumulation every GRAD_ACCUM steps — simulates larger batch
  - torch.set_num_threads() — uses all CPU cores
  - mixed precision skipped (CPU float32 is fine; bfloat16 optional)
  - EWC Fisher estimated on 50 samples (vs 200) — fast on CPU

All values auto-scale based on CPU/GPU detection.
Override via environment variables.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import socket
import struct
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from client.hypothesis.knowledge_preservation import (
    EWCRegulariser,
    GlobalModelSnapshot,
    kd_loss,
)
from client.kgc.kg_completion import KGCompletionModel, mask_edges_for_eval

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [FL-CLIENT] %(message)s")

# ── Hardware detection ────────────────────────────────────────────────────────
_HAS_GPU   = torch.cuda.is_available()
_CPU_MODE  = not _HAS_GPU or os.environ.get("FMRAG_CPU_MODE","").lower()=="true"
DEVICE     = "cpu"   # always CPU on edge devices — explicit

# Use all available CPU cores for PyTorch operations
_CPU_CORES = os.cpu_count() or 4
torch.set_num_threads(_CPU_CORES)
log.info("Using %d CPU threads", _CPU_CORES)

# ── FL hyperparameters — auto-scaled for CPU ──────────────────────────────────
SERVER_HOST   = os.environ.get("FMRAG_CENTRAL_IP",  "100.105.131.115")
SERVER_PORT   = int(os.environ.get("FMRAG_FL_PORT", "8080"))
# Async mode: send ABSOLUTE post-training weights + the base_version pulled,
# and loop continuously at the client's own pace (FedAsync). When false,
# the client sends deltas for synchronous FedAvg (the original behavior).
ASYNC_MODE    = os.environ.get("FMRAG_ASYNC", "").lower() == "true"
# In async mode, optionally cap how many local cycles this client runs
# (0 = unlimited, until the server stops). Useful for bounded experiments.
ASYNC_MAX_CYCLES = int(os.environ.get("FMRAG_ASYNC_MAX_CYCLES", "0"))

# CPU: 1 epoch × batch 4 × grad_accum 4 = effective batch 16
# GPU: 3 epochs × batch 16
LOCAL_EPOCHS  = int(os.environ.get("FMRAG_LOCAL_EPOCHS", "1" if _CPU_MODE else "3"))
BATCH_SIZE    = int(os.environ.get("FMRAG_BATCH_SIZE",   "4" if _CPU_MODE else "16"))
GRAD_ACCUM    = int(os.environ.get("FMRAG_GRAD_ACCUM",   "4" if _CPU_MODE else "1"))
LR            = float(os.environ.get("FMRAG_LR",         "2e-4"))
LAMBDA_EWC    = float(os.environ.get("FMRAG_LAMBDA_EWC", "500.0"))
ADE_LOSS_WEIGHT = float(os.environ.get("FMRAG_ADE_LOSS_WEIGHT", "1.0"))
LAMBDA_KD     = float(os.environ.get("FMRAG_LAMBDA_KD",  "1.0"))
EWC_SAMPLES   = int(os.environ.get("FMRAG_EWC_SAMPLES",  "50" if _CPU_MODE else "200"))

# ── Evaluation / metrics ──────────────────────────────────────────────────────
from client.fl.metrics import (
    evaluate, format_metrics, MetricsLogger,
    payload_bytes, count_trainable_params, CostTracker,
)
# Fraction of this client's data held out for evaluation (never trained on).
EVAL_TEST_FRAC = float(os.environ.get("FMRAG_EVAL_TEST_FRAC", "0.2"))
# Where per-round local/global metrics are written (JSONL, one row per eval).
METRICS_PATH   = os.environ.get("FMRAG_METRICS_PATH", "/var/fmrag/fl_metrics.jsonl")
_CLIENT_ID     = int(os.environ.get("FMRAG_CLIENT_ID", "0"))

# ── Local checkpointing — makes offline operation possible ─────────────────
# Without this, a client that loses connectivity (or just restarts) has
# nothing to fall back on and cannot make any prediction at all while
# disconnected from the FL server. This persists the last successfully
# received global state (LoRA + KGC) to local disk after every round.
CHECKPOINT_PATH = os.environ.get(
    "FMRAG_CHECKPOINT_PATH", "/var/fmrag/checkpoints/last_global_state.pt"
)


def save_local_checkpoint(global_state: dict, round_num: int = -1):
    """Persists the last successfully-received global state to local disk."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save({
        "global_state": global_state,
        "round": round_num,
        "saved_at": time.time(),
    }, CHECKPOINT_PATH)
    log.info("Saved local checkpoint (round %d) -> %s", round_num, CHECKPOINT_PATH)


def load_local_checkpoint() -> "dict | None":
    """
    Loads the last cached global state, if one exists. Returns None if no
    checkpoint has ever been saved (e.g. very first run, never connected
    to the server even once).
    """
    if not os.path.exists(CHECKPOINT_PATH):
        return None
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    age_hours = (time.time() - ckpt["saved_at"]) / 3600
    log.info("Loaded local checkpoint from round %d (%.1f hours old)",
             ckpt["round"], age_hours)
    return ckpt

log.info(
    "FL config — CPU mode: %s | epochs: %d | batch: %d | "
    "grad_accum: %d | EWC samples: %d",
    _CPU_MODE, LOCAL_EPOCHS, BATCH_SIZE, GRAD_ACCUM, EWC_SAMPLES,
)


# ── Socket helpers ────────────────────────────────────────────────────────────

def send_weights(conn: socket.socket, state_dict: dict):
    payload = pickle.dumps(state_dict)
    conn.sendall(struct.pack(">I", len(payload)))
    conn.sendall(payload)


def recv_weights(conn: socket.socket) -> dict:
    raw_len = _recv_exact(conn, 4)
    n       = struct.unpack(">I", raw_len)[0]
    return pickle.loads(_recv_exact(conn, n))


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


# ── Local training ────────────────────────────────────────────────────────────

def local_train(
    model,
    dataset,
    task:        str               = "mortality",
    ewc:         EWCRegulariser    = None,
    global_snap: GlobalModelSnapshot = None,
    baseline_term = None,          # (name, obj) for FedProx/FedCurv/FedLwF
) -> tuple[dict, int]:
    """
    Train model for LOCAL_EPOCHS with gradient accumulation.
    Returns (lora_delta, n_samples).
    """
    model.to(DEVICE)
    model.train()

    pre_state = copy.deepcopy(model.lora_state_dict())

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )

    criterion = (nn.BCEWithLogitsLoss()
                 if task in ("mortality", "readmission", "multitask")
                 else nn.CrossEntropyLoss())
    ade_criterion = nn.BCEWithLogitsLoss()

    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    n_samples = len(dataset)

    for epoch in range(LOCAL_EPOCHS):
        epoch_loss = 0.0

        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad()
            batch = batch.to(DEVICE)

            logits, _ = model(
                input_ids      = batch.input_ids,
                attention_mask = batch.attention_mask,
                node_ids       = batch.node_ids,
                rel_ids        = batch.rel_ids,
                edge_index     = batch.edge_index,
                batch          = batch.batch,
                visit_node     = batch.visit_node,
                ehr_nodes      = batch.ehr_nodes,
            )

            if isinstance(logits, dict):
                # ── Multitask: disease loss + ADE loss ──────────────────────
                disease_logits = logits["disease"]
                ade_logits     = logits["ade"]

                disease_sq = (
                    disease_logits[:, 1]
                    if disease_logits.dim() == 2 and disease_logits.shape[1] == 2
                    else disease_logits.squeeze()
                )
                disease_loss = criterion(disease_sq, batch.y.float())

                # y_ade uses -1 as a sentinel for "no ADE label on this
                # sample" (see graphcare_pipeline.py) — mask those out so
                # they don't corrupt the ADE loss. In practice this should
                # be all-or-nothing per batch since multitask task_fn
                # labels every sample, but the mask is cheap insurance.
                ade_sq = (
                    ade_logits[:, 1]
                    if ade_logits.dim() == 2 and ade_logits.shape[1] == 2
                    else ade_logits.squeeze()
                )
                ade_mask = batch.y_ade >= 0
                if ade_mask.any():
                    ade_loss = ade_criterion(ade_sq[ade_mask], batch.y_ade[ade_mask].float())
                else:
                    ade_loss = torch.zeros((), device=DEVICE)

                loss = disease_loss + ADE_LOSS_WEIGHT * ade_loss
            else:
                # Fix logits shape for binary cross entropy
                logits_squeezed = (
                    logits[:, 1]
                    if logits.dim() == 2 and logits.shape[1] == 2
                    else logits.squeeze()
                )

                # Task loss
                loss = criterion(logits_squeezed, batch.y.float())

            # EWC knowledge preservation (C3)
            if ewc is not None:
                loss = loss + ewc.penalty(model)

            # Federated baselines (FedProx / FedCurv / FedLwF) — run through the
            # same loop and metrics as PARC for a fair same-data comparison.
            if baseline_term is not None and baseline_term[0] is not None:
                _bname, _bobj = baseline_term
                if _bname in ("fedprox", "fedcurv"):
                    loss = loss + _bobj.penalty(model)
                elif _bname == "fedlwf":
                    _student = logits["disease"] if isinstance(logits, dict) else logits
                    loss = loss + _bobj.penalty(_student, batch)

            # KD distillation disabled — inplace tensor version conflict
            # if global_snap is not None:
            #     teacher = global_snap.predict(batch, DEVICE)
            #     loss = loss + LAMBDA_KD * kd_loss(logits.detach().clone(), teacher)

            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.detach().item() * GRAD_ACCUM

            # Step optimizer every GRAD_ACCUM batches
            if step % GRAD_ACCUM == 0 or step == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        avg = epoch_loss / max(len(loader), 1)
        log.info("  Epoch %d/%d  loss=%.4f  (batches=%d, effective_batch=%d)",
                 epoch + 1, LOCAL_EPOCHS, avg,
                 len(loader), BATCH_SIZE * GRAD_ACCUM)

    # Compute weight delta
    post_state = model.lora_state_dict()
    delta      = {k: post_state[k] - pre_state[k] for k in post_state}
    # Return both the delta (sync FedAvg) and the absolute post-training
    # weights (async FedAsync merges absolute models, not deltas).
    return delta, n_samples, post_state


# ── Main FL loop ──────────────────────────────────────────────────────────────

def run(model, dataset, task: str = "mortality",
        reference_dataset=None,
        kgc: "KGCompletionModel | None" = None,
        kgc_triples=None):
    """
    Connect to FL server, participate in FL rounds indefinitely.
    Handles both nightly (Trigger 1) and query-failure (Trigger 2) rounds.

    kgc, kgc_triples : optional — pass a KGCompletionModel plus this
        client's local (h_id, r_id, t_id) triples (as a (N,3) LongTensor)
        to additionally train + federate the KG completion embeddings each
        round, and to measure intrinsic completion quality (Hits@k, MRR)
        on a held-out edge split. Leave both None to run exactly as before
        (ablation: KGC component disabled).
    """
    ref_size   = max(1, len(dataset) // 10)
    ref_ds     = reference_dataset or dataset[:ref_size]
    ref_loader = DataLoader(ref_ds, batch_size=BATCH_SIZE, shuffle=False)
    snap       = GlobalModelSnapshot(model)

    # ── Held-out train/test split (deterministic, reproducible) ──────────
    # Evaluation metrics are ALWAYS computed on test_ds, which is never
    # trained on, so accuracy/F1/AUROC/AUPRC are honest generalisation
    # numbers rather than training-set fit. Seeded so the split is stable
    # across rounds and reruns.
    import random as _random
    _idx = list(range(len(dataset)))
    _random.Random(1234 + _CLIENT_ID).shuffle(_idx)
    _n_test = max(1, int(len(dataset) * EVAL_TEST_FRAC))
    _test_idx  = set(_idx[:_n_test])
    train_ds = [dataset[i] for i in range(len(dataset)) if i not in _test_idx]
    test_ds  = [dataset[i] for i in range(len(dataset)) if i in _test_idx]
    metrics_logger = MetricsLogger(METRICS_PATH, client_id=_CLIENT_ID)
    cost = CostTracker()
    cost.trainable_params = count_trainable_params(model)
    log.info("Eval split: %d train / %d test (test_frac=%.2f) — metrics -> %s",
             len(train_ds), len(test_ds), EVAL_TEST_FRAC, METRICS_PATH)
    log.info("Trainable params (per-round compute/upload surface): %d",
             cost.trainable_params)

    # ── Load cached state at startup, BEFORE attempting to connect ────────
    # This is what makes the client usable immediately even if the server
    # happens to be down on first boot — it loads whatever global state
    # was last successfully received, rather than starting from an
    # untrained model with nothing to fall back on.
    cached = load_local_checkpoint()
    if cached is not None:
        model.load_lora_state_dict(cached["global_state"]["lora"])
        if kgc is not None and cached["global_state"].get("kgc") is not None:
            kgc.load_kgc_state_dict(cached["global_state"]["kgc"])
        log.info("Startup: applied cached global state from round %d — "
                 "model is usable for local inference even if the server "
                 "is unreachable", cached["round"])
    else:
        log.warning("Startup: no local checkpoint found — model is at its "
                     "initial state until the first successful FL round")

    log.info("FL client ready — server: %s:%d | device: %s | KGC: %s",
             SERVER_HOST, SERVER_PORT, DEVICE, "on" if kgc is not None else "off")

    round_num = cached["round"] if cached is not None else 0

    while True:
        try:
            _round_t0 = time.time()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((SERVER_HOST, SERVER_PORT))
            log.info("Connected to FL server")

            # 1. Receive global LoRA weights (+ global KGC weights if enabled)
            global_state = recv_weights(s)
            _down_bytes = payload_bytes(global_state)   # communication: down
            model.load_lora_state_dict(global_state["lora"])
            base_version = global_state.get("version", 0)   # async: server's version
            log.info("Loaded global LoRA (%d param tensors)%s",
                     len(global_state["lora"]),
                     f" [v{base_version}]" if ASYNC_MODE else "")

            if kgc is not None and global_state.get("kgc") is not None:
                kgc.load_kgc_state_dict(global_state["kgc"])
                log.info("Loaded global KGC embeddings")

            round_num += 1
            save_local_checkpoint(global_state, round_num)

            # ── GLOBAL eval: the just-received aggregated model, scored on
            # this client's held-out test set BEFORE any local training.
            # This is the number that shows what federation achieved. (On
            # round 1 the global model is untrained, so expect ~chance.)
            _geval_t0 = time.time()
            global_metrics = evaluate(model, test_ds, task=task, device=DEVICE,
                                      batch_size=BATCH_SIZE)
            _global_eval_s = time.time() - _geval_t0     # GLOBAL-model eval time
            log.info("[GLOBAL eval r%d] %s", round_num, format_metrics(global_metrics))
            metrics_logger.log(round_num, "global", global_metrics,
                               extra={"version": base_version})

            # 2. Snapshot for KD distillation
            snap.update()

            # 3. Method selection: PARC uses EWC; the FL baselines
            #    (fedprox/fedcurv/fedlwf) use their own term anchored to the
            #    global model. FMRAG_METHOD picks which. Fisher timed separately.
            _fisher_t0 = time.time()
            _method = os.environ.get("FMRAG_METHOD", "").lower()
            baseline_term = (None, None)
            ewc = None
            if _method in ("fedprox", "fedcurv", "fedlwf"):
                from client.fl.baselines import build_baseline
                _mu  = float(os.environ.get("FMRAG_FEDPROX_MU", "0.01"))
                _lam = float(os.environ.get("FMRAG_BASELINE_LAMBDA", "1.0"))
                baseline_term = build_baseline(_method, model, ref_loader,
                                               DEVICE, mu=_mu, lam=_lam)
                log.info("Baseline method active: %s", _method)
            else:
                ewc = EWCRegulariser(model, ref_loader, DEVICE, LAMBDA_EWC)
                ewc.estimate_fisher(n_samples=EWC_SAMPLES, task=task)
            _fisher_s = time.time() - _fisher_t0

            # 4. Local training (on the TRAIN split only — test_ds is held out)
            log.info("Starting local training — %d train samples on %s",
                     len(train_ds), DEVICE)
            _train_t0 = time.time()
            delta, n_samples, post_state = local_train(
                model, train_ds, task=task,
                ewc=ewc, global_snap=snap,
                baseline_term=baseline_term,
            )
            _train_s = time.time() - _train_t0           # LOCAL training time
            log.info("  Local training wall-clock: %.1fs (%d samples)",
                     _train_s, len(train_ds))

            # ── LOCAL eval: this client's model AFTER local training, on the
            # same held-out test set. Comparing local-after-training vs the
            # global-before-training above shows how much this client's own
            # data improved things locally (the personalisation signal).
            _leval_t0 = time.time()
            local_metrics = evaluate(model, test_ds, task=task, device=DEVICE,
                                     batch_size=BATCH_SIZE)
            _local_eval_s = time.time() - _leval_t0       # LOCAL-model eval time
            # combined eval time kept for backward-compat with CostTracker
            _eval_s = _global_eval_s + _local_eval_s
            log.info("[LOCAL  eval r%d] %s", round_num, format_metrics(local_metrics))
            metrics_logger.log(round_num, "local", local_metrics,
                               extra={"n_train": len(train_ds)})

            # In async (FedAsync) mode, send ABSOLUTE post-training weights and
            # echo the base_version pulled, so the server can staleness-weight
            # the merge. In sync mode, send the delta as before.
            lora_payload = post_state if ASYNC_MODE else delta
            payload = {"lora": lora_payload, "n_samples": n_samples,
                       "kgc": None, "kgc_eval": None,
                       "base_version": base_version,
                       "hypothesis_stats": getattr(dataset, "hypothesis_stats", None)}

            # 4b. KGC: local training + held-out completion eval this round
            if kgc is not None and kgc_triples is not None and len(kgc_triples) > 0:
                train_triples, held_out = mask_edges_for_eval(
                    [tuple(t) for t in kgc_triples.tolist()], mask_ratio=0.1
                )
                pre_kgc = copy.deepcopy(kgc.kgc_state_dict())
                if train_triples:
                    train_tensor = torch.tensor(train_triples, dtype=torch.long, device=DEVICE)
                    kgc_loss = kgc.fit(train_tensor, epochs=1)
                    log.info("  KGC local loss=%.4f  (triples=%d)", kgc_loss, len(train_triples))

                post_kgc   = kgc.kgc_state_dict()
                # async merges absolute weights; sync uses deltas
                if ASYNC_MODE:
                    payload["kgc"] = post_kgc
                else:
                    payload["kgc"] = {k: post_kgc[k] - pre_kgc[k] for k in post_kgc}

                if held_out:
                    held_tensor = torch.tensor(held_out, dtype=torch.long, device=DEVICE)
                    metrics = kgc.evaluate(held_tensor)
                    log.info("  KGC eval — MRR=%.3f  Hits@1=%.3f  Hits@10=%.3f",
                             metrics["MRR"], metrics["Hits@1"], metrics["Hits@10"])
                    payload["kgc_eval"] = metrics

            # 5. Send weights back (absolute in async, delta in sync)
            _up_bytes = payload_bytes(payload)          # communication: up
            send_weights(s, payload)
            log.info("Sent %s (n=%d, keys=%d)%s",
                     "absolute weights" if ASYNC_MODE else "LoRA delta",
                     n_samples, len(lora_payload),
                     "  + KGC" if payload["kgc"] is not None else "")
            s.close()

            # ── Cost accounting for this round (timing + communication) ──
            _round_s = time.time() - _round_t0
            # LOCAL computation = training + Fisher + local-model eval
            # GLOBAL computation = global-model eval (the received model)
            _local_compute_s  = _train_s + _fisher_s + _local_eval_s
            _global_compute_s = _global_eval_s
            _round_cost = cost.record_round(
                down_bytes=_down_bytes, up_bytes=_up_bytes,
                train_s=_train_s, eval_s=_eval_s, round_s=_round_s,
                local_compute_s=_local_compute_s,
                global_compute_s=_global_compute_s,
            )
            # attach the split timings to the cost record for the paper
            _round_cost.update({
                "local_train_s":    round(_train_s, 3),
                "fisher_s":         round(_fisher_s, 3),
                "local_eval_s":     round(_local_eval_s, 3),
                "global_eval_s":    round(_global_eval_s, 3),
                "local_compute_s":  round(_local_compute_s, 3),
                "global_compute_s": round(_global_compute_s, 3),
            })
            log.info("[COST r%d] LOCAL compute=%.1fs (train=%.1f fisher=%.1f "
                     "eval=%.1f) | GLOBAL compute=%.1fs (eval) | round=%.1fs | "
                     "comm down=%.3fMB up=%.3fMB",
                     round_num, _local_compute_s, _train_s, _fisher_s,
                     _local_eval_s, _global_compute_s, _round_s,
                     _round_cost["down_mb"], _round_cost["up_mb"])
            metrics_logger.log(round_num, "cost", _round_cost,
                               extra={"trainable_params": cost.trainable_params})

            # Async: loop again immediately (pull fresh global, train, push),
            # at this client's own pace. Bounded by ASYNC_MAX_CYCLES if set.
            if ASYNC_MODE:
                if ASYNC_MAX_CYCLES and round_num >= ASYNC_MAX_CYCLES:
                    log.info("Async: reached max cycles (%d) — stopping client",
                             ASYNC_MAX_CYCLES)
                    _summary = cost.summary()
                    log.info("[COST SUMMARY] rounds=%d | total_train=%.1fs "
                             "(avg %.1fs/round) | total_comm=%.3fMB "
                             "(avg %.4fMB/round) | wall=%.1fs | params=%d",
                             _summary["rounds"], _summary["total_train_s"],
                             _summary["avg_train_s_per_round"],
                             _summary["total_comm_mb"],
                             _summary["avg_comm_mb_per_round"],
                             _summary["wall_clock_s"], _summary["trainable_params"])
                    metrics_logger.log(round_num, "summary", _summary)
                    break
                continue

        except ConnectionRefusedError:
            log.warning("FL server not reachable — retrying in 60s. "
                         "Cached model from round %d remains usable for "
                         "local inference in the meantime (see "
                         "client/inference/local_inference.py).", round_num)
            time.sleep(60)
        except ConnectionError as e:
            log.error("Connection error: %s — cached model from round %d "
                       "remains usable for local inference", e, round_num)
            time.sleep(30)
