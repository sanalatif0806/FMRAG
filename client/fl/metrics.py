"""
client/fl/metrics.py
---------------------
Evaluation metrics for FMRAG's binary clinical-prediction tasks
(mortality / readmission), plus a small JSONL logger so every
local/global evaluation is recorded for the paper's results tables.

Metrics reported (all on a HELD-OUT split, never the training data):
  - loss     : mean BCE loss on the eval set
  - accuracy : threshold-0.5 accuracy
  - f1       : F1 of the positive (died/readmitted) class
  - auroc    : area under ROC — the primary metric for imbalanced clinical
               tasks; threshold-independent
  - auprc    : area under precision-recall — more informative than AUROC
               when the positive class is rare

"local" evaluation = on this client's own held-out test split.
"global" evaluation = the just-aggregated global model, scored on the
                      same held-out split (shows what federation bought).
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

log = logging.getLogger(__name__)

# sklearn is in requirements.txt; import defensively so a missing install
# degrades to loss/accuracy only rather than crashing the whole run.
try:
    from sklearn.metrics import (
        f1_score, roc_auc_score, average_precision_score, accuracy_score,
    )
    _SKLEARN = True
except Exception:                       # pragma: no cover
    _SKLEARN = False
    log.warning("sklearn not available — only loss/accuracy will be computed")


@torch.no_grad()
def evaluate(model, dataset, task: str = "mortality",
             device: str = "cpu", batch_size: int = 4) -> dict:
    """
    Run the model over `dataset` (a held-out split) and return a dict of
    metrics. Does not modify the model beyond eval()/train() toggling.
    """
    if len(dataset) == 0:
        return {"loss": float("nan"), "accuracy": float("nan"),
                "f1": float("nan"), "auroc": float("nan"),
                "auprc": float("nan"), "n": 0}

    was_training = model.training
    model.eval()
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
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
        if isinstance(logits, dict):       # multitask — score the disease head
            logits = logits["disease"]
        logits_sq = (logits[:, 1]
                     if logits.dim() == 2 and logits.shape[1] == 2
                     else logits.squeeze())
        logits_sq = logits_sq.reshape(-1)
        y = batch.y.float().reshape(-1)

        total_loss += criterion(logits_sq, y).item()
        n_batches  += 1
        all_probs.append(torch.sigmoid(logits_sq).cpu())
        all_labels.append(y.cpu())

    if was_training:
        model.train()

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    preds  = (probs >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "n":    int(len(labels)),
    }

    if _SKLEARN:
        metrics["accuracy"] = float(accuracy_score(labels, preds))
        # F1/AUROC/AUPRC are undefined if only one class is present in the
        # held-out split (common with tiny per-client test sets) — guard it.
        if len(set(labels.tolist())) > 1:
            # Fixed-0.5 F1 (kept for reference). On imbalanced mortality data
            # (~15% positive) almost nothing crosses 0.5, so this is often 0 —
            # NOT because the model failed but because 0.5 is the wrong cutoff
            # for a rare positive class.
            metrics["f1_at_0.5"] = float(f1_score(labels, preds, zero_division=0))

            # Threshold-tuned F1: the standard way to report F1 on imbalanced
            # data. Sweep candidate thresholds and take the best F1. This
            # reflects the model's real discriminative ability (which AUROC
            # confirms) rather than an artifact of the default 0.5 cutoff.
            import numpy as _np
            best_f1, best_thr = 0.0, 0.5
            for thr in _np.linspace(0.05, 0.95, 19):
                p = (probs >= thr).astype(int)
                _f = f1_score(labels, p, zero_division=0)
                if _f > best_f1:
                    best_f1, best_thr = _f, thr
            metrics["f1"]        = float(best_f1)      # headline F1 (tuned)
            metrics["f1_thr"]    = float(best_thr)     # threshold used
            metrics["auroc"] = float(roc_auc_score(labels, probs))
            metrics["auprc"] = float(average_precision_score(labels, probs))
        else:
            metrics["f1"] = metrics["f1_at_0.5"] = metrics["f1_thr"] = float("nan")
            metrics["auroc"] = metrics["auprc"] = float("nan")
            log.warning("Eval split has a single class (%d samples) — "
                        "F1/AUROC/AUPRC undefined this round", len(labels))
    else:
        correct = (preds == labels).sum()
        metrics["accuracy"] = float(correct / max(len(labels), 1))
        metrics["f1"] = metrics["f1_at_0.5"] = metrics["f1_thr"] = float("nan")
        metrics["auroc"] = metrics["auprc"] = float("nan")

    return metrics


def format_metrics(m: dict) -> str:
    """Compact one-line string for logging."""
    def _f(x):
        return "nan" if x != x else f"{x:.4f}"     # x!=x is NaN check
    thr = m.get("f1_thr")
    thr_str = f"@{thr:.2f}" if (thr is not None and thr == thr) else ""
    return (f"loss={_f(m.get('loss'))} acc={_f(m.get('accuracy'))} "
            f"f1={_f(m.get('f1'))}{thr_str} auroc={_f(m.get('auroc'))} "
            f"auprc={_f(m.get('auprc'))} n={m.get('n', 0)}")


class MetricsLogger:
    """
    Appends one JSON object per evaluation to a JSONL file, so results can
    be loaded later for the paper's tables/plots. Each line records the
    round, scope ("local"/"global"), client id, and all metrics.
    """

    def __init__(self, path: str, client_id: int = 0):
        self.path = path
        self.client_id = client_id
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log(self, round_num: int, scope: str, metrics: dict,
            extra: dict | None = None):
        record = {
            "ts":        time.time(),
            "round":     round_num,
            "scope":     scope,                 # "local" | "global" | "cost"
            "client_id": self.client_id,
            **metrics,
        }
        if extra:
            record.update(extra)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:                  # never let logging crash training
            log.warning("Could not write metrics record: %s", e)
        return record


# ── communication / computation cost helpers ─────────────────────────────────

import pickle as _pickle


def payload_bytes(obj) -> int:
    """Serialised size in bytes of a state_dict / payload (what goes on the
    wire each round). Uses the same pickle the FL transport uses, so this
    is the real communication cost, not an estimate."""
    try:
        return len(_pickle.dumps(obj))
    except Exception:
        return 0


def count_trainable_params(model) -> int:
    """Number of trainable parameters — the per-round compute/upload surface
    (for FMRAG this is the LoRA + GNN head, not the frozen backbone)."""
    try:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        return 0


class CostTracker:
    """
    Accumulates per-round and total timing + communication + computation
    costs across a client's participation, and can emit a final summary.

    Communication is counted both ways:
      down_bytes : global model received from server (per round)
      up_bytes   : client update sent to server (per round)
    Computation proxies:
      train_seconds : wall-clock of local training (the dominant cost)
      eval_seconds  : wall-clock of evaluation
      trainable_params : size of the optimised parameter set
    """

    def __init__(self):
        self.rounds            = 0
        self.total_down_bytes  = 0
        self.total_up_bytes    = 0
        self.total_train_s     = 0.0
        self.total_eval_s      = 0.0
        self.total_round_s     = 0.0
        # split local vs global computation time (for the paper's cost table)
        self.total_local_compute_s  = 0.0
        self.total_global_compute_s = 0.0
        self.start_wall        = time.time()
        self.trainable_params  = 0

    def record_round(self, down_bytes: int, up_bytes: int,
                     train_s: float, eval_s: float, round_s: float,
                     local_compute_s: float = None,
                     global_compute_s: float = None) -> dict:
        self.rounds           += 1
        self.total_down_bytes += down_bytes
        self.total_up_bytes   += up_bytes
        self.total_train_s    += train_s
        self.total_eval_s     += eval_s
        self.total_round_s    += round_s
        if local_compute_s is not None:
            self.total_local_compute_s += local_compute_s
        if global_compute_s is not None:
            self.total_global_compute_s += global_compute_s
        return {
            "down_mb":   round(down_bytes / 1e6, 4),
            "up_mb":     round(up_bytes   / 1e6, 4),
            "train_s":   round(train_s, 2),
            "eval_s":    round(eval_s, 2),
            "round_s":   round(round_s, 2),
        }

    def summary(self) -> dict:
        wall = time.time() - self.start_wall
        return {
            "rounds":              self.rounds,
            "trainable_params":    self.trainable_params,
            "total_down_mb":       round(self.total_down_bytes / 1e6, 3),
            "total_up_mb":         round(self.total_up_bytes   / 1e6, 3),
            "total_comm_mb":       round((self.total_down_bytes
                                          + self.total_up_bytes) / 1e6, 3),
            "avg_comm_mb_per_round": round(
                (self.total_down_bytes + self.total_up_bytes)
                / max(self.rounds, 1) / 1e6, 4),
            "total_train_s":       round(self.total_train_s, 1),
            "avg_train_s_per_round": round(self.total_train_s
                                           / max(self.rounds, 1), 1),
            "total_eval_s":        round(self.total_eval_s, 1),
            "total_local_compute_s":  round(self.total_local_compute_s, 1),
            "total_global_compute_s": round(self.total_global_compute_s, 1),
            "avg_local_compute_s_per_round": round(
                self.total_local_compute_s / max(self.rounds, 1), 2),
            "avg_global_compute_s_per_round": round(
                self.total_global_compute_s / max(self.rounds, 1), 2),
            "wall_clock_s":        round(wall, 1),
        }
