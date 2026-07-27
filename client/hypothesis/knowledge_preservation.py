"""
FMRAG knowledge_preservation.py  —  Contribution C3
-----------------------------------------------------
"Mechanisms to prevent the loss of foundational medical insights
 during federated personalization"
                                   — Doctoral Consortium Paper, §2

Two complementary mechanisms are implemented:

1. EWC (Elastic Weight Consolidation)
   After receiving global LoRA weights from the server, compute
   Fisher information for the LoRA parameters on a small reference
   medical corpus.  Add EWC penalty to local training loss to
   prevent the LoRA weights drifting away from the global prior.

   Loss_total = Loss_task + lambda_ewc * EWC_penalty

2. Global-model distillation
   Keep a frozen copy of the just-received global model.
   Add a KL-divergence term between local model outputs and
   global model outputs on the same batch.  This is the
   "cyclic distillation" idea from FedDK [Xu & Fan, 2023]
   adapted to the LoRA setting.

   Loss_total = Loss_task + lambda_ewc * EWC_penalty
                           + lambda_kd  * KL_divergence

Usage in fl_client.py
---------------------
    from client.hypothesis.knowledge_preservation import (
        EWCRegulariser, compute_ewc_penalty, kd_loss
    )

    # --- After receiving global LoRA weights ---
    ewc = EWCRegulariser(model, reference_loader, device)
    ewc.estimate_fisher()          # run once per round on reference data

    # --- Inside local_train loss computation ---
    loss = criterion(logits, labels)
    loss = loss + ewc.penalty(model)
    loss = loss + kd_loss(logits, global_logits, temperature=4.0)
    loss.backward()
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── EWC Regulariser ───────────────────────────────────────────────────────────

class EWCRegulariser:
    """
    Elastic Weight Consolidation for LoRA adapter parameters.

    After each round's global weights arrive, call estimate_fisher()
    on a small held-out reference dataset (e.g. 50 examples from a
    public biomedical QA set or the client's own validation split).

    During local training, add ewc.penalty(model) to the task loss.

    Parameters
    ----------
    model           : FMRAGModel — the live model
    reference_loader: DataLoader over a small reference set
    device          : torch device string
    lambda_ewc      : EWC regularisation strength (default 500)
                      Higher = stronger protection against forgetting.
                      Paper ablation should sweep [100, 500, 1000, 5000].
    """

    def __init__(
        self,
        model,
        reference_loader,
        device:     str   = "cpu",
        lambda_ewc: float = 500.0,
    ):
        self.model      = model
        self.loader     = reference_loader
        self.device     = device
        self.lambda_ewc = lambda_ewc

        # Will be populated by estimate_fisher()
        self._means:   dict[str, torch.Tensor] = {}
        self._fisher:  dict[str, torch.Tensor] = {}
        self._ready    = False

    # ── Fisher estimation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _snapshot_params(self):
        """Snapshot the current LoRA parameter values as the EWC anchor."""
        self._means = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad and "lora_" in n
        }

    def estimate_fisher(
        self,
        n_samples: int = 200,
        task:      str = "mortality",
    ):
        """
        Estimate diagonal Fisher information matrix for LoRA parameters
        using the empirical Fisher approximation (squared gradients).

        Call this once after loading new global weights each FL round.
        Uses at most n_samples examples from reference_loader.
        """
        log.info("EWC: estimating Fisher information (n_samples=%d)", n_samples)
        self._snapshot_params()

        # Initialise Fisher accumulators
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad and "lora_" in n
        }

        self.model.train()
        seen = 0

        for batch in self.loader:
            if seen >= n_samples:
                break
            batch = batch.to(self.device)
            self.model.zero_grad()

            logits, _ = self.model(
                input_ids      = batch.input_ids,
                attention_mask = batch.attention_mask,
                node_ids       = batch.node_ids,
                rel_ids        = batch.rel_ids,
                edge_index     = batch.edge_index,
                batch          = batch.batch,
                visit_node     = batch.visit_node,
                ehr_nodes      = batch.ehr_nodes,
            )

            # Use log-softmax likelihood as the loss for Fisher estimation.
            # Multitask: logits is {"disease": tensor, "ade": tensor} — sum
            # both heads' log-likelihoods so Fisher reflects sensitivity to
            # forgetting on EITHER task, not just one.
            if isinstance(logits, dict):
                log_prob = (
                    F.logsigmoid(logits["disease"].squeeze())
                    + F.logsigmoid(logits["ade"].squeeze())
                )
            elif task in ("mortality", "readmission"):
                log_prob = F.logsigmoid(logits.squeeze())
            else:
                log_prob = F.log_softmax(logits, dim=-1).max(dim=-1).values

            log_prob.sum().backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and "lora_" in n and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)

            seen += batch.num_graphs

        # Normalise by number of samples
        self._fisher = {n: f / max(seen, 1) for n, f in fisher.items()}
        self._ready  = True
        self.model.zero_grad(set_to_none=True)  # clear stale graphs
        log.info("EWC: Fisher estimated over %d samples, %d LoRA params",
                 seen, len(self._fisher))

    # ── EWC penalty ───────────────────────────────────────────────────────────

    def penalty(self, model) -> torch.Tensor:
        """
        Returns the EWC penalty term to add to the training loss.
        Penalty = lambda/2 * sum_i [ F_i * (theta_i - theta*_i)^2 ]

        If estimate_fisher() has not been called yet, returns 0.
        """
        if not self._ready:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n]).pow(2)).sum()

        return (self.lambda_ewc / 2.0) * loss


# ── Knowledge Distillation loss ───────────────────────────────────────────────

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature:    float = 4.0,
) -> torch.Tensor:
    """
    Soft-target KL divergence between local model and frozen global model.
    Used for the FedDK-style cyclic distillation component.

    Parameters
    ----------
    student_logits : logits from the current (being-trained) local model
    teacher_logits : logits from the frozen global model snapshot
    temperature    : softmax temperature (higher = softer targets)

    Returns
    -------
    Scalar KL-divergence loss.
    """
    T = temperature
    p_teacher = F.softmax(teacher_logits / T, dim=-1)
    q_student = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(q_student, p_teacher, reduction="batchmean") * (T ** 2)


# ── Global model snapshot helper ──────────────────────────────────────────────

class GlobalModelSnapshot:
    """
    Keeps a frozen copy of the global model for distillation.
    Call update() each round after loading new global LoRA weights.
    """

    def __init__(self, model):
        self._snapshot: Optional[dict] = None
        self._model_ref = model

    def update(self):
        """Snapshot current model state (called after loading global weights)."""
        self._snapshot = copy.deepcopy(self._model_ref.state_dict())
        log.info("GlobalModelSnapshot updated (%d param tensors)",
                 len(self._snapshot))

    @torch.no_grad()
    def predict(self, batch, device: str) -> torch.Tensor:
        """
        Run the snapshotted global model on a batch, return logits.
        Uses a separate model copy — never modifies the training model.
        """
        if self._snapshot is None:
            raise RuntimeError("Call update() before predict()")

        teacher_model = copy.deepcopy(self._model_ref)
        teacher_model.load_state_dict(self._snapshot)
        teacher_model.eval()

        batch = batch.to(device)
        with torch.no_grad():
            logits, _ = teacher_model(
                input_ids      = batch.input_ids,
                attention_mask = batch.attention_mask,
                node_ids       = batch.node_ids,
                rel_ids        = batch.rel_ids,
                edge_index     = batch.edge_index,
                batch          = batch.batch,
                visit_node     = batch.visit_node,
                ehr_nodes      = batch.ehr_nodes,
            )

        del teacher_model
        return logits.detach()
