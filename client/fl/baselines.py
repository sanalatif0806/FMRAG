"""
client/fl/baselines.py
-----------------------
Federated-learning BASELINE methods, implemented as composable loss terms so
they run inside PARC's existing FL loop and produce identical metrics/cost
records. This lets the paper compare PARC against standard baselines on the
SAME data, model, and evaluation --- which is what reviewers want, and far
more rigorous (and tractable) than running four separate external repos built
for CIFAR/MNIST.

Baselines implemented here:
  - FedProx  (Li et al., MLSys 2020, arXiv:1812.06127)
        proximal term mu/2 * ||theta - theta_global||^2 pulling the local
        model toward the received global model. The standard non-IID FL
        baseline.
  - FedCurv  (Shoham et al., 2019, arXiv:1910.07796)
        EWC adapted to FL: a Fisher-weighted penalty anchoring local params
        to the GLOBAL model (not a previous local snapshot). This is the
        closest prior work to PARC's continual-learning contribution, so it
        is the most important baseline to include.
  - FedLwF   (federated Learning-without-Forgetting)
        knowledge distillation from the global model as teacher, adapting
        LwF (Li & Hoiem, 2017) to the federated setting.

Method selection is driven by FMRAG_METHOD (set by run_sweep.sh):
    fedavg    : no extra term (plain FedAvg)
    fedprox   : + FedProx proximal term
    fedcurv   : + FedCurv Fisher penalty to global
    fedlwf    : + KD-to-global term
    parc_ewc  : PARC's EWC (anchors to previous snapshot) --- handled in
                fl_client via the existing EWCRegulariser
    parc_full : PARC's full objective (EWC + KD + KGC)

NOTE on honesty: FedProx/FedCurv/FedLwF here are faithful re-implementations
of the published loss terms, not the original authors' code. State this in the
paper ("baselines re-implemented within our framework"). This is standard and
accepted practice, but should be disclosed.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── FedProx ───────────────────────────────────────────────────────────────────

class FedProxTerm:
    """
    Proximal regularisation toward the received global model.
    L += (mu/2) * sum ||theta - theta_global||^2   over trainable params.

    Usage: build ONCE per round right after loading the global weights
    (so it captures theta_global), then add .penalty(model) to the loss
    each step.
    """

    def __init__(self, model, mu: float = 0.01, device: str = "cpu"):
        self.mu = mu
        self.device = device
        # snapshot the global (= current) trainable params as the anchor
        self._anchor = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }

    def penalty(self, model) -> torch.Tensor:
        if self.mu <= 0:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._anchor:
                loss = loss + (p - self._anchor[n]).pow(2).sum()
        return (self.mu / 2.0) * loss


# ── FedCurv ───────────────────────────────────────────────────────────────────

class FedCurvTerm:
    """
    FedCurv: EWC-style Fisher-weighted penalty anchoring local params to the
    GLOBAL model. Distinct from PARC's EWC, which anchors to the previous
    LOCAL snapshot. This is the key continual-learning baseline.

    L += (lambda/2) * sum F_i * (theta_i - theta_global_i)^2

    We reuse the same Fisher-estimation approach as the existing EWC code
    (diagonal empirical Fisher from a reference batch), but anchor to global.
    """

    def __init__(self, model, ref_loader, device: str = "cpu",
                 lam: float = 1.0):
        self.lam = lam
        self.device = device
        self._anchor = {
            n: p.detach().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self._fisher = {n: torch.zeros_like(p)
                        for n, p in model.named_parameters() if p.requires_grad}
        self._ready = False
        self._ref_loader = ref_loader

    @torch.enable_grad()
    def estimate_fisher(self, model, task: str = "mortality",
                        n_samples: int = 50):
        """Diagonal empirical Fisher from a few reference batches."""
        model.eval()
        seen = 0
        for batch in self._ref_loader:
            if seen >= n_samples:
                break
            batch = batch.to(self.device)
            model.zero_grad()
            try:
                logits, _ = model(
                    input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                    node_ids=batch.node_ids, rel_ids=batch.rel_ids,
                    edge_index=batch.edge_index, batch=batch.batch,
                    visit_node=batch.visit_node, ehr_nodes=batch.ehr_nodes,
                )
                if isinstance(logits, dict):
                    logits = logits["disease"]
                logits_sq = (logits[:, 1] if logits.dim() == 2 and logits.shape[1] == 2
                             else logits.squeeze()).reshape(-1)
                y = batch.y.float().reshape(-1)
                loss = F.binary_cross_entropy_with_logits(logits_sq, y)
                loss.backward()
            except Exception as e:
                log.warning("FedCurv Fisher: skipping batch (%s)", e)
                continue
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self._fisher[n] += p.grad.detach().pow(2)
            seen += batch.num_graphs if hasattr(batch, "num_graphs") else 1
        for n in self._fisher:
            self._fisher[n] /= max(seen, 1)
        self._ready = True
        model.zero_grad()
        log.info("FedCurv: Fisher estimated over %d samples", seen)

    def penalty(self, model) -> torch.Tensor:
        if not self._ready or self.lam <= 0:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for n, p in model.named_parameters():
            if p.requires_grad and n in self._fisher:
                loss = loss + (self._fisher[n] * (p - self._anchor[n]).pow(2)).sum()
        return (self.lam / 2.0) * loss


# ── FedLwF ────────────────────────────────────────────────────────────────────

class FedLwFTerm:
    """
    Federated Learning-without-Forgetting: KD from the GLOBAL model as teacher.
    L += lambda_lwf * KL(student || teacher)  with the received global model
    as the frozen teacher. Adapts LwF (Li & Hoiem 2017) to FL.

    This reuses the same soft-target KD as PARC's KD term, but the teacher is
    explicitly the global model at round start (the LwF formulation).
    """

    def __init__(self, model, device: str = "cpu", lam: float = 0.5,
                 temperature: float = 4.0):
        self.lam = lam
        self.T = temperature
        self.device = device
        self._teacher = copy.deepcopy(model).to(device)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _teacher_logits(self, batch):
        logits, _ = self._teacher(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask,
            node_ids=batch.node_ids, rel_ids=batch.rel_ids,
            edge_index=batch.edge_index, batch=batch.batch,
            visit_node=batch.visit_node, ehr_nodes=batch.ehr_nodes,
        )
        if isinstance(logits, dict):
            logits = logits["disease"]
        return logits

    def penalty(self, student_logits, batch) -> torch.Tensor:
        if self.lam <= 0:
            return torch.tensor(0.0, device=self.device)
        teacher_logits = self._teacher_logits(batch)
        # align shapes
        s = student_logits
        t = teacher_logits
        if s.dim() == 1:
            s = torch.stack([-s, s], dim=-1)
        if t.dim() == 1:
            t = torch.stack([-t, t], dim=-1)
        p_teacher = F.softmax(t / self.T, dim=-1)
        q_student = F.log_softmax(s / self.T, dim=-1)
        return self.lam * F.kl_div(q_student, p_teacher,
                                   reduction="batchmean") * (self.T ** 2)


def build_baseline(method: str, model, ref_loader, device: str,
                   mu: float = 0.01, lam: float = 1.0):
    """
    Factory: return the appropriate baseline term object for `method`, or None
    if the method needs no extra term (fedavg) or is handled elsewhere
    (parc_* via the existing EWC/KD path).
    """
    method = (method or "").lower()
    if method == "fedprox":
        return ("fedprox", FedProxTerm(model, mu=mu, device=device))
    if method == "fedcurv":
        term = FedCurvTerm(model, ref_loader, device=device, lam=lam)
        term.estimate_fisher(model)
        return ("fedcurv", term)
    if method == "fedlwf":
        return ("fedlwf", FedLwFTerm(model, device=device, lam=lam))
    return (None, None)
