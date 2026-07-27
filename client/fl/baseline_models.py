"""
client/fl/baseline_models.py
----------------------------
Standalone BASELINE MODEL ARCHITECTURES for PARC's comparison table.

*** IMPORTANT — READ THIS ***
These are SCAFFOLD implementations. They follow the published architectures
and consume PARC's data interface (batch.ehr_nodes / batch.visit_node /
batch.y), but they have NOT been executed here (no torch/GPU/data in the dev
sandbox). You MUST validate each on your VM with a tiny run before trusting
its numbers. They are a starting point that saves you the architecture design,
not drop-in verified code.

Also: in the paper, disclose that RETAIN/G-BERT are "re-implemented within our
framework following the original papers," and that TARGET is "an adapted
re-implementation." Do not claim you ran the authors' official code.

Contents:
  - RETAIN       (Choi et al., NeurIPS 2016) — two-level attention over visits
  - GBERTLite    (Shang et al., 2019) — code-embedding transformer (simplified;
                 NOT the full dual-KG pretraining, which needs a separate
                 pretraining corpus — see note in the class)
  - TARGETReplay (Zhang et al., ICCV 2023) — generative-replay scaffold for
                 federated class-continual learning (server-side generator +
                 distillation). This is the heaviest and least likely to work
                 without substantial tuning — treat as experimental.

Each model exposes the SAME forward signature shape as PARC's model so it can
slot into fl_client's eval/train, returning (logits, aux). Because these are
single-task discriminative models, aux is None.

The data interface (per batched sample):
  batch.ehr_nodes  : (B, V) multi-hot over the code vocab (V = num_nodes)
  batch.visit_node : (B, max_visit, V) per-visit code occurrence
  batch.y          : (B,) binary label
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── RETAIN (Choi et al., NeurIPS 2016) ────────────────────────────────────────

class RETAIN(nn.Module):
    """
    RETAIN: REverse Time AttentIoN. Two RNNs generate visit-level (alpha) and
    code-level (beta) attention over the visit sequence, combined into a
    context vector for prediction. Consumes batch.visit_node (B, T, V).

    Reference: Choi et al., "RETAIN: An Interpretable Predictive Model for
    Healthcare using Reverse Time Attention", NeurIPS 2016.

    SCAFFOLD — validate on VM. The reverse-time detail (processing visits in
    reverse) is implemented; tune hidden_dim / embed_dim for your data.
    """

    def __init__(self, num_nodes: int, embed_dim: int = 128,
                 hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Linear(num_nodes, embed_dim)
        self.rnn_alpha = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.rnn_beta  = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.alpha_fc  = nn.Linear(hidden_dim, 1)      # visit attention
        self.beta_fc   = nn.Linear(hidden_dim, embed_dim)  # code attention
        self.output    = nn.Linear(embed_dim, 2)

    def forward(self, input_ids=None, attention_mask=None, node_ids=None,
                rel_ids=None, edge_index=None, batch=None,
                visit_node=None, ehr_nodes=None):
        # visit_node: (B, T, V) -> embed each visit
        x = visit_node
        if x is None:
            raise ValueError("RETAIN needs visit_node (B, T, V)")
        B, T, V = x.shape
        v = self.embed(x)                      # (B, T, E)
        # reverse time for RETAIN's reverse-attention
        v_rev = torch.flip(v, dims=[1])
        h_alpha, _ = self.rnn_alpha(v_rev)
        h_beta,  _ = self.rnn_beta(v_rev)
        alpha = torch.softmax(self.alpha_fc(h_alpha), dim=1)  # (B, T, 1)
        beta  = torch.tanh(self.beta_fc(h_beta))              # (B, T, E)
        # context = sum_t alpha_t * (beta_t ⊙ v_t)   (flip back)
        ctx = torch.sum(alpha * beta * torch.flip(v, dims=[1]), dim=1)  # (B, E)
        logits = self.output(ctx)              # (B, 2)
        return logits, None

    # PARC's FL loop expects LoRA-style state helpers; for a full baseline we
    # federate ALL params. Provide compatible shims.
    def lora_state_dict(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_lora_state_dict(self, sd):
        self.load_state_dict(sd, strict=False)


# ── G-BERT-lite (Shang et al., 2019) ──────────────────────────────────────────

class GBERTLite(nn.Module):
    """
    Simplified G-BERT: a transformer encoder over medical-code embeddings with
    a classification head. Consumes batch.ehr_nodes (B, V) as a bag of codes.

    IMPORTANT: The ORIGINAL G-BERT's main contribution is dual-KG-aware
    PRETRAINING on ICD/ATC ontologies before fine-tuning. That pretraining
    needs a separate corpus and protocol not reproduced here. This is the
    fine-tuning architecture only, trained from scratch — so it under-
    represents true G-BERT. Disclose this in the paper if you report it, or
    (better) cite G-BERT as related work rather than as a run baseline.

    Reference: Shang et al., "Pre-training of Graph Augmented Transformers
    for Medication Recommendation", IJCAI 2019.

    SCAFFOLD — validate on VM.
    """

    def __init__(self, num_nodes: int, embed_dim: int = 128, n_heads: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.code_embed = nn.Embedding(num_nodes, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Linear(embed_dim, 2)

    def forward(self, input_ids=None, attention_mask=None, node_ids=None,
                rel_ids=None, edge_index=None, batch=None,
                visit_node=None, ehr_nodes=None):
        if ehr_nodes is None:
            raise ValueError("GBERTLite needs ehr_nodes (B, V)")
        B, V = ehr_nodes.shape
        # treat present codes as a token set; embed all codes, weight by presence
        all_emb = self.code_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B,V,E)
        mask = ehr_nodes > 0                                              # (B,V)
        # encode; use presence mask as key_padding_mask (True = ignore)
        enc = self.encoder(all_emb, src_key_padding_mask=~mask)
        # mean-pool over PRESENT codes
        m = mask.unsqueeze(-1).float()
        pooled = (enc * m).sum(1) / m.sum(1).clamp(min=1.0)
        logits = self.cls(pooled)
        return logits, None

    def lora_state_dict(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def load_lora_state_dict(self, sd):
        self.load_state_dict(sd, strict=False)


# ── TARGET (Zhang et al., ICCV 2023) — generative-replay scaffold ─────────────

class TARGETGenerator(nn.Module):
    """
    Lightweight generator for TARGET-style federated class-continual learning.
    Generates synthetic code-multi-hot vectors conditioned on a class label,
    used to rehearse previous knowledge at the server and distill to clients.

    *** THIS IS THE HEAVIEST BASELINE AND THE LEAST LIKELY TO WORK WITHOUT
    SUBSTANTIAL TUNING. *** The original TARGET is designed for class-
    incremental IMAGE classification. Adapting it to binary clinical
    prediction on code vectors is non-trivial: with only 2 classes and no
    task-sequence, the "class-continual" premise barely applies. I strongly
    recommend citing TARGET as related work rather than reporting it as a
    comparison. This scaffold is provided because you asked for all of them.

    Reference: Zhang et al., "TARGET: Federated Class-Continual Learning via
    Exemplar-Free Distillation", ICCV 2023.
    Code (image domain): github.com/zj-jayzhang/Federated-Class-Continual-Learning
    """

    def __init__(self, num_nodes: int, noise_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.noise_dim = noise_dim
        self.label_embed = nn.Embedding(n_classes, noise_dim)
        self.net = nn.Sequential(
            nn.Linear(noise_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, num_nodes), nn.Sigmoid(),   # multi-hot-ish output
        )

    def forward(self, labels):
        B = labels.shape[0]
        z = torch.randn(B, self.noise_dim, device=labels.device)
        c = self.label_embed(labels)
        return self.net(torch.cat([z, c], dim=-1))     # (B, num_nodes) in [0,1]


def target_replay_distill(student, teacher, generator, n_classes: int = 2,
                          n_samples: int = 64, device: str = "cpu",
                          temperature: float = 4.0) -> torch.Tensor:
    """
    One TARGET distillation step: generate synthetic past-task samples, get
    the teacher (previous global) predictions, and distill into the student.
    Returns a KD loss to add to training.

    SCAFFOLD — the generator here is UNTRAINED unless you add a generator
    training loop (the original trains it to match the teacher's feature
    statistics). Without that, this reduces to random-input distillation,
    which is weak. Provided for completeness; needs real work to be faithful.
    """
    labels = torch.randint(0, n_classes, (n_samples,), device=device)
    with torch.no_grad():
        synth = generator(labels)                       # (N, V)
        ehr = (synth > 0.5).float()
        # teacher forward on synthetic ehr_nodes
        t_logits, _ = teacher(ehr_nodes=ehr)
    s_logits, _ = student(ehr_nodes=ehr)
    p_t = F.softmax(t_logits / temperature, dim=-1)
    q_s = F.log_softmax(s_logits / temperature, dim=-1)
    return F.kl_div(q_s, p_t, reduction="batchmean") * (temperature ** 2)


# ── factory ───────────────────────────────────────────────────────────────────

def build_baseline_model(name: str, num_nodes: int, **kw):
    """
    Return a baseline MODEL (not a loss term) for the model-level baselines.
    Used when FMRAG_BASELINE_MODEL is set (retain | gbert). TARGET is handled
    separately via the generator + distill helpers.
    """
    name = (name or "").lower()
    if name == "retain":
        return RETAIN(num_nodes, **kw)
    if name == "gbert":
        return GBERTLite(num_nodes, **kw)
    raise ValueError(f"unknown baseline model: {name}")
