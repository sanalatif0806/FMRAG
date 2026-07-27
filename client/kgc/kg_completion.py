"""
client/kgc/kg_completion.py
-----------------------------
Federated, retrieval-constrained Knowledge Graph Completion (KGC).

What "completion" means here (and why it's new vs. the existing pipeline):

  Before this module, AgenticRAG.retrieve() returns every triple it finds
  via UMLS similarity + local KG lookup, and graphcare_pipeline.merge_rag_triples()
  merges ALL of them into the patient graph unconditionally. That is
  retrieval + augmentation, not completion — nothing is being predicted,
  everything retrieved is trusted and kept.

  This module adds the missing piece: a learned scoring model that predicts
  how plausible a (head, relation, tail) triple actually is, trained on each
  client's locally observed KG. AgenticRAG's retrieved triples become
  *candidates*; this model decides which candidates are kept (i.e. which
  missing edges actually get "completed") and which are discarded as noise.

  This is open-world KGC in the formal sense (Shi & Weninger style): the
  candidate set comes from retrieval over an open, growing entity space,
  not from a fixed pre-enumerated closed-world vocabulary. Restricting
  scoring to retrieved candidates (rather than the full entity vocabulary)
  is also what keeps this tractable on CPU edge devices.

  The embedding tables here are trained locally per client and exchanged
  via FL exactly like the LoRA adapters in client/peft/model_setup.py —
  kgc_state_dict() / load_kgc_state_dict() mirror that pattern so fl_client.py
  can treat KGC as a second FL-aggregated component alongside the encoder.

Scoring function: DistMult bilinear form.
  score(h, r, t) = sum(e_h * e_r * e_t)
  Simple, cheap on CPU, and standard for this scale of KG.

Toggle for ablation studies (stability-plasticity research question):
  FMRAG_KGC_ENABLED=true/false   — turn KGC filtering on/off entirely
  FMRAG_KGC_THRESHOLD            — plausibility cutoff for keeping a candidate
  FMRAG_KGC_DIM                  — embedding dimension
  FMRAG_KGC_NEG_SAMPLES          — negative samples per positive during training
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [KGC] %(message)s")

# ── config (env-overridable, same pattern as fl_client.py / model_setup.py) ──
KGC_ENABLED   = os.environ.get("FMRAG_KGC_ENABLED", "true").lower() == "true"
EMB_DIM       = int(os.environ.get("FMRAG_KGC_DIM", "64"))
NEG_SAMPLES   = int(os.environ.get("FMRAG_KGC_NEG_SAMPLES", "10"))
SCORE_THRESH  = float(os.environ.get("FMRAG_KGC_THRESHOLD", "0.5"))
LR            = float(os.environ.get("FMRAG_KGC_LR", "1e-3"))

log.info("KGC enabled: %s | dim: %d | threshold: %.2f", KGC_ENABLED, EMB_DIM, SCORE_THRESH)


class KGCompletionModel(nn.Module):
    """
    DistMult-style bilinear scorer over the same entity/relation ID space
    as GraphCare (ent2id, rel2id from graphcare_pipeline.py), so no separate
    vocabulary needs to be maintained.
    """

    def __init__(self, num_nodes: int, num_rels: int, dim: int = EMB_DIM):
        super().__init__()
        self.entity_emb   = nn.Embedding(num_nodes, dim)
        self.relation_emb = nn.Embedding(num_rels, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.num_nodes = num_nodes
        self.num_rels  = num_rels

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eh = self.entity_emb(h)
        er = self.relation_emb(r)
        et = self.entity_emb(t)
        return (eh * er * et).sum(dim=-1)

    def forward(self, h, r, t):
        return self.score(h, r, t)

    # ── training ──────────────────────────────────────────────────────────

    def train_step(
        self,
        pos_triples: torch.Tensor,     # (N, 3) long: [head_id, rel_id, tail_id]
        optimizer: torch.optim.Optimizer,
        neg_samples: int = NEG_SAMPLES,
    ) -> float:
        """One local training step on a client's observed (true) triples."""
        device = self.entity_emb.weight.device
        h, r, t = pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2]

        pos_score = self.score(h, r, t)
        pos_loss  = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))

        neg_loss_acc = torch.zeros((), device=device)
        for _ in range(neg_samples):
            t_neg     = torch.randint(0, self.num_nodes, t.shape, device=device)
            neg_score = self.score(h, r, t_neg)
            neg_loss_acc = neg_loss_acc + F.binary_cross_entropy_with_logits(
                neg_score, torch.zeros_like(neg_score)
            )
        neg_loss = neg_loss_acc / max(neg_samples, 1)

        loss = pos_loss + neg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss)

    def fit(self, pos_triples: torch.Tensor, epochs: int = 1, lr: float = LR) -> float:
        """Convenience wrapper — trains for `epochs` over the full triple set."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        last_loss = 0.0
        for _ in range(epochs):
            last_loss = self.train_step(pos_triples, optimizer)
        return last_loss

    # ── retrieval-constrained completion (the actual "completion" step) ──

    @torch.no_grad()
    def rank_candidates(
        self,
        head_id: int,
        rel_id: int,
        candidate_tail_ids: list[int],
    ) -> list[tuple[int, float]]:
        """
        Score only candidates AgenticRAG actually retrieved — not the full
        entity vocabulary. Returns [(tail_id, plausibility_score), ...]
        sorted descending. plausibility_score is sigmoid(DistMult score) in [0,1].
        """
        if not candidate_tail_ids:
            return []
        device = self.entity_emb.weight.device
        h = torch.full((len(candidate_tail_ids),), head_id, device=device, dtype=torch.long)
        r = torch.full((len(candidate_tail_ids),), rel_id,  device=device, dtype=torch.long)
        t = torch.tensor(candidate_tail_ids, device=device, dtype=torch.long)
        scores = torch.sigmoid(self.score(h, r, t)).tolist()
        return sorted(zip(candidate_tail_ids, scores), key=lambda x: -x[1])

    def filter_triples(
        self,
        triples: list[tuple[str, str, str]],
        ent2id: dict,
        rel2id: dict,
        threshold: float = SCORE_THRESH,
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str, float]]]:
        """
        Drop-in replacement for "merge everything AgenticRAG retrieved".

        Takes the raw (head_name, rel_name, tail_name) triples returned by
        AgenticRAG.retrieve(), scores each one, and splits them into
        (kept, dropped_with_scores). Unknown entities/relations (not yet
        in ent2id/rel2id) are kept automatically — there's nothing to score
        them against yet, and graphcare_pipeline.merge_rag_triples() will
        assign them new ids as it currently does.

        Use `kept` exactly where `new_triples` is currently passed into
        merge_rag_triples() in graphcare_pipeline.py.
        """
        if not KGC_ENABLED:
            return triples, []

        kept, dropped = [], []
        for h_name, r_name, t_name in triples:
            if h_name not in ent2id or t_name not in ent2id or r_name not in rel2id:
                # Can't score an entity/relation we have no embedding for yet.
                kept.append((h_name, r_name, t_name))
                continue
            h_id, r_id, t_id = ent2id[h_name], rel2id[r_name], ent2id[t_name]
            if h_id >= self.num_nodes or t_id >= self.num_nodes or r_id >= self.num_rels:
                kept.append((h_name, r_name, t_name))
                continue
            score = self.rank_candidates(h_id, r_id, [t_id])[0][1]
            if score >= threshold:
                kept.append((h_name, r_name, t_name))
            else:
                dropped.append((h_name, r_name, t_name, score))

        if dropped:
            log.debug("KGC dropped %d/%d candidate triples below threshold %.2f",
                       len(dropped), len(triples), threshold)
        return kept, dropped

    # ── FL helpers (mirrors FMRAGModel.lora_state_dict in model_setup.py) ─

    def kgc_state_dict(self) -> dict:
        """What goes to the FL server — entity + relation embeddings."""
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def load_kgc_state_dict(self, state_dict: dict) -> None:
        own = self.state_dict()
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k].data.copy_(v)

    # ── evaluation: intrinsic completion quality, independent of the ──────
    # ── downstream clinical prediction task                            ──

    @torch.no_grad()
    def evaluate(
        self,
        test_triples: torch.Tensor,         # (N, 3) held-out true triples
        candidate_pool_size: int = 50,
        ks: tuple[int, ...] = (1, 3, 10),
    ) -> dict:
        """
        Standard KGC eval: for each held-out (h, r, t), rank the true tail
        against `candidate_pool_size - 1` random negative tails. Reports
        Hits@k and MRR. Run this on the masked edges produced by
        mask_edges_for_eval() to measure completion quality directly,
        separate from mortality/readmission AUROC.
        """
        device = self.entity_emb.weight.device
        hits = {k: 0 for k in ks}
        mrr_total = 0.0
        n = test_triples.shape[0]
        if n == 0:
            return {"MRR": 0.0, **{f"Hits@{k}": 0.0 for k in ks}}

        for i in range(n):
            h, r, t_true = test_triples[i].tolist()
            negs = random.sample(
                [e for e in range(self.num_nodes) if e != t_true],
                min(candidate_pool_size - 1, self.num_nodes - 1),
            )
            candidates = [t_true] + negs

            h_t = torch.full((len(candidates),), h, device=device, dtype=torch.long)
            r_t = torch.full((len(candidates),), r, device=device, dtype=torch.long)
            c_t = torch.tensor(candidates, device=device, dtype=torch.long)
            scores  = self.score(h_t, r_t, c_t)
            ranking = torch.argsort(scores, descending=True).tolist()
            rank    = ranking.index(0) + 1   # true tail is always at index 0 pre-shuffle

            mrr_total += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1

        return {"MRR": mrr_total / n, **{f"Hits@{k}": hits[k] / n for k in ks}}


def mask_edges_for_eval(
    triples: list[tuple[int, int, int]],
    mask_ratio: float = 0.1,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """
    Randomly hold out `mask_ratio` of a client's local (id-form) triples to
    simulate "missing" edges for completion evaluation. Call once per FL
    round (fresh split each round) so EWC's stability claims and KGC's
    completion accuracy can be measured on the same round's data.

    Returns (train_triples, held_out_triples), both lists of (h_id, r_id, t_id).
    """
    triples = list(triples)
    random.shuffle(triples)
    n_mask = max(1, int(len(triples) * mask_ratio)) if triples else 0
    held_out = triples[:n_mask]
    train    = triples[n_mask:]
    return train, held_out


def build_kgc_model(num_nodes: int, num_rels: int) -> KGCompletionModel:
    """Convenience constructor — mirrors build_model() in model_setup.py."""
    model = KGCompletionModel(num_nodes=num_nodes, num_rels=num_rels)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("KGC model built — %s params (dim=%d, nodes=%d, rels=%d)",
             f"{n_params:,}", EMB_DIM, num_nodes, num_rels)
    return model
