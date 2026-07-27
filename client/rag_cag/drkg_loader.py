"""
client/rag_cag/drkg_loader.py
-------------------------------
Open-data replacement for the UMLS backend in agentic_rag.py.

Loads DRKG (Drug Repurposing Knowledge Graph — free, no license) and
exposes the SAME interface as UMLSRetriever + LocalKGStore, so AgenticRAG
and HypothesisEngine work unchanged with a different `umls_id` namespace.

DRKG entity IDs look like:
    Compound::DB00945      (a drug, DrugBank ID)
    Disease::MESH:D001351  (a disease)
    Gene::2931             (a gene, NCBI ID)
DRKG relations look like:
    bioarx::Coronavirus_ass_host_gene::Disease:Gene
    DRUGBANK::treats::Compound:Disease

Why a SEPARATE text index instead of DRKG's own embeddings:
DRKG ships TransE_l2 embeddings, but those are GRAPH-STRUCTURAL — they
encode where an entity sits in the KG, not what its name means in text.
Our query is a BioBERT embedding of a clinical code's *name* (e.g.
"Metformin"). To match a name to a DRKG entity we need TEXT similarity,
not graph similarity. So this loader builds a BioBERT embedding index
over DRKG entity *names* (parsed/cleaned from the entity IDs), and uses
DRKG's own structural embeddings only if/when downstream graph traversal
needs them. This mirrors exactly what UMLSRetriever does with
concept_names.txt.

Required files (from drkg.tar.gz, extracted to drkg_root):
    drkg.tsv                         — the full triple list (head <rel> tail)
    embed/entities.tsv               — entity_name per row (DRKG entity IDs)
Optional (not needed for the text-index approach, but loaded if present):
    embed/DRKG_TransE_l2_entity.npy  — structural embeddings

The BioBERT name index is cached to disk after first build (it's the
slow step — embedding ~97k entity names), so subsequent runs are fast.
"""

from __future__ import annotations

import logging
import os
import pickle
import re

import numpy as np
import torch

from client.rag_cag.drkg_ids import humanize_drkg_id as _humanize_drkg_id

log = logging.getLogger(__name__)

# Same threshold the UMLS path uses, so behavior is comparable across backends.
SIMILARITY_THRESHOLD = 0.72
MAX_TRIPLES_PER_ENTITY = 5




class DRKGRetriever:
    """
    Drop-in replacement for agentic_rag.UMLSRetriever.

    Same public method: find_similar(query_emb, top_k) -> list of
    {"umls_id", "name", "score"}. The "umls_id" field carries the DRKG
    entity ID (e.g. "Compound::DB00945") so downstream code that treats it
    as an opaque identifier keeps working; only the namespace changed.
    """

    def __init__(
        self,
        drkg_root: str,
        encoder=None,             # FMRAGModel — needed only to BUILD the cache
        tokenizer=None,           # ditto
        device: str = "cpu",
        name_emb_cache: str | None = None,
    ):
        self.device = device
        entities_tsv = os.path.join(drkg_root, "embed", "entities.tsv")
        if not os.path.exists(entities_tsv):
            # entities.tsv may sit at the root in some DRKG distributions
            alt = os.path.join(drkg_root, "entities.tsv")
            entities_tsv = alt if os.path.exists(alt) else entities_tsv

        if not os.path.exists(entities_tsv):
            raise FileNotFoundError(
                f"DRKG entities.tsv not found under {drkg_root}. Extract "
                f"drkg.tar.gz there first (see DRKG_INTEGRATION_PLAN.md)."
            )

        # entities.tsv: each line is "<entity_id>\t<row_index>" (or just the
        # id, one per line, row = line number). Handle both.
        self.entity_ids: list[str] = []
        with open(entities_tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                self.entity_ids.append(parts[0])
        self.entity_names = [_humanize_drkg_id(e) for e in self.entity_ids]
        log.info("DRKG: loaded %d entities", len(self.entity_ids))

        # Build or load the BioBERT text-embedding index over entity names.
        cache = name_emb_cache or os.path.join(drkg_root, "drkg_name_biobert_emb.pkl")
        if os.path.exists(cache):
            log.info("DRKG: loading cached BioBERT name index from %s", cache)
            with open(cache, "rb") as f:
                emb = pickle.load(f)
            self.name_emb = torch.tensor(emb, dtype=torch.float32, device=device)
        else:
            if encoder is None or tokenizer is None:
                raise RuntimeError(
                    "DRKG name-embedding cache does not exist yet and no "
                    "encoder/tokenizer was provided to build it. Construct "
                    "DRKGRetriever with encoder+tokenizer once to build the "
                    "cache (slow: embeds ~97k names), then it's reused."
                )
            self.name_emb = self._build_name_index(encoder, tokenizer, cache)

        self.name_emb = torch.nn.functional.normalize(self.name_emb, dim=1)

    @torch.no_grad()
    def _build_name_index(self, encoder, tokenizer, cache_path: str) -> torch.Tensor:
        log.info("DRKG: building BioBERT name index for %d entities "
                 "(one-time, slow)…", len(self.entity_names))
        embs = []
        BATCH = 64
        for i in range(0, len(self.entity_names), BATCH):
            batch = self.entity_names[i:i + BATCH]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=32, padding=True)
            out = encoder.encoder(**enc)
            cls = out.last_hidden_state[:, 0, :]      # (B, 768)
            embs.append(cls.cpu())
            if i % (BATCH * 50) == 0:
                log.info("  …%d / %d", i, len(self.entity_names))
        mat = torch.cat(embs, dim=0)
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(mat.numpy(), f)
        log.info("DRKG: name index cached to %s", cache_path)
        return mat.to(self.device)

    @torch.no_grad()
    def find_similar(self, query_emb: torch.Tensor, top_k: int = 5) -> list[dict]:
        """Identical signature/return to UMLSRetriever.find_similar."""
        q = torch.nn.functional.normalize(query_emb.unsqueeze(0), dim=1).to(self.device)
        sims = (self.name_emb @ q.T).squeeze()
        top_idx = sims.topk(top_k).indices.tolist()
        return [
            {"umls_id": self.entity_ids[i],     # DRKG id in the umls_id slot
             "name":    self.entity_names[i],
             "score":   float(sims[i])}
            for i in top_idx
            if float(sims[i]) >= SIMILARITY_THRESHOLD
        ]


class DRKGStore:
    """
    Drop-in replacement for agentic_rag.LocalKGStore, backed by drkg.tsv.

    Same public method: query(entity_name, max_triples) -> list of
    (head, rel, tail). Indexed by head entity ID (case-insensitive), same
    as LocalKGStore.
    """

    def __init__(self, drkg_root: str):
        self._head_idx: dict[str, list[tuple]] = {}
        drkg_tsv = os.path.join(drkg_root, "drkg.tsv")
        if not os.path.exists(drkg_tsv):
            raise FileNotFoundError(
                f"drkg.tsv not found under {drkg_root}. Extract drkg.tar.gz "
                f"there first."
            )
        log.info("DRKG: loading triple store from %s", drkg_tsv)
        n = 0
        with open(drkg_tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                self._head_idx.setdefault(h.lower(), []).append((h, r, t))
                n += 1
        log.info("DRKG: loaded %d triples over %d unique head entities",
                 n, len(self._head_idx))

    def query(self, entity_name: str, max_triples: int = MAX_TRIPLES_PER_ENTITY):
        return self._head_idx.get(entity_name.lower(), [])[:max_triples]
