"""
FMRAG agentic_rag.py  —  Agentic RAG over patient KG + UMLS
------------------------------------------------------------
Replaces the static ChatGPT.py KG extraction in GraphCare.

The agent:
  1. Receives a patient's CCS codes (conditions, procedures, drugs)
  2. Plans a sequence of retrieval steps (ReAct-style)
  3. Queries local patient subgraph → UMLS similarity → LLM KG extraction
  4. Re-ranks and deduplicates retrieved triples
  5. Returns augmented triples to be merged into the patient's PyG graph

CAG integration: if the patient is already in PatientCAGCache, only
NEW/unseen entities (below SIMILARITY_THRESHOLD) trigger RAG retrieval.
Known entities are served from cache directly.
"""

from __future__ import annotations

import logging
import pickle
from typing import Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.72   # same as umls_sim_retriever.py default
MAX_RAG_HOPS = 2              # max retrieval chain length per entity
MAX_TRIPLES_PER_ENTITY = 5


# ── UMLS local retriever (wraps umls_sim_retriever.py logic) ─────────────────

class UMLSRetriever:
    """
    Wraps the UMLS embedding index for fast cosine similarity lookup.
    Loads from the pkl files produced by umls_sim_retriever.py.
    """

    def __init__(self, umls_ent_emb_path: str, umls_names_path: str,
                 device: str = "cpu"):
        log.info("Loading UMLS embeddings from %s", umls_ent_emb_path)
        with open(umls_ent_emb_path, "rb") as f:
            emb = pickle.load(f)
        self.umls_emb   = torch.tensor(emb, dtype=torch.float32, device=device)
        self.umls_emb   = torch.nn.functional.normalize(self.umls_emb, dim=1)
        self.device     = device

        with open(umls_names_path) as f:
            lines = f.readlines()
        self.umls_ids   = [l.split("\t")[0] for l in lines]
        self.umls_names = [l.split("\t")[1].strip() for l in lines]

    @torch.no_grad()
    def find_similar(self, query_emb: torch.Tensor, top_k: int = 5) -> list[dict]:
        """
        query_emb : (768,) normalised BioBERT embedding
        Returns list of {umls_id, name, score}
        """
        q = torch.nn.functional.normalize(query_emb.unsqueeze(0), dim=1).to(self.device)
        sims = (self.umls_emb @ q.T).squeeze()
        top_idx = sims.topk(top_k).indices.tolist()
        return [
            {"umls_id": self.umls_ids[i],
             "name":    self.umls_names[i],
             "score":   float(sims[i])}
            for i in top_idx
            if float(sims[i]) >= SIMILARITY_THRESHOLD
        ]


# ── KG triple store (from KG_mapping/global_node_triple_store.txt) ────────────

class LocalKGStore:
    """
    In-memory index of (head, relation, tail) triples from the
    GraphCare KG_mapping/global_node_triple_store.txt and UMLS.
    """

    def __init__(self, triple_store_path: str):
        self._head_idx: dict[str, list[tuple]] = {}
        log.info("Loading KG triple store from %s", triple_store_path)
        with open(triple_store_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                h, r, t = parts[0], parts[1], parts[2]
                self._head_idx.setdefault(h.lower(), []).append((h, r, t))
        log.info("Loaded %d unique head entities",  len(self._head_idx))

    def query(self, entity_name: str, max_triples: int = MAX_TRIPLES_PER_ENTITY):
        return self._head_idx.get(entity_name.lower(), [])[:max_triples]


# ── Agentic RAG orchestrator ──────────────────────────────────────────────────

class AgenticRAG:
    """
    ReAct-style retrieval agent.

    Parameters
    ----------
    umls_retriever  : UMLSRetriever
    kg_store        : LocalKGStore
    encoder         : FMRAGModel  (BioBERT encoder for query embeddings)
    tokenizer
    patient_cache   : PatientCAGCache (to check what's already cached)
    """

    def __init__(
        self,
        umls_retriever: UMLSRetriever,
        kg_store:        LocalKGStore,
        encoder,
        tokenizer,
        patient_cache=None,
    ):
        self.umls   = umls_retriever
        self.kg     = kg_store
        self.enc    = encoder
        self.tok    = tokenizer
        self.cache  = patient_cache

    # ── main entry point ──────────────────────────────────────────────────────

    def retrieve(
        self,
        patient_id:   str,
        ccs_codes:    list[str],    # e.g. ["CCSCM-127", "ATC3-A10", ...]
        code_to_name: dict[str, str],
    ) -> list[tuple[str, str, str]]:
        """
        For each code in ccs_codes that is NOT already in the patient cache,
        run multi-hop retrieval and return new (head, rel, tail) triples.
        """
        cached_entry = self.cache.get(patient_id) if self.cache else None
        cached_codes = set()
        if cached_entry:
            cached_codes = {t[0].lower() for t in cached_entry.rag_triples}

        all_triples = []

        for code in ccs_codes:
            name = code_to_name.get(code, code)

            # CAG HIT: entity already in cache → skip RAG
            if name.lower() in cached_codes:
                log.debug("RAG skip (cached): %s", name)
                continue

            log.info("RAG retrieve: %s (%s)", code, name)
            triples = self._retrieve_for_entity(name, hops=0)
            all_triples.extend(triples)

        log.info("RAG total new triples: %d for patient %s",
                 len(all_triples), patient_id)
        return all_triples

    # ── CUI resolution for drug hypothesis generation ──────────────────────────

    def resolve_cuis(
        self,
        codes: list[str],
        code_to_name: dict[str, str],
    ) -> list[str]:
        """
        Resolves CCS/ATC codes to their nearest UMLS CUI, for use with
        HypothesisEngine (which operates purely in UMLS CUI space).

        Reuses the exact same embedding + similarity mechanism as RAG
        triple retrieval (_encode + umls.find_similar) rather than a
        separate ground-truth mapping table — this repo has no
        CCS-to-UMLS crosswalk file, so this is an approximate resolution,
        not an exact one. Codes with no match above SIMILARITY_THRESHOLD
        are silently dropped (HypothesisEngine.generate() already handles
        missing CUIs gracefully by skipping them).
        """
        cuis = []
        for code in codes:
            name = code_to_name.get(code, code)
            emb = self._encode(name)
            matches = self.umls.find_similar(emb, top_k=1)
            if matches:
                cuis.append(matches[0]["umls_id"])
            else:
                log.debug("No UMLS CUI match for code %s (%s) above threshold", code, name)
        return cuis

    @torch.no_grad()
    def _encode(self, text: str) -> torch.Tensor:
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=32)
        out = self.enc.encoder(**enc)
        return out.last_hidden_state[:, 0, :].squeeze(0)   # (768,)

    def _retrieve_for_entity(
        self, entity_name: str, hops: int
    ) -> list[tuple[str, str, str]]:
        if hops >= MAX_RAG_HOPS:
            return []

        triples = []

        # Step 1: direct KG lookup
        direct = self.kg.query(entity_name)
        triples.extend(direct)
        log.debug("  KG direct: %d triples for '%s'", len(direct), entity_name)

        # Step 2: UMLS similarity — find related concepts
        emb = self._encode(entity_name)
        similar = self.umls.find_similar(emb, top_k=3)

        for item in similar:
            umls_name = item["name"]
            if umls_name.lower() == entity_name.lower():
                continue
            # Build a bridging triple
            triples.append((entity_name, "similar_to_umls", umls_name))
            # Recursively follow one more hop for strong matches
            if item["score"] > 0.85 and hops < MAX_RAG_HOPS - 1:
                triples.extend(
                    self._retrieve_for_entity(umls_name, hops=hops + 1)
                )

        # Deduplicate
        seen = set()
        unique = []
        for t in triples:
            key = (t[0].lower(), t[1].lower(), t[2].lower())
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique[:MAX_TRIPLES_PER_ENTITY * (hops + 1)]
