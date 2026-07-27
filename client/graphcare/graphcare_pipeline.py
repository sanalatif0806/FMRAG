"""
FMRAG graphcare_pipeline.py
---------------------------
Adapts GraphCare's data_prepare.py + graphcare.py for FMRAG:

  1. Loads local EHR data (MIMIC-III/IV or custom CSV)
  2. Maps ICD/NDC → CCSCM / CCSPROC / ATC3 (existing GraphCare resources/)
  3. For each patient:
       a. Check PatientCAGCache → use cached graph if present
       b. Else: build subgraph via get_subgraph(), call AgenticRAG for new entities
       c. Merge RAG triples into PyG Data object
       d. Encode node/rel texts via BioBERT (or load from cache)
       e. Store in PatientCAGCache
  4. Returns a PyG InMemoryDataset ready for fl_client.local_train()

Key change from original GraphCare:
  - node_emb is sourced from BioBERT CLS (via PatientCAGCache._encode_texts)
    instead of static word2vec / SapBERT embeddings from get_emb.py
  - RAG triples augment the subgraph beyond the original 2-hop UMLS sampling
"""

from __future__ import annotations

import logging
import os
import pickle
import json
import hashlib
from typing import Optional

import torch
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx
from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset

from graphcare_.task_fn import (
    mortality_prediction_mimic3_fn,
    readmission_prediction_mimic3_fn,
    drug_recommendation_fn,
    length_of_stay_prediction_mimic3_fn,
)
from graphcare_ import split_by_patient
from client.kgc.kg_completion import KGCompletionModel

log = logging.getLogger(__name__)


# ── RAG triple → graph edge helper ───────────────────────────────────────────

def merge_rag_triples(
    graph: Data,
    rag_triples: list[tuple[str, str, str]],
    ent2id: dict,
    rel2id: dict,
) -> Data:
    """
    Append RAG-retrieved (head, rel, tail) triples as new edges to an
    existing PyG Data object.  Unknown entities/relations get new IDs.
    """
    if not rag_triples:
        return graph

    next_ent = max(ent2id.values()) + 1 if ent2id else 0
    next_rel = max(rel2id.values()) + 1 if rel2id else 0

    new_edges_src, new_edges_dst, new_rel_ids = [], [], []

    for h, r, t in rag_triples:
        if h not in ent2id:
            ent2id[h] = next_ent; next_ent += 1
        if t not in ent2id:
            ent2id[t] = next_ent; next_ent += 1
        if r not in rel2id:
            rel2id[r] = next_rel; next_rel += 1

        new_edges_src.append(ent2id[h])
        new_edges_dst.append(ent2id[t])
        new_rel_ids.append(rel2id[r])

    if not new_edges_src:
        return graph

    new_edge_index = torch.tensor(
        [new_edges_src, new_edges_dst], dtype=torch.long
    )
    new_rel_tensor = torch.tensor(new_rel_ids, dtype=torch.long)

    graph.edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1)
    graph.rel_ids    = torch.cat([graph.rel_ids,    new_rel_tensor])

    return graph


# ── main pipeline ─────────────────────────────────────────────────────────────

class FMRAGPatientDataset(InMemoryDataset):
    """
    Builds one PyG Data object per patient visit, with:
      - GraphCare subgraph (conditions + procedures + drugs)
      - BioBERT node embeddings (from PatientCAGCache)
      - RAG-augmented edges (from AgenticRAG)
      - BioBERT tokenisation of the patient's condition text (for BERT forward)
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        task: str,
        ehr_dataset,          # pyhealth SampleDataset
        rag,                  # AgenticRAG instance
        cache,                # PatientCAGCache instance
        ent2id: dict,
        rel2id: dict,
        code_to_name: dict,
        tokenizer,
        max_seq_len: int = 128,
        kgc: Optional[KGCompletionModel] = None,
        hypothesis_engine = None,            # HypothesisEngine instance, or None to disable
        hypothesis_top_k: int = 5,
        hypothesis_log_path: str = "./data/local/hypotheses_log.jsonl",
        transform=None,
    ):
        self.dataset_name = dataset_name
        self.task         = task
        self.ehr          = ehr_dataset
        self.rag          = rag
        self.cache        = cache
        self.ent2id       = ent2id
        self.rel2id       = rel2id
        self.code2name    = code_to_name
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len
        self.kgc          = kgc   # None => no completion filtering, all RAG triples kept
        self.hypothesis_engine   = hypothesis_engine   # None => hypothesis generation disabled
        self.hypothesis_top_k    = hypothesis_top_k
        self.hypothesis_log_path = hypothesis_log_path
        # Aggregate-only stats exposed to fl_client.py for FL payload reporting.
        # Never holds narrative text or patient identifiers — see process().
        self.hypothesis_stats: Optional[dict] = None
        super().__init__(root, transform=transform)

    @property
    def processed_file_names(self):
        return [f"fmrag_{self.dataset_name}_{self.task}.pt"]

    def process(self):
        data_list = []

        # Aggregate-only accumulators for hypothesis generation (no text,
        # no patient identifiers — see the per-patient block below).
        hyp_patients_processed = 0
        hyp_total_count        = 0
        hyp_top_scores: list[float] = []

        for sample in self.ehr.samples:
            pid        = sample["patient_id"]
            visit_id   = sample.get("visit_id", pid)
            conditions = sample.get("conditions", [])
            drugs      = sample.get("drugs", [])
            ccs_codes  = (
                sample.get("conditions", []) +
                sample.get("procedures", []) +
                sample.get("drugs", [])
            )
            label = sample["label"]

            # ── Drug hypothesis generation (Contribution C5, optional) ──────
            # Runs independent of the CAG/RAG cache state — it only depends
            # on this patient's condition/drug codes, not on retrieved KG
            # triples. Disabled entirely (no-op) if hypothesis_engine is None.
            #
            # Privacy: narrative text is written to a LOCAL-ONLY log file,
            # keyed by a one-way hash of patient_id (never the raw ID, never
            # transmitted via FL). Only aggregate numeric stats — counts and
            # scores, no text, no identifiers — get exposed via
            # self.hypothesis_stats for optional reporting in the FL payload.
            if self.hypothesis_engine is not None:
                try:
                    condition_cuis = self.rag.resolve_cuis(conditions, self.code2name) \
                                     if self.rag is not None else []
                    drug_cuis      = self.rag.resolve_cuis(drugs, self.code2name) \
                                     if self.rag is not None else []

                    hypotheses = self.hypothesis_engine.generate(
                        patient_conditions=condition_cuis,
                        patient_drugs=drug_cuis,
                        top_k=self.hypothesis_top_k,
                    )

                    hyp_patients_processed += 1
                    hyp_total_count += len(hypotheses)
                    if hypotheses:
                        hyp_top_scores.append(hypotheses[0].score)

                    if hypotheses:
                        anon_id = hashlib.sha256(pid.encode()).hexdigest()[:12]
                        os.makedirs(os.path.dirname(self.hypothesis_log_path), exist_ok=True)
                        with open(self.hypothesis_log_path, "a") as f:
                            f.write(json.dumps({
                                "anon_id": anon_id,
                                "n_hypotheses": len(hypotheses),
                                "hypotheses": [
                                    {"drug_cui": h.drug_cui, "score": h.score,
                                     "narrative": h.narrative}
                                    for h in hypotheses
                                ],
                            }) + "\n")
                except Exception as e:
                    # Hypothesis generation is best-effort — never let it
                    # break the main training data pipeline.
                    log.warning("Hypothesis generation failed for a patient "
                                "(continuing without it): %s", e)

            # ── CAG lookup ──────────────────────────────────────────────────
            entry = self.cache.get(pid)

            if entry is None:
                # ── RAG retrieval for new patient ──────────────────────────
                new_triples = self.rag.retrieve(
                    patient_id=pid,
                    ccs_codes=ccs_codes,
                    code_to_name=self.code2name,
                )

                # ── KG completion: score retrieved candidates, keep only ───
                # plausible ones (open-world KGC over the retrieved set,
                # not the full vocabulary). If no kgc model was passed,
                # this is a no-op and behaves exactly as before.
                if self.kgc is not None:
                    new_triples, dropped = self.kgc.filter_triples(
                        new_triples, self.ent2id, self.rel2id,
                    )
                    if dropped:
                        log.debug("KGC filtered %d low-plausibility triples for patient %s",
                                  len(dropped), pid)

                # Build base subgraph (stub — replace with real get_subgraph)
                G = nx.Graph()
                for code in ccs_codes:
                    G.add_node(code, node_id=self.ent2id.get(code, 0))
                pyg_graph = from_networkx(G)
                if not hasattr(pyg_graph, "edge_index") or pyg_graph.edge_index is None:
                    pyg_graph.edge_index = torch.zeros(2, 0, dtype=torch.long)
                if not hasattr(pyg_graph, "rel_ids"):
                    pyg_graph.rel_ids = torch.zeros(0, dtype=torch.long)

                # Merge RAG triples
                pyg_graph = merge_rag_triples(
                    pyg_graph, new_triples, self.ent2id, self.rel2id
                )

                # Store in CAG cache
                node_texts = [self.code2name.get(c, c) for c in ccs_codes]
                rel_texts  = list({t[1] for t in new_triples}) or ["unknown"]
                entry = self.cache.put(
                    patient_id=pid,
                    node_texts=node_texts,
                    rel_texts=rel_texts,
                    graph=pyg_graph,
                    cluster_map={},
                    rag_triples=new_triples,
                )
            else:
                pyg_graph = entry.graph

            # ── BioBERT tokenisation of patient summary ─────────────────────
            summary_text = " ".join(
                self.code2name.get(c, c) for c in ccs_codes[:30]
            )
            enc = self.tokenizer(
                summary_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
            )

            # ── Assemble Data object ─────────────────────────────────────────
            data = pyg_graph.clone()
            data.input_ids      = enc["input_ids"].squeeze(0)
            data.attention_mask = enc["attention_mask"].squeeze(0)
            data.node_ids       = torch.tensor(
                [self.ent2id.get(c, 0) for c in ccs_codes], dtype=torch.long
            )
            data.ehr_nodes      = data.node_ids.clone()
            data.y              = torch.tensor(label, dtype=torch.float)
            # -1 sentinel = no ADE label on this sample (single-task runs).
            # Multitask runs always populate this via make_multitask_fn().
            data.y_ade          = torch.tensor(
                sample.get("ade_label", -1), dtype=torch.float
            )

            # visit_node: (1, max_visit, num_nodes) — simplified single visit
            num_nodes = len(ccs_codes)
            data.visit_node = torch.zeros(1, 1, num_nodes, dtype=torch.long)
            for i in range(num_nodes):
                data.visit_node[0, 0, i] = 1

            data_list.append(data)

        if self.hypothesis_engine is not None:
            self.hypothesis_stats = {
                "patients_processed": hyp_patients_processed,
                "avg_hypotheses_per_patient": (
                    hyp_total_count / hyp_patients_processed
                    if hyp_patients_processed else 0.0
                ),
                "avg_top_score": (
                    sum(hyp_top_scores) / len(hyp_top_scores)
                    if hyp_top_scores else 0.0
                ),
            }
            log.info("Drug hypothesis generation summary: %s", self.hypothesis_stats)

        torch.save(self.collate(data_list), self.processed_paths[0])
        log.info("Processed %d patient samples → %s",
                 len(data_list), self.processed_paths[0])

    def len(self):
        return len(self._data_list) if hasattr(self, "_data_list") else 0


# ── convenience loader ────────────────────────────────────────────────────────

def load_ehr_dataset(
    dataset_name: str,
    task: str,
    data_root: str,
    processed_cache: Optional[str] = None,
    disease_task: str = "mortality",
):
    """
    Wraps pyhealth MIMIC3/4 loading with optional pickle cache.
    Mirrors data_prepare.load_dataset() but with configurable paths.

    Cache safety: the cache file stores (dataset_name, task, data_root)
    metadata alongside the dataset. If the current call's parameters don't
    match what's in the cache — e.g. you switched data.root from the demo
    path to full MIMIC-III but kept the same processed_cache filename —
    the stale cache is rejected and the dataset is rebuilt from scratch,
    rather than silently training on the wrong data.
    """
    cache_key = {"dataset_name": dataset_name, "task": task, "data_root": data_root,
                 "disease_task": disease_task if task == "multitask" else None}

    if processed_cache and os.path.exists(processed_cache):
        with open(processed_cache, "rb") as f:
            cached = pickle.load(f)

        if isinstance(cached, dict) and cached.get("_cache_key") == cache_key:
            log.info("Loading cached EHR dataset from %s (matches dataset_name=%s, "
                     "task=%s, data_root=%s)", processed_cache,
                     dataset_name, task, data_root)
            return cached["sample_ds"]
        else:
            log.warning(
                "Cache at %s does not match current config (dataset_name=%s, "
                "task=%s, data_root=%s) — likely stale from a previous run "
                "(e.g. demo data) at a different scale. Rebuilding from "
                "scratch rather than risk training on the wrong dataset.",
                processed_cache, dataset_name, task, data_root,
            )

    code_mapping = {
        "NDC":     ("ATC", {"target_kwargs": {"level": 3}}),
        "ICD9CM":  "CCSCM",
        "ICD9PROC":"CCSPROC",
    }

    if dataset_name == "mimic3":
        ds = MIMIC3Dataset(
            root=data_root,
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping=code_mapping,
        )
    elif dataset_name == "mimic4":
        code_mapping["ICD10CM"]   = "CCSCM"
        code_mapping["ICD10PROC"] = "CCSPROC"
        ds = MIMIC4Dataset(
            root=data_root,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            code_mapping=code_mapping,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if task == "adverse_event":
        # ADE labeling needs raw (unmapped) ICD9 E-codes, which the CCS-mapped
        # `ds` above can no longer provide — see adverse_event_task.py docstring.
        from client.graphcare.adverse_event_task import (
            build_raw_diagnosis_lookup,
            make_adverse_event_task_fn,
        )
        raw_lookup = build_raw_diagnosis_lookup(data_root, dataset_name)
        task_fn = make_adverse_event_task_fn(raw_lookup)
    elif task == "multitask":
        if disease_task not in ("mortality", "readmission"):
            raise ValueError(
                f"multitask mode only supports disease_task in "
                f"('mortality', 'readmission') — got '{disease_task}'. "
                f"drugrec/lenofstay are multi-label/multi-class and aren't "
                f"wired into the multitask loss design in fl_client.py."
            )
        from client.graphcare.adverse_event_task import (
            build_raw_diagnosis_lookup,
            make_multitask_fn,
        )
        raw_lookup = build_raw_diagnosis_lookup(data_root, dataset_name)
        base_task_fns = {
            "mortality":   mortality_prediction_mimic3_fn,
            "readmission": readmission_prediction_mimic3_fn,
        }
        task_fn = make_multitask_fn(base_task_fns[disease_task], raw_lookup)
    else:
        task_fns = {
            "mortality":   mortality_prediction_mimic3_fn,
            "readmission": readmission_prediction_mimic3_fn,
            "drugrec":     drug_recommendation_fn,
            "lenofstay":   length_of_stay_prediction_mimic3_fn,
        }
        task_fn = task_fns[task]

    sample_ds = ds.set_task(task_fn)

    if processed_cache:
        os.makedirs(os.path.dirname(processed_cache), exist_ok=True)
        with open(processed_cache, "wb") as f:
            pickle.dump({"_cache_key": cache_key, "sample_ds": sample_ds}, f)
        log.info("Saved processed EHR dataset → %s (key=%s)", processed_cache, cache_key)

    return sample_ds
