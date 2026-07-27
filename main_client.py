"""
FMRAG main_client.py
--------------------
Top-level entrypoint for a hospital/cloudlet client node.
Wires together:
  - model_setup.py        → BioBERT + LoRA + PEFT + BAT-GNN
  - agentic_rag.py        → RAG retrieval (UMLS + local KG)
  - kg_completion.py      → scores RAG candidates, federates completion embeddings
  - synthea_loader.py     → Synthea CSV → patient graph dataset
  - graphcare_pipeline.py → MIMIC/EHR → patient KG dataset
  - fl_client.py          → FL round (receive weights → train → send delta)

CAG (cache-augmented generation) removed — RAG only.

Usage:
    python main_client.py --config /etc/fmrag/client_config.yaml
"""

import argparse
import logging
import os
import yaml
import torch

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [FMRAG] %(message)s")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_kgc_triples(triple_store_path: str, ent2id: dict, rel2id: dict,
                      max_triples: int = 5000) -> "torch.Tensor | None":
    """
    Reads the same global_node_triple_store.txt AgenticRAG's LocalKGStore
    uses, maps (head, rel, tail) names to ids via the existing ent2id/rel2id
    vocab, and returns an (N, 3) LongTensor for KGCompletionModel training.
    Skips any triple whose head/rel/tail isn't already in the vocab.
    """
    if not os.path.exists(triple_store_path):
        return None

    triples = []
    with open(triple_store_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            if h in ent2id and r in rel2id and t in ent2id:
                triples.append((ent2id[h], rel2id[r], ent2id[t]))
            if len(triples) >= max_triples:
                break

    if not triples:
        return None
    return torch.tensor(triples, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="FMRAG Client Node")
    parser.add_argument("--config", default="config/client_config.yaml")
    args = parser.parse_args()
    cfg  = load_config(args.config)

    # ── One-flag dataset switch ───────────────────────────────────────────────
    # Set FMRAG_DATASET=synthea|mimic3 to switch datasets WITHOUT editing the
    # YAML. It overrides dataset name, root, graphcare.num_nodes, and cache in
    # the in-memory config. Roots/vocab can be overridden per-dataset with
    # FMRAG_DATA_ROOT / FMRAG_NUM_NODES if your paths differ from the defaults.
    _ds = os.environ.get("FMRAG_DATASET", "").lower()
    if _ds:
        _presets = {
            "synthea": {
                "dataset": "synthea",
                "root": "/data/synthea/csv/csv",
                "num_nodes": 260,
                "cache": "/data/synthea/cache/synthea_mortality.pkl",
            },
            "mimic3": {
                "dataset": "mimic3",
                "root": "/data/mimic-iii-demo",
                "num_nodes": 1335,
                "cache": "./data/cache/mimic3_demo_mortality.pkl",
            },
        }
        # allow aliases
        if _ds in ("mimic", "mimic-iii", "mimic3-demo"):
            _ds = "mimic3"
        if _ds not in _presets:
            raise SystemExit(f"FMRAG_DATASET='{_ds}' not recognized "
                             f"(use: synthea | mimic3)")
        p = _presets[_ds]
        cfg.setdefault("data", {})
        cfg.setdefault("graphcare", {})
        cfg["data"]["dataset"] = p["dataset"]
        cfg["data"]["root"] = os.environ.get("FMRAG_DATA_ROOT", p["root"])
        cfg["data"]["processed_cache"] = p["cache"]
        cfg["graphcare"]["num_nodes"] = int(
            os.environ.get("FMRAG_NUM_NODES", p["num_nodes"]))
        log.info("FMRAG_DATASET=%s -> dataset=%s root=%s num_nodes=%d",
                 _ds, cfg["data"]["dataset"], cfg["data"]["root"],
                 cfg["graphcare"]["num_nodes"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── 1. Build model ────────────────────────────────────────────────────────
    from client.peft.model_setup import build_model, get_tokenizer
    tokenizer = get_tokenizer()

    # Model-level baselines (RETAIN, G-BERT) via FMRAG_BASELINE_MODEL — replace
    # the PARC model entirely, federating all params. SCAFFOLD: validate first.
    _baseline_model = os.environ.get("FMRAG_BASELINE_MODEL", "").lower()
    if _baseline_model in ("retain", "gbert"):
        from client.fl.baseline_models import build_baseline_model
        model = build_baseline_model(
            _baseline_model, num_nodes=cfg["graphcare"]["num_nodes"])
        log.info("BASELINE MODEL active: %s (federating all params)",
                 _baseline_model)
    else:
        model = build_model(
            num_nodes    = cfg["graphcare"]["num_nodes"],
            num_rels     = cfg["graphcare"]["num_rels"],
            max_visit    = cfg["graphcare"]["max_visit"],
            hidden_dim   = cfg["graphcare"].get("hidden_dim", 128),
            out_channels = cfg["graphcare"].get("out_channels", 2),
            multitask    = (cfg["data"]["task"] == "multitask"),
        )

    if cfg["data"]["task"] == "multitask":
        os.environ.setdefault(
            "FMRAG_ADE_LOSS_WEIGHT",
            str(cfg.get("fl", {}).get("ade_loss_weight", 1.0)),
        )

    # ── 1b. Build KG completion model (ablation: kgc.enabled in config) ──────
    kgc = None
    kgc_triples = None
    kgc_cfg = cfg.get("kgc", {})
    if kgc_cfg.get("enabled", True):
        os.environ.setdefault("FMRAG_KGC_ENABLED", "true")
        os.environ.setdefault("FMRAG_KGC_DIM", str(kgc_cfg.get("dim", 64)))
        os.environ.setdefault("FMRAG_KGC_THRESHOLD", str(kgc_cfg.get("threshold", 0.5)))
        os.environ.setdefault("FMRAG_KGC_NEG_SAMPLES", str(kgc_cfg.get("neg_samples", 10)))

        from client.kgc.kg_completion import build_kgc_model
        kgc = build_kgc_model(
            num_nodes = cfg["graphcare"]["num_nodes"],
            num_rels  = cfg["graphcare"]["num_rels"],
        )
    else:
        os.environ["FMRAG_KGC_ENABLED"] = "false"
        log.info("KGC disabled via config — running ablation with completion off")

    # ── 2. Build Agentic RAG ──────────────────────────────────────────────────
    rag = None
    rag_cfg = cfg.get("rag", {})
    rag_backend = rag_cfg.get("backend", "umls")   # "umls" | "drkg"

    if rag_backend == "drkg":
        drkg_root = rag_cfg.get("drkg_root", "")
        if drkg_root and os.path.exists(os.path.join(drkg_root, "drkg.tsv")):
            try:
                from client.rag_cag.agentic_rag import AgenticRAG
                from client.rag_cag.drkg_loader import DRKGRetriever, DRKGStore
                retriever = DRKGRetriever(
                    drkg_root = drkg_root,
                    encoder   = model,
                    tokenizer = tokenizer,
                    device    = device,
                    name_emb_cache = rag_cfg.get("drkg_name_emb_cache"),
                )
                kg_store = DRKGStore(drkg_root)
                rag = AgenticRAG(
                    umls_retriever = retriever,   # DRKGRetriever has the same interface
                    kg_store       = kg_store,
                    encoder        = model,
                    tokenizer      = tokenizer,
                )
                log.info("Agentic RAG initialised (DRKG backend)")
            except Exception as e:
                log.warning("DRKG RAG init failed (continuing without RAG): %s", e)
        else:
            log.info("RAG skipped — DRKG not found at %s (expected drkg.tsv there)",
                     drkg_root or "not set")

    elif (rag_cfg.get("umls_emb_path")
            and os.path.exists(rag_cfg.get("umls_emb_path", ""))):
        try:
            from client.rag_cag.agentic_rag import (
                AgenticRAG, UMLSRetriever, LocalKGStore
            )
            umls = UMLSRetriever(
                umls_ent_emb_path = rag_cfg["umls_emb_path"],
                umls_names_path   = rag_cfg["umls_names_path"],
                device            = device,
            )
            kg_store = LocalKGStore(rag_cfg["triple_store_path"])
            rag = AgenticRAG(
                umls_retriever = umls,
                kg_store       = kg_store,
                encoder        = model,
                tokenizer      = tokenizer,
            )
            log.info("Agentic RAG initialised (UMLS backend)")
        except Exception as e:
            log.warning("RAG init failed (continuing without RAG): %s", e)
    else:
        log.info("RAG skipped — UMLS embeddings not found at %s",
                 rag_cfg.get("umls_emb_path", "not set"))

    # ── 3. Load EHR + build patient dataset ───────────────────────────────────
    dataset_name = cfg["data"]["dataset"]

    if dataset_name == "synthea":
        from client.graphcare.synthea_loader import SyntheaDataset
        _data_cfg = cfg["data"]
        dataset = SyntheaDataset(
            root       = _data_cfg["root"],
            task       = _data_cfg["task"],
            cache_path = _data_cfg.get("processed_cache",
                         "/data/synthea/cache/synthea_mortality.pkl"),
            client_id   = int(os.environ.get("FMRAG_CLIENT_ID",
                                             _data_cfg.get("client_id", 0))),
            num_clients = int(os.environ.get("FMRAG_NUM_CLIENTS",
                                             _data_cfg.get("num_clients", 1))),
            partition   = os.environ.get("FMRAG_PARTITION",
                                         _data_cfg.get("partition", "iid")),
        )
        log.info("Synthea dataset ready: %d patient graphs", len(dataset))

    elif dataset_name in ("mimic3", "mimic-iii", "mimic3-demo"):
        from client.graphcare.mimic_loader import MIMICDataset
        _data_cfg = cfg["data"]
        dataset = MIMICDataset(
            root       = _data_cfg["root"],
            task       = _data_cfg["task"],
            cache_path = _data_cfg.get("processed_cache",
                         "./data/cache/mimic3_demo_mortality.pkl"),
            client_id   = int(os.environ.get("FMRAG_CLIENT_ID",
                                             _data_cfg.get("client_id", 0))),
            num_clients = int(os.environ.get("FMRAG_NUM_CLIENTS",
                                             _data_cfg.get("num_clients", 1))),
            partition   = os.environ.get("FMRAG_PARTITION",
                                         _data_cfg.get("partition", "iid")),
        )
        log.info("MIMIC-III dataset ready: %d patient graphs", len(dataset))

    else:
        import json
        from client.graphcare.graphcare_pipeline import (
            load_ehr_dataset,
            FMRAGPatientDataset,
        )

        with open(cfg["graphcare"]["ent2id_path"]) as f:
            ent2id = json.load(f)
        with open(cfg["graphcare"]["rel2id_path"]) as f:
            rel2id = json.load(f)
        with open(cfg["graphcare"]["code_to_name_path"]) as f:
            code_to_name = json.load(f)

        ehr_ds = load_ehr_dataset(
            dataset_name    = dataset_name,
            task            = cfg["data"]["task"],
            data_root       = cfg["data"]["root"],
            processed_cache = cfg["data"].get("processed_cache"),
            disease_task    = cfg["data"].get("disease_task", "mortality"),
        )

        # ── 3a. Optionally fold in this client's own locally-observed cases ──
        # "Each user's input" — every institution can supply its own ADE
        # data on top of the shared MIMIC cohort. Stays local: only the
        # resulting FL deltas leave this client, never the CSV.
        user_cfg = cfg.get("user_data", {})
        if user_cfg.get("enabled", False):
            from client.graphcare.user_ade_loader import load_user_ade_csv, CombinedSampleDataset
            user_samples = load_user_ade_csv(user_cfg["path"]).samples
            ehr_ds = CombinedSampleDataset(ehr_ds.samples, user_samples)

        # ── 3c. Drug hypothesis generation (Contribution C5, optional) ───────
        hypothesis_engine = None
        hyp_cfg = cfg.get("hypothesis", {})
        if hyp_cfg.get("enabled", False) and rag is not None:
            try:
                from client.hypothesis.drug_hypothesis import HypothesisEngine
                kg_backend = hyp_cfg.get("kg_backend", rag_backend)  # default: match RAG
                if kg_backend == "drkg":
                    hypothesis_engine = HypothesisEngine(
                        kg_backend = "drkg",
                        drkg_root  = hyp_cfg.get("drkg_root", rag_cfg.get("drkg_root", "")),
                        max_hops   = hyp_cfg.get("max_hops", 3),
                    )
                else:
                    hypothesis_engine = HypothesisEngine(
                        umls_path     = hyp_cfg["umls_path"],
                        names_path    = rag_cfg["umls_names_path"],
                        atc_path      = hyp_cfg["atc_path"],
                        atc2umls_path = hyp_cfg["atc2umls_path"],
                        max_hops      = hyp_cfg.get("max_hops", 3),
                    )
                log.info("Drug hypothesis engine initialised (%s backend)", kg_backend)
            except Exception as e:
                log.warning("Hypothesis engine init failed (continuing without it): %s", e)
        elif hyp_cfg.get("enabled", False) and rag is None:
            log.warning("hypothesis.enabled=true but RAG failed to initialise — "
                         "hypothesis generation needs RAG's resolver, skipping")

        dataset = FMRAGPatientDataset(
            root         = cfg["data"]["pyg_root"],
            dataset_name = dataset_name,
            task         = cfg["data"]["task"],
            ehr_dataset  = ehr_ds,
            rag          = rag,
            ent2id       = ent2id,
            rel2id       = rel2id,
            code_to_name = code_to_name,
            tokenizer    = tokenizer,
            max_seq_len  = cfg["model"].get("max_seq_len", 64),
            kgc          = kgc,
            hypothesis_engine    = hypothesis_engine,
            hypothesis_top_k     = hyp_cfg.get("top_k", 5),
            hypothesis_log_path  = hyp_cfg.get("log_path", "./data/local/hypotheses_log.jsonl"),
        )

        # ── 3b. Load this client's local triples for federated KGC training ──
        if kgc is not None:
            kgc_triples = load_kgc_triples(
                triple_store_path = rag_cfg.get("triple_store_path", ""),
                ent2id            = ent2id,
                rel2id            = rel2id,
            )
            if kgc_triples is None:
                log.warning("KGC enabled but no usable triples found at %s — "
                             "completion training will be skipped this run",
                             rag_cfg.get("triple_store_path"))
            else:
                log.info("Loaded %d local triples for KGC training", kgc_triples.shape[0])

    # ── 4. FL training loop ───────────────────────────────────────────────────
    from client.fl.fl_client import run as fl_run
    fl_run(
        model       = model,
        dataset     = dataset,
        task        = cfg["data"]["task"],
        kgc         = kgc,
        kgc_triples = kgc_triples,
    )


if __name__ == "__main__":
    main()
