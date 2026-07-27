"""
client/graphcare/synthea_loader.py
------------------------------------
Loads Synthea CSV data and converts it to a GraphCare-compatible
PyTorch Geometric dataset for FL training.

Synthea columns used:
  patients.csv   : Id, BIRTHDATE, DEATHDATE, GENDER, RACE
  conditions.csv : PATIENT, ENCOUNTER, CODE, DESCRIPTION, START, STOP
  medications.csv: PATIENT, ENCOUNTER, CODE, DESCRIPTION, START, STOP
  encounters.csv : Id, PATIENT, START, STOP, CODE, DESCRIPTION

Task: mortality prediction
  Label 1 = patient has a DEATHDATE (died)
  Label 0 = patient is alive
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

log = logging.getLogger(__name__)


# ── Synthea ICD-like code vocabulary ─────────────────────────────────────────

def build_vocab(conditions: pd.DataFrame,
                medications: pd.DataFrame) -> tuple[dict, dict]:
    """Build code→id and description→id vocabularies."""
    all_codes = (
        list(conditions["CODE"].astype(str).unique()) +
        list(medications["CODE"].astype(str).unique())
    )
    code2id = {c: i for i, c in enumerate(sorted(set(all_codes)))}

    all_descs = (
        list(conditions["DESCRIPTION"].astype(str).unique()) +
        list(medications["DESCRIPTION"].astype(str).unique())
    )
    desc2id = {d: i for i, d in enumerate(sorted(set(all_descs)))}

    return code2id, desc2id


# ── Per-patient graph builder ─────────────────────────────────────────────────

def build_patient_graph(patient_id: str,
                        conditions: pd.DataFrame,
                        medications: pd.DataFrame,
                        encounters: pd.DataFrame,
                        code2id: dict,
                        max_nodes: int = 64) -> Data | None:
    """
    Build a PyG Data object for one patient.
    Nodes = medical codes (conditions + medications)
    Edges = co-occurrence within same encounter
    """
    pt_cond = conditions[conditions["PATIENT"] == patient_id]
    pt_meds = medications[medications["PATIENT"] == patient_id]

    if len(pt_cond) == 0 and len(pt_meds) == 0:
        return None

    # Collect all codes for this patient
    codes = []
    enc_map = {}  # encounter_id → list of code indices

    for _, row in pt_cond.iterrows():
        code = str(row["CODE"])
        if code in code2id:
            idx = len(codes)
            codes.append(code2id[code])
            enc_id = str(row["ENCOUNTER"])
            enc_map.setdefault(enc_id, []).append(idx)

    for _, row in pt_meds.iterrows():
        code = str(row["CODE"])
        if code in code2id:
            idx = len(codes)
            codes.append(code2id[code])
            enc_id = str(row["ENCOUNTER"])
            enc_map.setdefault(enc_id, []).append(idx)

    if not codes:
        return None

    # Truncate to max_nodes
    codes = codes[:max_nodes]

    # Node features: one-hot-like code embedding (just the code id as feature)
    x = torch.tensor(codes, dtype=torch.long).unsqueeze(1).float()

    # Edges: connect codes that appear in the same encounter
    src, dst = [], []
    for enc_codes in enc_map.values():
        enc_codes = [c for c in enc_codes if c < max_nodes]
        for i in range(len(enc_codes)):
            for j in range(i + 1, len(enc_codes)):
                src.extend([enc_codes[i], enc_codes[j]])
                dst.extend([enc_codes[j], enc_codes[i]])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        # No edges — self-loops
        n = len(codes)
        edge_index = torch.tensor([list(range(n)), list(range(n))],
                                  dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=len(codes))


# ── Main dataset class ────────────────────────────────────────────────────────

class SyntheaDataset(Dataset):
    """
    PyG Dataset wrapping Synthea CSV data.
    Compatible with GraphCare FL training pipeline.
    """

    def __init__(self,
                 root: str,
                 task: str = "mortality",
                 cache_path: str | None = None,
                 transform=None,
                 client_id: int = 0,
                 num_clients: int = 1,
                 partition: str = "iid"):
        self.data_root  = root
        self.task       = task
        self.cache_path = cache_path
        # Federated partitioning: each client keeps only its own disjoint
        # slice of patients. client_id in [0, num_clients). num_clients=1
        # (the default) means no partitioning — the whole dataset, exactly
        # as before. partition: "iid" (random even split) or "label_skew"
        # (non-IID — clients get skewed mortality-label proportions, the
        # realistic hospital-heterogeneity setting).
        self.client_id   = client_id
        self.num_clients  = max(1, num_clients)
        self.partition    = partition
        self._graphs: list[Data] = []
        self._load()
        super().__init__(root, transform)

    def _partition_graphs(self, graphs: list) -> list:
        """Keep only this client's disjoint slice of the patient graphs.

        IMPORTANT: the vocabulary/num_nodes was already built from the FULL
        patient set before this runs, so every client shares the same code
        space (the model's num_nodes matches across clients). Only the
        *patients used for training* are partitioned here.
        """
        if self.num_clients <= 1:
            return graphs

        import random
        n = len(graphs)

        if self.partition == "label_skew":
            # Non-IID (HARD skew): sort by label so consecutive shards have
            # skewed death/survival ratios. WARNING: with few clients this
            # can give a client only ONE class (e.g. 0% mortality), which
            # makes that client's binary task degenerate. Fine as a quick
            # heterogeneity demo, but for paper-quality non-IID use the
            # "dirichlet" mode below, which is the FL-literature standard.
            graphs = sorted(graphs, key=lambda g: int(g.y.item()))
        elif self.partition == "dirichlet":
            # Non-IID (TUNABLE skew): Dirichlet(alpha) over labels — the
            # standard non-IID FL partition (Hsu et al. 2019). Lower alpha =
            # more skew; alpha→inf approaches IID. alpha=0.5 is a common
            # moderately-heterogeneous setting. Each client gets a mix of
            # both classes, just in skewed proportions, avoiding the
            # single-class degeneracy of label_skew.
            import numpy as np
            alpha = float(os.environ.get("FMRAG_DIRICHLET_ALPHA", "0.5"))
            rng = np.random.default_rng(42)
            by_label: dict[int, list] = {}
            for g in graphs:
                by_label.setdefault(int(g.y.item()), []).append(g)
            client_bins: list[list] = [[] for _ in range(self.num_clients)]
            for label, items in by_label.items():
                # shuffle by INDEX (never hand graph objects to numpy)
                idx = list(range(len(items)))
                rng.shuffle(idx)
                items = [items[i] for i in idx]
                proportions = rng.dirichlet([alpha] * self.num_clients)
                counts = (proportions * len(items)).astype(int)
                while counts.sum() < len(items):
                    counts[counts.argmin()] += 1
                while counts.sum() > len(items):
                    counts[counts.argmax()] -= 1
                pos = 0
                for ci in range(self.num_clients):
                    client_bins[ci].extend(items[pos:pos + counts[ci]])
                    pos += counts[ci]
            shard = client_bins[self.client_id]
            log.info("FL partition (dirichlet a=%.2f): client %d/%d gets %d "
                     "of %d patient graphs", alpha, self.client_id,
                     self.num_clients, len(shard), len(graphs))
            return shard
        else:
            # IID: deterministic shuffle (fixed seed so the split is stable
            # and reproducible across runs), then even contiguous slices.
            rng = random.Random(42)
            graphs = graphs[:]
            rng.shuffle(graphs)

        # Even contiguous slices; remainder spread across the first clients.
        base = n // self.num_clients
        rem  = n %  self.num_clients
        start = self.client_id * base + min(self.client_id, rem)
        count = base + (1 if self.client_id < rem else 0)
        shard = graphs[start:start + count]
        log.info("FL partition (%s): client %d/%d gets %d of %d patient graphs",
                 self.partition, self.client_id, self.num_clients, len(shard), n)
        return shard

    def _load(self):
        # NOTE: the cache stores the FULL dataset (all patients); the
        # per-client partition is applied AFTER loading, so one cache file
        # is reused by every client and each still trains on its own slice.
        if self.cache_path and os.path.isfile(self.cache_path):
            log.info("Loading cached Synthea dataset from %s", self.cache_path)
            with open(self.cache_path, "rb") as f:
                self._graphs = pickle.load(f)
            log.info("Loaded %d patient graphs from cache", len(self._graphs))
            self._graphs = self._partition_graphs(self._graphs)
            return

        log.info("Building Synthea dataset from %s", self.data_root)

        patients   = pd.read_csv(os.path.join(self.data_root, "patients.csv"))
        conditions = pd.read_csv(os.path.join(self.data_root, "conditions.csv"))
        medications= pd.read_csv(os.path.join(self.data_root, "medications.csv"))
        encounters = pd.read_csv(os.path.join(self.data_root, "encounters.csv"))

        log.info("Loaded: %d patients, %d conditions, %d medications",
                 len(patients), len(conditions), len(medications))

        code2id, _ = build_vocab(conditions, medications)
        _num_nodes = len(code2id)   # total vocab size
        _max_visit = 10             # must match config graphcare.max_visit
        log.info("Vocabulary size: %d codes", _num_nodes)

        # Mortality label: 1 if patient has a death date
        patients["label"] = patients["DEATHDATE"].notna().astype(int)
        log.info("Mortality labels: %d died, %d alive",
                 patients["label"].sum(),
                 (patients["label"] == 0).sum())

        graphs = []
        for _, row in patients.iterrows():
            pid   = str(row["Id"])
            label = int(row["label"])

            graph = build_patient_graph(
                pid, conditions, medications, encounters, code2id
            )
            if graph is None:
                continue

            graph.y          = torch.tensor([label], dtype=torch.float)
            graph.patient_id = pid

            # Extra features for compatibility with fl_client.py
            graph.input_ids      = torch.zeros(1, 64, dtype=torch.long)
            graph.attention_mask = torch.zeros(1, 64, dtype=torch.long)

            # Change 4d: node_ids never scalar (reshape not squeeze)
            graph.node_ids = graph.x.reshape(-1).long()

            # Change 4a: rel_ids per-edge not per-node
            graph.rel_ids = torch.zeros(
                graph.edge_index.shape[1], dtype=torch.long
            )

            # Change 4b: visit_node shape (1, max_visit, num_nodes)
            _vn = torch.zeros(1, _max_visit, _num_nodes, dtype=torch.float)
            _node_ids = graph.node_ids.tolist()
            _node_ids = _node_ids if isinstance(_node_ids, list) else [_node_ids]
            for _i, _code_idx in enumerate(_node_ids[:_max_visit]):
                if _code_idx < _num_nodes:
                    _vn[0, _i, _code_idx] = 1.0
            graph.visit_node = _vn

            # Change 4c: ehr_nodes shape (1, num_nodes) for correct PyG batching
            _ehr = torch.zeros(_num_nodes, dtype=torch.float)
            for _code in _node_ids:
                if _code < _num_nodes:
                    _ehr[_code] = 1.0
            graph.ehr_nodes = _ehr.unsqueeze(0)  # (1, num_nodes)

            graphs.append(graph)

        log.info("Built %d patient graphs", len(graphs))

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(graphs, f)
            log.info("Cached to %s", self.cache_path)

        # Partition AFTER caching the full set, so the cache is client-agnostic
        # and each client still trains on only its own slice.
        self._graphs = self._partition_graphs(graphs)

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx]

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx):
        return self._graphs[idx]


def load_synthea(config: dict) -> SyntheaDataset:
    """Load Synthea dataset from client_config.yaml data section.

    Federated partitioning is controlled by (in priority order) env vars
    then the config's data section:
      FMRAG_CLIENT_ID    / data.client_id     (default 0)
      FMRAG_NUM_CLIENTS  / data.num_clients    (default 1 = no partition)
      FMRAG_PARTITION    / data.partition      ("iid" | "label_skew")
    Set a distinct FMRAG_CLIENT_ID on each client (0, 1, 2, …) and the same
    FMRAG_NUM_CLIENTS everywhere, so each trains on a disjoint slice.
    """
    root       = config.get("root", "/data/synthea/csv")
    task       = config.get("task", "mortality")
    cache_path = config.get("processed_cache",
                            "/data/synthea/cache/synthea_mortality.pkl")

    client_id   = int(os.environ.get("FMRAG_CLIENT_ID",
                                     config.get("client_id", 0)))
    num_clients = int(os.environ.get("FMRAG_NUM_CLIENTS",
                                     config.get("num_clients", 1)))
    partition   = os.environ.get("FMRAG_PARTITION",
                                 config.get("partition", "iid"))

    return SyntheaDataset(root=root, task=task, cache_path=cache_path,
                          client_id=client_id, num_clients=num_clients,
                          partition=partition)
