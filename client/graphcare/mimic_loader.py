"""
client/graphcare/mimic_loader.py
---------------------------------
Direct MIMIC-III → patient-graph loader that BYPASSES the GraphCare/UMLS
preprocessing pipeline (which needs a UMLS license and has hardcoded paths).

Built to mirror synthea_loader.py exactly: same Data fields, same
__init__ signature (including FL partitioning), so it drops into the
existing FL pipeline with only a config change (dataset: "mimic3").

Reads the MIMIC-III CSVs (uppercase names, e.g. /data/mimic3):
  PATIENTS.csv       : SUBJECT_ID, GENDER, DOB, DOD, ...
  ADMISSIONS.csv     : SUBJECT_ID, HADM_ID, ..., HOSPITAL_EXPIRE_FLAG
  DIAGNOSES_ICD.csv  : SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
  PROCEDURES_ICD.csv : SUBJECT_ID, HADM_ID, SEQ_NUM, ICD9_CODE
  PRESCRIPTIONS.csv  : SUBJECT_ID, HADM_ID, ..., DRUG, ..., NDC

Graph construction (per patient = SUBJECT_ID):
  - nodes  = the medical codes seen across the patient's admissions
             (ICD9 diagnoses prefixed "D_", ICD9 procedures "P_",
             drug names "M_"), mapped to a global integer vocab
  - edges  = co-occurrence within the same admission (HADM_ID)

Tasks (set task=... or data.task in config):
  - "mortality" (default): in-hospital mortality = 1 if ANY of the
    patient's admissions has HOSPITAL_EXPIRE_FLAG == 1, else 0.
  - "adverse_event": ADE = 1 if the patient has ANY raw ICD9-CM diagnosis
    code in the E930-E949 block ("drugs/medicinal substances causing
    adverse effects in therapeutic use"). Because this loader reads raw
    ICD9 codes directly (no CCS mapping), the E-codes are present in
    DIAGNOSES_ICD.csv and need no pyhealth / second dataset load.

PERFORMANCE / MEMORY:
  - Only the columns actually used are read from each CSV. Full-MIMIC
    PRESCRIPTIONS has ~19 columns over 4M rows; reading all of them
    exhausts a small VM's RAM. We read only SUBJECT_ID, HADM_ID, DRUG.
  - Tables are grouped by patient ONCE (single pass), so building graphs
    for all 46k+ patients is O(rows), not O(patients*rows) — minutes not
    hours.

IMPORTANT — vocab/num_nodes:
  As with Synthea, the vocabulary is built from ALL patients so every
  client shares the same code space. Set graphcare.num_nodes in
  client_config.yaml (and FMRAG_NUM_NODES) to the "Vocabulary size: N
  codes" this logs on first run (and graphcare.max_visit to _MAX_VISIT).

IMPORTANT — cache per task:
  Use a DIFFERENT processed_cache filename for each task (e.g.
  mimic3_ade.pkl vs mimic3_mortality.pkl). The cache stores the labels, so
  reusing a mortality cache for an ADE run would silently train on the
  wrong target.
"""

from __future__ import annotations

import logging
import os
import pickle

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

log = logging.getLogger(__name__)

_MAX_VISIT = 10          # must match config graphcare.max_visit
_MAX_NODES = 64          # cap nodes per patient graph (same as synthea)

# Only load the columns we actually use. MIMIC PRESCRIPTIONS has ~19 columns
# over ~4M rows; reading all of them exhausts a small VM. Keys are the
# UPPERCASE canonical names; the loader maps to the file's real casing.
_USECOLS = {
    "PATIENTS.csv":       ["SUBJECT_ID"],
    "ADMISSIONS.csv":     ["SUBJECT_ID", "HADM_ID", "HOSPITAL_EXPIRE_FLAG"],
    "DIAGNOSES_ICD.csv":  ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    "PROCEDURES_ICD.csv": ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    "PRESCRIPTIONS.csv":  ["SUBJECT_ID", "HADM_ID", "DRUG"],
}


# ── Adverse Drug Event (ADE) labeling ────────────────────────────────────────
# E930-E949 = "Drugs/medicinal substances causing adverse effects in
# therapeutic use" (standard pharmacovigilance ADE block). Excludes E950-E959
# (self-harm). These are RAW ICD9-CM E-codes, present directly in
# DIAGNOSES_ICD.csv (this loader does no CCS mapping, so they're available).
_ADE_E_MIN = 930
_ADE_E_MAX = 949


def _is_ade_code(raw_icd9: str) -> bool:
    """True if a raw ICD9-CM code is in the E930-E949 ADE block."""
    code = str(raw_icd9).strip().upper()
    if not code.startswith("E"):
        return False
    digits = code[1:].split(".")[0]
    if len(digits) < 3 or not digits[:3].isdigit():
        return False
    return _ADE_E_MIN <= int(digits[:3]) <= _ADE_E_MAX


def build_vocab(diag: pd.DataFrame,
                proc: pd.DataFrame,
                pres: pd.DataFrame) -> dict:
    """Global code → contiguous id map across all patients.

    Codes are namespaced so a diagnosis ICD9 and a procedure ICD9 with the
    same digits don't collide: "D_<icd9>", "P_<icd9>", "M_<drug>".
    """
    codes = set()
    for c in diag["ICD9_CODE"].dropna().astype(str):
        codes.add("D_" + c)
    for c in proc["ICD9_CODE"].dropna().astype(str):
        codes.add("P_" + c)
    for d in pres["DRUG"].dropna().astype(str):
        codes.add("M_" + d.strip())
    return {code: i for i, code in enumerate(sorted(codes))}


def build_patient_graph_fast(pt_diag, pt_proc, pt_pres,
                             code2id: dict,
                             max_nodes: int = _MAX_NODES) -> "Data | None":
    """One PyG Data object per patient; edges = same-admission co-occurrence.

    pt_diag / pt_proc / pt_pres are lists of (code, hadm) tuples for THIS
    patient (already extracted via a one-time groupby -- no dataframe
    filtering here, so building all patients is O(rows) not O(patients*rows)).
    """
    if not pt_diag and not pt_proc and not pt_pres:
        return None

    codes: list = []
    adm_map: dict = {}   # HADM_ID → node indices

    def _add(rows, prefix):
        for raw, hadm in rows:
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            code = prefix + str(raw).strip()
            cid = code2id.get(code)
            if cid is None:
                continue
            idx = len(codes)
            codes.append(cid)
            adm_map.setdefault(str(hadm), []).append(idx)

    _add(pt_diag, "D_")
    _add(pt_proc, "P_")
    _add(pt_pres, "M_")

    if not codes:
        return None

    codes = codes[:max_nodes]
    x = torch.tensor(codes, dtype=torch.long).unsqueeze(1).float()

    src, dst = [], []
    for adm_codes in adm_map.values():
        adm_codes = [c for c in adm_codes if c < max_nodes]
        for i in range(len(adm_codes)):
            for j in range(i + 1, len(adm_codes)):
                src.extend([adm_codes[i], adm_codes[j]])
                dst.extend([adm_codes[j], adm_codes[i]])

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        n = len(codes)
        edge_index = torch.tensor([list(range(n)), list(range(n))],
                                  dtype=torch.long)

    return Data(x=x, edge_index=edge_index, num_nodes=len(codes))


class MIMICDataset(Dataset):
    """
    PyG Dataset wrapping MIMIC-III CSVs. Same interface as
    SyntheaDataset, including federated partitioning by client_id.
    Supports task in {"mortality", "adverse_event"}.
    """

    def __init__(self,
                 root: str,
                 task: str = "mortality",
                 cache_path: "str | None" = None,
                 transform=None,
                 client_id: int = 0,
                 num_clients: int = 1,
                 partition: str = "iid"):
        self.data_root  = root
        self.task       = task
        self.cache_path = cache_path
        self.client_id   = client_id
        self.num_clients  = max(1, num_clients)
        self.partition    = partition
        self._graphs: list = []
        self._load()
        super().__init__(root, transform)

    # Partitioning is identical to synthea_loader's (kept in sync).
    def _partition_graphs(self, graphs: list) -> list:
        if self.num_clients <= 1:
            return graphs
        import random
        n = len(graphs)
        if self.partition == "label_skew":
            graphs = sorted(graphs, key=lambda g: int(g.y.item()))
        elif self.partition == "dirichlet":
            import numpy as np
            alpha = float(os.environ.get("FMRAG_DIRICHLET_ALPHA", "0.5"))
            rng = np.random.default_rng(42)
            by_label: dict = {}
            for g in graphs:
                by_label.setdefault(int(g.y.item()), []).append(g)
            client_bins: list = [[] for _ in range(self.num_clients)]
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
            rng = random.Random(42)
            graphs = graphs[:]
            rng.shuffle(graphs)
        base = n // self.num_clients
        rem  = n %  self.num_clients
        start = self.client_id * base + min(self.client_id, rem)
        count = base + (1 if self.client_id < rem else 0)
        shard = graphs[start:start + count]
        log.info("FL partition (%s): client %d/%d gets %d of %d patient graphs",
                 self.partition, self.client_id, self.num_clients, len(shard), n)
        return shard

    def _load(self):
        if self.cache_path and os.path.isfile(self.cache_path):
            log.info("Loading cached MIMIC dataset from %s", self.cache_path)
            with open(self.cache_path, "rb") as f:
                self._graphs = pickle.load(f)
            log.info("Loaded %d patient graphs from cache", len(self._graphs))
            self._graphs = self._partition_graphs(self._graphs)
            return

        log.info("Building MIMIC-III dataset from %s (task=%s)",
                 self.data_root, self.task)

        def _csv(name):
            # Read only the columns we use. First read the header (nrows=0) to
            # learn the file's real column casing (demo=lowercase,
            # full=uppercase), then select just the needed columns.
            path = os.path.join(self.data_root, name)
            head = pd.read_csv(path, nrows=0)
            colmap = {c.upper(): c for c in head.columns}
            want_upper = _USECOLS.get(name)
            usecols = ([colmap[u] for u in want_upper if u in colmap]
                       if want_upper else None)
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
            df.columns = [c.upper() for c in df.columns]
            return df

        patients   = _csv("PATIENTS.csv")
        admissions = _csv("ADMISSIONS.csv")
        diag       = _csv("DIAGNOSES_ICD.csv")
        proc       = _csv("PROCEDURES_ICD.csv")
        pres       = _csv("PRESCRIPTIONS.csv")

        log.info("Loaded: %d patients, %d admissions, %d diagnoses, "
                 "%d procedures, %d prescriptions",
                 len(patients), len(admissions), len(diag), len(proc), len(pres))

        code2id = build_vocab(diag, proc, pres)
        _num_nodes = len(code2id)
        log.info("Vocabulary size: %d codes", _num_nodes)

        # ── Per-patient label, depending on the task ─────────────────────────
        if self.task == "adverse_event":
            # ADE label per patient: 1 if ANY diagnosis is an E930-E949 code.
            # Raw ICD9 codes are read directly here (no CCS mapping), so the
            # E-codes are available -- no pyhealth / second dataset needed.
            diag["_IS_ADE"] = diag["ICD9_CODE"].map(_is_ade_code)
            label_map = (diag.groupby("SUBJECT_ID")["_IS_ADE"]
                         .max().astype(int))
            log.info("Adverse event (E930-E949): %d positive, %d negative",
                     int((label_map == 1).sum()), int((label_map == 0).sum()))
        else:
            # In-hospital mortality label: 1 if any admission expired.
            label_map = (admissions.groupby("SUBJECT_ID")["HOSPITAL_EXPIRE_FLAG"]
                         .max().astype(int))
            log.info("In-hospital mortality: %d died, %d survived",
                     int((label_map == 1).sum()), int((label_map == 0).sum()))

        # ── Pre-group each table by patient ONCE (O(rows), not O(patients*rows)) ─
        log.info("Grouping tables by patient (one-time)...")
        diag_by_pt: dict = {}
        for sid, code, hadm in zip(diag["SUBJECT_ID"], diag["ICD9_CODE"],
                                   diag["HADM_ID"]):
            diag_by_pt.setdefault(sid, []).append((code, hadm))
        proc_by_pt: dict = {}
        for sid, code, hadm in zip(proc["SUBJECT_ID"], proc["ICD9_CODE"],
                                   proc["HADM_ID"]):
            proc_by_pt.setdefault(sid, []).append((code, hadm))
        pres_by_pt: dict = {}
        for sid, drug, hadm in zip(pres["SUBJECT_ID"], pres["DRUG"],
                                   pres["HADM_ID"]):
            pres_by_pt.setdefault(sid, []).append((drug, hadm))
        log.info("Grouped. Building graphs for %d patients...",
                 patients["SUBJECT_ID"].nunique())

        # Free the big dataframes we no longer need (grouping dicts hold the data now).
        del pres, proc

        graphs = []
        for sid in patients["SUBJECT_ID"].unique():
            label = int(label_map.get(sid, 0))
            graph = build_patient_graph_fast(
                diag_by_pt.get(sid, []),
                proc_by_pt.get(sid, []),
                pres_by_pt.get(sid, []),
                code2id,
            )
            if graph is None:
                continue

            graph.y          = torch.tensor([label], dtype=torch.float)
            graph.patient_id = str(sid)

            graph.input_ids      = torch.zeros(1, 64, dtype=torch.long)
            graph.attention_mask = torch.zeros(1, 64, dtype=torch.long)

            graph.node_ids = graph.x.reshape(-1).long()
            graph.rel_ids  = torch.zeros(graph.edge_index.shape[1], dtype=torch.long)

            _vn = torch.zeros(1, _MAX_VISIT, _num_nodes, dtype=torch.float)
            _node_ids = graph.node_ids.tolist()
            _node_ids = _node_ids if isinstance(_node_ids, list) else [_node_ids]
            for _i, _code_idx in enumerate(_node_ids[:_MAX_VISIT]):
                if _code_idx < _num_nodes:
                    _vn[0, _i, _code_idx] = 1.0
            graph.visit_node = _vn

            _ehr = torch.zeros(_num_nodes, dtype=torch.float)
            for _code in _node_ids:
                if _code < _num_nodes:
                    _ehr[_code] = 1.0
            graph.ehr_nodes = _ehr.unsqueeze(0)

            graphs.append(graph)

        log.info("Built %d patient graphs", len(graphs))

        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(graphs, f)
            log.info("Cached to %s", self.cache_path)

        self._graphs = self._partition_graphs(graphs)

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> "Data":
        return self._graphs[idx]

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx):
        return self._graphs[idx]
