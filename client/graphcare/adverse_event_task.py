"""
client/graphcare/adverse_event_task.py
----------------------------------------
Adverse Drug Event (ADE) detection task for MIMIC-III/IV, in the same
pyhealth 1.1.4 function-based task_fn style as GraphCare's existing
mortality/readmission/drugrec/lenofstay tasks.

IMPORTANT — why this needs a *second* raw dataset load:

  graphcare_pipeline.load_ehr_dataset() applies code_mapping at dataset
  construction time: ICD9CM -> CCSCM, NDC -> ATC3. Once that mapping is
  applied, EVERY visit.get_code_list("DIAGNOSES_ICD") call returns the
  *mapped* CCS codes, not the raw ICD9 codes — there is no way to recover
  the original ICD9 E-codes from the already-mapped dataset instance.

  ADE labeling needs the raw ICD9-CM "E930-E949" range ("Drugs, medicinal
  and biological substances causing adverse effects in therapeutic use"
  — the standard ICD9-CM E-code block for ADEs, distinct from E950-E959
  which covers self-harm and is deliberately excluded here).

  So: build_raw_diagnosis_lookup() loads a SECOND MIMIC3Dataset/MIMIC4Dataset
  with no code_mapping on diagnoses, just to read raw E-codes per
  (patient_id, visit_id). make_adverse_event_task_fn() then returns a
  closure that uses this lookup for the *label* while still reading
  conditions/procedures/drugs from the (CCS/ATC-mapped) `patient` object
  passed in by ds.set_task() — so GraphCare's existing vocab/ent2id
  (CCSCM_CCSPROC_ATC3) keeps working unchanged for model features.

VERIFY BEFORE RELYING ON THIS:
  This was written against the pyhealth==1.1.4 API (Patient/Visit objects,
  visit.get_code_list(table=...), patient[i] indexing) by inspecting the
  pinned package directly, not by running it against real MIMIC data —
  there was no MIMIC access in the environment this was written in.
  Run a small `dev=True` load first and inspect `sample_ds.samples[:5]`
  before trusting this on a full run.
"""

from __future__ import annotations

import logging
from typing import Optional

from pyhealth.data import Patient, Visit

log = logging.getLogger(__name__)

# ── ADE-indicating raw ICD9-CM E-code block ──────────────────────────────────
# E930-E949: "Drugs, medicinal and biological substances causing adverse
# effects in therapeutic use" (the standard pharmacovigilance ADE block).
# Deliberately excludes E950-E959 (self-harm/suicide) and E960-E969
# (assault) — those are not adverse *drug* events in the therapeutic sense.
ADE_E_CODE_MIN = 930
ADE_E_CODE_MAX = 949


def is_ade_code(raw_icd9_code: str) -> bool:
    """True if a raw (unmapped) ICD9-CM code falls in the E930-E949 ADE block."""
    code = raw_icd9_code.strip().upper().lstrip("E").split(".")[0]
    if not code.isdigit():
        return False
    try:
        return ADE_E_CODE_MIN <= int(code[:3]) <= ADE_E_CODE_MAX
    except ValueError:
        return False


def build_raw_diagnosis_lookup(
    data_root: str,
    dataset_name: str = "mimic3",
    dev: bool = False,
) -> dict[tuple[str, str], list[str]]:
    """
    Loads a second, *unmapped* MIMIC dataset purely to read raw ICD9-CM
    diagnosis codes per (patient_id, visit_id), for ADE label computation.
    This does NOT replace the CCS-mapped dataset used for model features —
    use it alongside load_ehr_dataset(), not instead of it.

    Returns: {(patient_id, visit_id): [raw_icd9_code, ...]}
    """
    if dataset_name == "mimic3":
        from pyhealth.datasets import MIMIC3Dataset
        raw_ds = MIMIC3Dataset(
            root=data_root,
            tables=["DIAGNOSES_ICD"],
            code_mapping={},     # no mapping — keep raw ICD9-CM
            dev=dev,
        )
        dx_table = "DIAGNOSES_ICD"
    elif dataset_name == "mimic4":
        from pyhealth.datasets import MIMIC4Dataset
        raw_ds = MIMIC4Dataset(
            root=data_root,
            tables=["diagnoses_icd"],
            code_mapping={},
            dev=dev,
        )
        dx_table = "diagnoses_icd"
    else:
        raise ValueError(f"Unknown dataset for ADE raw lookup: {dataset_name}")

    lookup: dict[tuple[str, str], list[str]] = {}
    for patient in raw_ds.patients.values():
        for visit in patient:
            codes = visit.get_code_list(table=dx_table)
            lookup[(patient.patient_id, visit.visit_id)] = codes

    log.info("Built raw ICD9 lookup for ADE labeling — %d (patient, visit) pairs",
             len(lookup))
    return lookup


def make_adverse_event_task_fn(raw_dx_lookup: dict[tuple[str, str], list[str]]):
    """
    Returns a pyhealth-compatible task_fn(patient) closure for ADE detection,
    structured like GraphCare's existing mortality_prediction_mimic3_fn but
    labeling each *current* visit (not the next one — an ADE is observed
    within the visit it occurs in, unlike mortality which predicts forward).

    Usage:
        raw_lookup = build_raw_diagnosis_lookup(data_root, dataset_name)
        ade_fn = make_adverse_event_task_fn(raw_lookup)
        sample_ds = mapped_dataset.set_task(ade_fn)
    """

    def adverse_event_prediction_fn(patient: Patient):
        samples = []
        for i in range(len(patient)):
            visit: Visit = patient[i]

            raw_codes = raw_dx_lookup.get((patient.patient_id, visit.visit_id), [])
            label = int(any(is_ade_code(c) for c in raw_codes))

            conditions = visit.get_code_list(table="DIAGNOSES_ICD")
            procedures = visit.get_code_list(table="PROCEDURES_ICD")
            drugs      = visit.get_code_list(table="PRESCRIPTIONS")

            if len(conditions) * len(procedures) * len(drugs) == 0:
                continue

            samples.append({
                "visit_id":   visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs":      [drugs],
                "label":      label,
            })
        return samples

    return adverse_event_prediction_fn


def make_multitask_fn(disease_task_fn, raw_dx_lookup: dict[tuple[str, str], list[str]]):
    """
    Wraps ANY existing disease task_fn (mortality, readmission, etc.) and
    attaches an `ade_label` field to every sample it produces, without
    needing to know that task_fn's internal labeling logic — it just
    post-processes whatever samples come out.

    Usage:
        raw_lookup = build_raw_diagnosis_lookup(data_root, dataset_name)
        mt_fn = make_multitask_fn(mortality_prediction_mimic3_fn, raw_lookup)
        sample_ds = mapped_dataset.set_task(mt_fn)
        # sample_ds.samples[i] now has both "label" (disease) and
        # "ade_label" (adverse event) keys.

    Scope note: this combines a BINARY disease task (mortality,
    readmission) with ADE detection — both are binary, so a shared
    BCEWithLogitsLoss-per-head training setup in fl_client.py works
    cleanly. drugrec (multi-label) and lenofstay (multi-class) are NOT
    wired into multitask mode for this reason; combining a multi-label or
    multi-class disease task with a binary ADE task would need a separate
    loss-combination design that hasn't been built here.
    """

    def multitask_fn(patient):
        samples = disease_task_fn(patient)
        for s in samples:
            key = (s["patient_id"], s.get("visit_id", s["patient_id"]))
            raw_codes = raw_dx_lookup.get(key, [])
            s["ade_label"] = int(any(is_ade_code(c) for c in raw_codes))
        return samples

    return multitask_fn
