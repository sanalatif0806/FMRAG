"""
client/graphcare/user_ade_loader.py
--------------------------------------
Lets each client (hospital/institution — "user" of the federated system)
supply their OWN locally observed adverse-event-labeled visits, on top of
the shared MIMIC-III/IV backbone. This stays entirely local: the CSV never
leaves the client, only the resulting model deltas do, same as everything
else in this codebase.

Why this matters for the "personalized" framing: every client trains on
the same shared MIMIC cohort PLUS whatever real cases that institution has
actually observed, so each client's local model (and the federated
average) reflects both the shared public benchmark and the participating
institutions' real, private experience — without those institutions ever
exposing the raw records.

Expected CSV schema (one row per visit):
    patient_id, visit_id, conditions, procedures, drugs, adverse_event

  - conditions / procedures / drugs: semicolon-separated codes IN THE SAME
    VOCABULARY as the shared GraphCare ent2id (i.e. CCSCM / CCSPROC / ATC3
    codes, not raw ICD9/NDC — map them yourself before exporting this CSV,
    or extend this loader to do so if your institution's EHR exports raw
    codes). Codes not already in ent2id get new ids assigned dynamically,
    exactly like merge_rag_triples() does for AgenticRAG triples.
  - adverse_event: 0 or 1

This was written without a real example CSV to validate column-handling
edge cases against (missing codes, encoding issues, etc.) — treat the
parsing as a starting point and test against your institution's actual
export format before trusting it on real data.
"""

from __future__ import annotations

import csv
import logging
import os

log = logging.getLogger(__name__)


class UserSampleSet:
    """
    Minimal stand-in with a `.samples` list in the exact schema
    FMRAGPatientDataset.process() expects (same keys pyhealth task_fns
    produce): patient_id, conditions, procedures, drugs, label.
    """

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)


def load_user_ade_csv(path: str) -> UserSampleSet:
    """Parses a client's local ADE CSV into the shared sample schema."""
    if not os.path.exists(path):
        log.warning("User ADE data file not found at %s — skipping", path)
        return UserSampleSet([])

    samples = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"patient_id", "visit_id", "conditions", "procedures", "drugs", "adverse_event"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            log.error("User ADE CSV %s missing required columns: %s — skipping file",
                      path, missing)
            return UserSampleSet([])

        for row in reader:
            conditions = [c.strip() for c in row["conditions"].split(";") if c.strip()]
            procedures = [c.strip() for c in row["procedures"].split(";") if c.strip()]
            drugs      = [c.strip() for c in row["drugs"].split(";") if c.strip()]

            if not (conditions and procedures and drugs):
                continue   # same exclusion rule as the pyhealth task_fns

            try:
                label = int(row["adverse_event"])
            except (KeyError, ValueError):
                log.warning("Skipping row with invalid adverse_event value: %r", row)
                continue

            samples.append({
                "visit_id":   row["visit_id"],
                "patient_id": row["patient_id"],
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs":      [drugs],
                "label":      label,
            })

    log.info("Loaded %d local user-supplied ADE samples from %s", len(samples), path)
    return UserSampleSet(samples)


class CombinedSampleDataset:
    """
    Concatenates the shared MIMIC-derived samples with this client's own
    locally-supplied samples into one `.samples` list, so it can be passed
    straight into FMRAGPatientDataset(ehr_dataset=combined, ...) unchanged.
    """

    def __init__(self, mimic_samples: list[dict], user_samples: list[dict]):
        self.samples = list(mimic_samples) + list(user_samples)
        log.info("Combined dataset: %d MIMIC samples + %d user samples = %d total",
                  len(mimic_samples), len(user_samples), len(self.samples))

    def __len__(self):
        return len(self.samples)
