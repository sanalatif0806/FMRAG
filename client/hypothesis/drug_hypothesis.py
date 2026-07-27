"""
FMRAG drug_hypothesis.py  —  Contribution C5
---------------------------------------------
"Mechanistically grounded therapeutic hypotheses through KG traversal"
                                        — Doctoral Consortium Paper, §3

Architecture
------------
1. HypothesisKG  : loads UMLS triples filtered to drug-relevant relations
                   (may_treat, may_cause, is_a_risk_factor_of,
                    interacts_with, may_contraindicate, has_active_ingredient)
                   + ATC drug catalogue (name, indication, class)

2. PathFinder    : BFS over (patient_condition → ... → candidate_drug)
                   up to MAX_HOPS, collecting mechanistic evidence paths

3. HypothesisScorer : ranks candidate drugs by path evidence quality:
                      - path length (shorter = more direct)
                      - relation type weights (may_treat > is_a_risk_factor_of)
                      - contraindication penalty (existing patient drugs)
                      - interaction penalty (existing patient drugs)

4. HypothesisGenerator : converts top-k scored paths into structured
                         natural language hypotheses + evidence summaries

Privacy guarantee: all traversal happens locally on the client node.
Only the ranked hypothesis text (no patient identifiers) may be exported.

Usage
-----
    from client.hypothesis.drug_hypothesis import HypothesisEngine

    engine = HypothesisEngine(
        umls_path   = "KG_mapping/umls/umls.csv",
        names_path  = "KG_mapping/umls/concept_names.txt",
        atc_path    = "resources/ATC.csv",
        atc2umls_path = "KG_mapping/ATC_to_UMLS.csv",
    )

    hypotheses = engine.generate(
        patient_conditions = ["C0011860", "C0020557"],   # UMLS CUIs
        patient_drugs      = ["C0000545"],               # current medications
        top_k              = 5,
    )
    for h in hypotheses:
        print(h.narrative)
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ── Relation weights for hypothesis scoring ───────────────────────────────────
RELATION_WEIGHTS: dict[str, float] = {
    "may_treat":              1.00,
    "has_active_ingredient":  0.85,
    "is_a_risk_factor_of":    0.70,
    "may_cause":              0.55,   # drug may cause condition → repurposing signal
    "interacts_with":         0.40,
    "may_contraindicate":    -1.50,   # hard penalty applied separately
    "isa":                    0.20,
    "is_a_subtype_of":        0.20,
    "belongs_to_drug_super-family": 0.30,
    "belongs_to_the_drug_family_of": 0.30,
}

# Relations that indicate a drug → condition direction
TREATMENT_RELATIONS = {"may_treat", "has_active_ingredient"}
RISK_RELATIONS      = {"is_a_risk_factor_of", "may_cause"}
SAFETY_RELATIONS    = {"interacts_with", "may_contraindicate"}

MAX_HOPS     = 3
MAX_CANDIDATES = 50   # max drug candidates per traversal
TOP_K_DEFAULT  = 5


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvidencePath:
    """A single traversal path from a patient condition to a candidate drug."""
    source_condition: str          # UMLS CUI of starting condition
    target_drug:      str          # UMLS CUI of candidate drug
    path_nodes:       list[str]    # [source, intermediate..., target]
    path_relations:   list[str]    # relations between consecutive nodes
    raw_score:        float = 0.0
    penalised:        bool  = False  # True if contraindicated/interacting

    @property
    def length(self) -> int:
        return len(self.path_relations)

    def hop_summary(self, id2name: dict) -> str:
        parts = [id2name.get(self.path_nodes[0], self.path_nodes[0])]
        for rel, node in zip(self.path_relations, self.path_nodes[1:]):
            parts.append(f"--[{rel}]--> {id2name.get(node, node)}")
        return " ".join(parts)


@dataclass
class DrugHypothesis:
    """Ranked hypothesis for a candidate drug with supporting evidence."""
    drug_cui:       str
    drug_name:      str
    drug_class:     str
    indication:     str
    score:          float
    evidence_paths: list[EvidencePath]
    contraindicated: bool = False
    interactions:   list[str] = field(default_factory=list)
    narrative:      str = ""

    def __post_init__(self):
        if not self.narrative:
            self.narrative = self._build_narrative()

    def _build_narrative(self) -> str:
        status = " [CAUTION: possible contraindication]" if self.contraindicated else ""
        top_path = self.evidence_paths[0] if self.evidence_paths else None
        evidence = ""
        if top_path:
            rels = " → ".join(top_path.path_relations)
            evidence = f" Evidence pathway: {rels}."
        interactions = ""
        if self.interactions:
            interactions = f" Known interactions with current medications: {', '.join(self.interactions[:3])}."
        return (
            f"Hypothesis{status}: {self.drug_name} ({self.drug_class}) "
            f"may be therapeutically relevant for this patient. "
            f"Indication: {self.indication or 'not specified'}.{evidence}{interactions} "
            f"Confidence score: {self.score:.3f}."
        )


# ── HypothesisKG ──────────────────────────────────────────────────────────────

class HypothesisKG:
    """
    Loads the subset of UMLS relevant to drug hypothesis generation.
    Builds two indexes:
      - forward_adj[node] → {neighbor: relation}  (for BFS traversal)
      - reverse_drug_idx[drug_cui] → list of (condition_cui, relation)
    Also loads:
      - id2name : UMLS CUI → concept name
      - atc2cui : ATC code → UMLS CUI
      - cui2atc : UMLS CUI → ATC code
      - atc_info: ATC code → {name, class, indication, drugbank_id}
    """

    DRUG_RELATIONS = set(RELATION_WEIGHTS.keys())

    def __init__(
        self,
        umls_path:     str,
        names_path:    str,
        atc_path:      str,
        atc2umls_path: str,
    ):
        self.id2name:        dict[str, str]       = {}
        self.forward_adj:    dict[str, dict]      = defaultdict(dict)
        self.reverse_drug:   dict[str, list]      = defaultdict(list)
        self.drug_cuis:      set[str]             = set()
        self.atc_info:       dict[str, dict]      = {}
        self.atc2cui:        dict[str, str]        = {}
        self.cui2atc:        dict[str, str]        = {}

        self._load_names(names_path)
        self._load_atc(atc_path, atc2umls_path)
        self._load_umls(umls_path)
        log.info(
            "HypothesisKG loaded: %d nodes, %d drug CUIs, %d ATC entries",
            len(self.forward_adj), len(self.drug_cuis), len(self.atc_info),
        )

    def _load_names(self, path: str):
        log.info("Loading UMLS concept names from %s", path)
        with open(path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    self.id2name[parts[0]] = parts[1]

    def _load_atc(self, atc_path: str, atc2umls_path: str):
        log.info("Loading ATC drug catalogue from %s", atc_path)
        with open(atc_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("code", "").strip()
                if not code:
                    continue
                self.atc_info[code] = {
                    "name":       row.get("name", ""),
                    "class":      row.get("parent_code", ""),
                    "indication": row.get("indication", ""),
                    "drugbank_id":row.get("drugbank_id", ""),
                    "level":      row.get("level", ""),
                }

        log.info("Loading ATC→UMLS mapping from %s", atc2umls_path)
        with open(atc2umls_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                atc = row.get("ATC", "").strip()
                cui = row.get("UMLS", "").strip()
                if atc and cui:
                    self.atc2cui[atc]  = cui
                    self.cui2atc[cui]  = atc
                    self.drug_cuis.add(cui)
                    # Add ATC name to id2name for readability
                    if cui not in self.id2name and atc in self.atc_info:
                        self.id2name[cui] = self.atc_info[atc]["name"]

    def _load_umls(self, path: str):
        log.info("Loading UMLS triples (drug-relevant relations) from %s", path)
        loaded = 0
        with open(path, newline="", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                rel, head, tail = parts[0], parts[1], parts[2]
                if rel not in self.DRUG_RELATIONS:
                    continue
                self.forward_adj[head][tail] = rel
                # Build reverse index for drug nodes
                if rel in TREATMENT_RELATIONS and tail in self.drug_cuis:
                    self.reverse_drug[tail].append((head, rel))
                loaded += 1
        log.info("Loaded %d drug-relevant UMLS triples", loaded)

    def name(self, cui: str) -> str:
        return self.id2name.get(cui, cui)

    def is_drug(self, cui: str) -> bool:
        return cui in self.drug_cuis


# ── DRKG-backed KG (open-data alternative to UMLS) ────────────────────────────

# Maps DRKG relation-name substrings onto the engine's weighted scheme.
# DRKG relations look like "GNBR::T::Compound:Disease" or
# "DRUGBANK::treats::Compound:Disease" or "Hetionet::CtD::Compound:Disease"
# (CtD = "Compound treats Disease"). We match on lowercased substrings so
# the same weighting logic the UMLS path uses still applies.
_DRKG_REL_MAP = [
    ("treat",          "may_treat"),
    ("ctd",            "may_treat"),               # Hetionet Compound-treats-Disease
    ("palliate",       "may_treat"),
    ("cpd",            "has_active_ingredient"),
    ("active_ingredient", "has_active_ingredient"),
    ("risk",           "is_a_risk_factor_of"),
    ("cause",          "may_cause"),
    ("associate",      "may_cause"),
    ("interact",       "interacts_with"),
    ("contraindicat",  "may_contraindicate"),
    ("isa",            "isa"),
    ("is_a",           "isa"),
]


def _map_drkg_relation(drkg_rel: str) -> str | None:
    r = drkg_rel.lower()

    # GNBR uses single/short letter codes for its relation themes, which are
    # too short to substring-match safely, so handle them by exact theme code.
    # GNBR Compound-Disease themes: T=treatment, C=inhibits/causes,
    # Pa=prevents/alleviates, Pr=prevents, J=role in disease, Mp=biomarker.
    # See GNBR theme table (Percha & Altman 2018).
    if "gnbr::" in r:
        # relation looks like "gnbr::t::compound:disease"
        segs = drkg_rel.split("::")
        if len(segs) >= 2:
            theme = segs[1].strip().lower()
            gnbr_map = {
                "t":  "may_treat",
                "pa": "may_treat",
                "pr": "is_a_risk_factor_of",
                "c":  "may_cause",
                "j":  "may_cause",
                "mp": "may_cause",
            }
            if theme in gnbr_map:
                return gnbr_map[theme]

    for needle, canonical in _DRKG_REL_MAP:
        if needle in r:
            return canonical
    return None


class DRKGHypothesisKG:
    """
    Same interface and indexes as HypothesisKG, but built from DRKG's
    drkg.tsv instead of licensed UMLS files. Drugs are DRKG `Compound::`
    entities; the `umls_id`/CUI slot everywhere downstream now carries a
    DRKG entity ID (e.g. "Compound::DB00945"), which is fine because the
    engine treats those IDs as opaque.

    Builds:
      - id2name      : DRKG entity id → humanized name
      - forward_adj  : head → {tail: canonical_relation}
      - reverse_drug : drug_id → [(condition_id, relation), ...]
      - drug_cuis    : set of Compound:: entity ids
    """

    def __init__(self, drkg_root: str):
        import os
        from client.rag_cag.drkg_ids import humanize_drkg_id as _humanize_drkg_id

        self.id2name:      dict[str, str]  = {}
        self.forward_adj:  dict[str, dict] = defaultdict(dict)
        self.reverse_drug: dict[str, list] = defaultdict(list)
        self.drug_cuis:    set[str]        = set()
        self.atc_info:     dict[str, dict] = {}
        self.atc2cui:      dict[str, str]   = {}
        self.cui2atc:      dict[str, str]   = {}

        drkg_tsv = os.path.join(drkg_root, "drkg.tsv")
        if not os.path.exists(drkg_tsv):
            raise FileNotFoundError(
                f"drkg.tsv not found under {drkg_root} — extract drkg.tar.gz "
                f"there first (see DRKG_INTEGRATION_PLAN.md)."
            )

        log.info("Loading DRKG triples (drug-relevant relations) from %s", drkg_tsv)
        loaded = 0
        with open(drkg_tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                head, drkg_rel, tail = parts[0], parts[1], parts[2]
                rel = _map_drkg_relation(drkg_rel)
                if rel is None:
                    continue   # relation type not relevant to drug hypotheses

                self.forward_adj[head][tail] = rel
                for ent in (head, tail):
                    if ent not in self.id2name:
                        self.id2name[ent] = _humanize_drkg_id(ent)
                    if ent.startswith("Compound::"):
                        self.drug_cuis.add(ent)

                # reverse index: drug → condition via a treatment relation
                if rel in TREATMENT_RELATIONS:
                    if head.startswith("Compound::"):
                        self.reverse_drug[head].append((tail, rel))
                    elif tail.startswith("Compound::"):
                        self.reverse_drug[tail].append((head, rel))
                loaded += 1

        log.info("DRKGHypothesisKG loaded: %d drug-relevant triples, "
                 "%d nodes, %d drug compounds",
                 loaded, len(self.forward_adj), len(self.drug_cuis))

    def name(self, cui: str) -> str:
        return self.id2name.get(cui, cui)

    def is_drug(self, cui: str) -> bool:
        return cui in self.drug_cuis


# ── PathFinder ────────────────────────────────────────────────────────────────

class PathFinder:
    """
    BFS from each patient condition node toward drug nodes.
    Returns all paths of length <= MAX_HOPS that end at a drug CUI.
    """

    def __init__(self, kg: HypothesisKG, max_hops: int = MAX_HOPS):
        self.kg       = kg
        self.max_hops = max_hops

    def find_paths(
        self,
        source_cui:   str,
        visited_global: set[str],
    ) -> list[EvidencePath]:
        """
        BFS from source_cui.  Returns evidence paths to drug candidates.
        visited_global prevents revisiting nodes across multiple source calls.
        """
        paths: list[EvidencePath] = []

        # queue: (current_node, path_nodes_so_far, path_relations_so_far)
        queue: deque = deque()
        queue.append((source_cui, [source_cui], []))
        visited = {source_cui}

        while queue:
            node, path_nodes, path_rels = queue.popleft()

            if len(path_rels) >= self.max_hops:
                continue

            for neighbor, relation in self.kg.forward_adj.get(node, {}).items():
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                new_nodes = path_nodes + [neighbor]
                new_rels  = path_rels  + [relation]

                if self.kg.is_drug(neighbor):
                    paths.append(EvidencePath(
                        source_condition = source_cui,
                        target_drug      = neighbor,
                        path_nodes       = new_nodes,
                        path_relations   = new_rels,
                    ))
                    if len(paths) >= MAX_CANDIDATES * 3:
                        return paths
                else:
                    queue.append((neighbor, new_nodes, new_rels))

        return paths


# ── HypothesisScorer ──────────────────────────────────────────────────────────

class HypothesisScorer:
    """
    Scores and deduplicates drug candidates.
    Applies contraindication and interaction penalties.
    """

    def score(
        self,
        paths:             list[EvidencePath],
        current_drug_cuis: set[str],
        kg:                HypothesisKG,
    ) -> list[DrugHypothesis]:
        """
        Groups paths by target drug, computes composite score,
        applies safety penalties, returns ranked DrugHypothesis list.
        """
        # Group paths by drug
        drug_paths: dict[str, list[EvidencePath]] = defaultdict(list)
        for p in paths:
            drug_paths[p.target_drug].append(p)

        hypotheses: list[DrugHypothesis] = []

        for drug_cui, drug_evidence in drug_paths.items():
            # Score: sum of (relation_weight / hop_length) across paths
            raw_score = 0.0
            for path in drug_evidence:
                path_score = 0.0
                for rel in path.path_relations:
                    path_score += RELATION_WEIGHTS.get(rel, 0.1)
                # Penalise longer paths
                path_score /= (path.length ** 0.8)
                path.raw_score = path_score
                raw_score += path_score

            # Sort paths: best first
            drug_evidence.sort(key=lambda p: p.raw_score, reverse=True)

            # Safety checks against current medications
            contraindicated = False
            interactions:   list[str] = []

            for current_cui in current_drug_cuis:
                # Check if UMLS says drug contraindicated with current med
                rel_to_current = kg.forward_adj.get(drug_cui, {}).get(current_cui, "")
                if rel_to_current == "may_contraindicate":
                    contraindicated = True
                    raw_score += RELATION_WEIGHTS["may_contraindicate"]
                elif rel_to_current == "interacts_with":
                    interactions.append(kg.name(current_cui))
                    raw_score += RELATION_WEIGHTS["interacts_with"]

            # Normalise to [0, 1]
            final_score = max(0.0, min(1.0, raw_score / max(len(drug_evidence), 1)))

            # Resolve drug metadata from ATC catalogue
            atc_code = kg.cui2atc.get(drug_cui, "")
            atc_data = kg.atc_info.get(atc_code, {})

            hypotheses.append(DrugHypothesis(
                drug_cui        = drug_cui,
                drug_name       = kg.name(drug_cui) or atc_data.get("name", drug_cui),
                drug_class      = atc_data.get("class", ""),
                indication      = atc_data.get("indication", ""),
                score           = final_score,
                evidence_paths  = drug_evidence[:3],   # top 3 paths as evidence
                contraindicated = contraindicated,
                interactions    = interactions,
            ))

        hypotheses.sort(key=lambda h: h.score, reverse=True)
        return hypotheses


# ── HypothesisEngine (public API) ─────────────────────────────────────────────

class HypothesisEngine:
    """
    Main entry point for drug hypothesis generation.

    Parameters
    ----------
    umls_path     : path to KG_mapping/umls/umls.csv
    names_path    : path to KG_mapping/umls/concept_names.txt
    atc_path      : path to resources/ATC.csv
    atc2umls_path : path to KG_mapping/ATC_to_UMLS.csv
    max_hops      : BFS depth (default 3)
    """

    def __init__(
        self,
        umls_path:     str = "",
        names_path:    str = "",
        atc_path:      str = "",
        atc2umls_path: str = "",
        max_hops:      int = MAX_HOPS,
        kg_backend:    str = "umls",     # "umls" | "drkg"
        drkg_root:     str = "",
    ):
        if kg_backend == "drkg":
            if not drkg_root:
                raise ValueError("kg_backend='drkg' requires drkg_root")
            self.kg = DRKGHypothesisKG(drkg_root)
        else:
            self.kg = HypothesisKG(umls_path, names_path, atc_path, atc2umls_path)
        self.finder  = PathFinder(self.kg, max_hops=max_hops)
        self.scorer  = HypothesisScorer()

    def generate(
        self,
        patient_conditions: list[str],   # UMLS CUIs from patient KG
        patient_drugs:      list[str],   # current medication CUIs (for safety)
        top_k:              int = TOP_K_DEFAULT,
        exclude_current:    bool = True,
    ) -> list[DrugHypothesis]:
        """
        Generate drug hypotheses for a patient.

        Parameters
        ----------
        patient_conditions : UMLS CUIs of patient's active conditions
        patient_drugs      : UMLS CUIs of current medications
        top_k              : number of hypotheses to return
        exclude_current    : if True, exclude drugs already prescribed

        Returns
        -------
        List of DrugHypothesis, ranked by score descending.
        """
        current_drug_set = set(patient_drugs)
        all_paths: list[EvidencePath] = []
        visited_global: set[str] = set()

        for condition_cui in patient_conditions:
            if condition_cui not in self.kg.forward_adj:
                log.debug("Condition CUI not in KG: %s", condition_cui)
                continue
            log.info(
                "Traversing KG from condition: %s (%s)",
                condition_cui, self.kg.name(condition_cui)
            )
            paths = self.finder.find_paths(condition_cui, visited_global)
            all_paths.extend(paths)
            visited_global.update(condition_cui)

        if not all_paths:
            log.warning("No evidence paths found for conditions: %s",
                        patient_conditions)
            return []

        log.info("Total evidence paths found: %d", len(all_paths))

        hypotheses = self.scorer.score(all_paths, current_drug_set, self.kg)

        # Optionally remove drugs already prescribed
        if exclude_current:
            hypotheses = [h for h in hypotheses
                          if h.drug_cui not in current_drug_set]

        ranked = hypotheses[:top_k]
        log.info("Generated %d drug hypotheses (top_k=%d)", len(ranked), top_k)
        return ranked

    def explain_path(self, path: EvidencePath) -> str:
        """Human-readable explanation of a single evidence path."""
        return path.hop_summary(self.kg.id2name)

    def format_report(
        self,
        hypotheses:     list[DrugHypothesis],
        patient_id:     str = "ANONYMOUS",
        include_paths:  bool = True,
    ) -> str:
        """
        Formats hypothesis list as a structured clinical report string.
        Suitable for export or clinician review.
        No raw patient identifiers are included.
        """
        lines = [
            f"Drug Hypothesis Report — Patient: {patient_id}",
            "=" * 60,
            f"Total candidates evaluated: see evidence below",
            "",
        ]
        for rank, h in enumerate(hypotheses, 1):
            safety = " [CONTRAINDICATED]" if h.contraindicated else ""
            lines.append(f"Rank {rank}: {h.drug_name}{safety}")
            lines.append(f"  Drug class : {h.drug_class or 'N/A'}")
            lines.append(f"  Indication : {h.indication or 'N/A'}")
            lines.append(f"  Score      : {h.score:.4f}")
            if h.interactions:
                lines.append(f"  Interactions with current meds: "
                              f"{', '.join(h.interactions)}")
            if include_paths and h.evidence_paths:
                lines.append("  Evidence paths:")
                for ep in h.evidence_paths[:2]:
                    lines.append(f"    [{ep.length} hop(s)] "
                                 f"{self.explain_path(ep)}")
            lines.append(f"  Narrative  : {h.narrative}")
            lines.append("")
        return "\n".join(lines)
