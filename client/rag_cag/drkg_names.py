"""
client/rag_cag/drkg_names.py
-----------------------------
Resolves DRKG entity IDs to real English names by joining against public
biomedical vocabularies. This is the missing "name-join" that makes DRKG
text-retrieval meaningful: without it, drkg_loader embeds the raw ID string
("2157") and clinical names like "Metformin" match nothing.

DRKG entity-type prefixes (from entities.tsv) and where their names come from:
    Gene::<ncbi_id>          -> NCBI Gene   (gene_info: symbol + description)
    Compound::<drugbank_id>  -> DrugBank    (drug name)
    Compound::<mesh_id>      -> MeSH        (some compounds are MeSH IDs)
    Disease::MESH:<id>       -> MeSH        (descriptor name)
    Disease::DOID:<id>       -> Disease Ontology (optional)
    Anatomy::UBERON:<id>     -> UBERON      (optional)
    Side Effect::<umls>      -> (left as cleaned id; UMLS-licensed)
    Pharmacologic Class, Symptom, Pathway, Biological Process, ... ->
        cleaned id fallback (these have few/no free name tables)

Design:
  - Each source parser returns a dict {drkg_id: english_name}.
  - resolve_names() merges them; anything unmatched falls back to the
    humanized id (so behavior degrades gracefully, never crashes).
  - The result is a {entity_id: name} dict that drkg_loader uses INSTEAD of
    humanize_drkg_id, so the BioBERT index is built over real names.

All three vocabularies are FREE (NCBI Gene and MeSH are fully open; DrugBank's
open vocabulary CSV needs a no-cost account). None require UMLS.

Download sources (put paths in config or pass to build_name_map):
  NCBI Gene:  https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz
  MeSH:       https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/d2024.bin
  DrugBank:   https://go.drugbank.com/releases/latest#open-data  (drugbank vocabulary.csv)

Usage (one-time, builds a cache):
    from client.rag_cag.drkg_names import build_name_map
    names = build_name_map(
        entities_tsv = "/data/drkg/embed/entities.tsv",
        ncbi_gene_info = "/data/drkg/names/gene_info.gz",
        drugbank_vocab = "/data/drkg/names/drugbank_vocabulary.csv",
        mesh_desc      = "/data/drkg/names/d2024.bin",
        out_cache      = "/data/drkg/entity_names.pkl",
    )
    # names: {"Gene::2157": "F13A1 coagulation factor XIII A chain", ...}
"""
from __future__ import annotations

import csv
import gzip
import logging
import os
import pickle
import re

log = logging.getLogger(__name__)


# ── entity id parsing ─────────────────────────────────────────────────────────

def split_entity(entity_id: str) -> tuple[str, str]:
    """'Gene::2157' -> ('Gene', '2157');  'Disease::MESH:D00' -> ('Disease','MESH:D00')."""
    if "::" in entity_id:
        etype, body = entity_id.split("::", 1)
        return etype, body
    return "", entity_id


def _mesh_id(body: str) -> str | None:
    """Extract a MeSH descriptor id like 'D001351' from 'MESH:D001351'."""
    m = re.search(r"MESH:?(?P<id>[CD]\d+)", body, re.IGNORECASE)
    return m.group("id") if m else None


def _drugbank_id(body: str) -> str | None:
    """Extract a DrugBank id like 'DB00945'."""
    m = re.search(r"(DB\d{5})", body)
    return m.group(1) if m else None


def _ncbi_gene_id(etype: str, body: str) -> str | None:
    """DRKG Gene bodies are NCBI gene IDs (may be 'id' or 'id;id2')."""
    if etype != "Gene":
        return None
    first = body.split(";", 1)[0].strip()
    return first if first.isdigit() else None


# ── source parsers ────────────────────────────────────────────────────────────

def parse_ncbi_gene(gene_info_path: str) -> dict[str, str]:
    """
    NCBI gene_info(.gz): tab-separated, columns include
      tax_id  GeneID  Symbol ... description ...
    Returns {ncbi_gene_id: "SYMBOL description"}.
    Only human (tax_id 9606) is kept to limit size and match clinical use.
    """
    names: dict[str, str] = {}
    if not gene_info_path or not os.path.exists(gene_info_path):
        log.warning("NCBI gene_info not found at %s — genes will use raw ids",
                    gene_info_path)
        return names
    opener = gzip.open if gene_info_path.endswith(".gz") else open
    with opener(gene_info_path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        try:
            i_tax = header.index("tax_id")
            i_gid = header.index("GeneID")
            i_sym = header.index("Symbol")
            i_desc = header.index("description")
        except ValueError:
            # fall back to known column positions
            i_tax, i_gid, i_sym, i_desc = 0, 1, 2, 8
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= i_desc:
                continue
            if p[i_tax] != "9606":         # human only
                continue
            gid, sym, desc = p[i_gid], p[i_sym], p[i_desc]
            name = f"{sym} {desc}".strip() if desc and desc != "-" else sym
            names[gid] = name
    log.info("NCBI: parsed %d human gene names", len(names))
    return names


def parse_drugbank_vocab(vocab_csv_path: str) -> dict[str, str]:
    """
    DrugBank open 'vocabulary.csv': columns include
      'DrugBank ID', 'Common name', 'Synonyms', ...
    Returns {drugbank_id: common_name}.
    """
    names: dict[str, str] = {}
    if not vocab_csv_path or not os.path.exists(vocab_csv_path):
        log.warning("DrugBank vocabulary not found at %s — drugs will use raw ids",
                    vocab_csv_path)
        return names
    with open(vocab_csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        # tolerate column-name variations
        id_col = next((c for c in reader.fieldnames or []
                       if c.strip().lower() in ("drugbank id", "drugbank_id")), None)
        name_col = next((c for c in reader.fieldnames or []
                         if c.strip().lower() in ("common name", "common_name", "name")), None)
        if not id_col or not name_col:
            log.warning("DrugBank vocab columns unrecognized: %s", reader.fieldnames)
            return names
        for row in reader:
            db_id = (row.get(id_col) or "").strip()
            name = (row.get(name_col) or "").strip()
            if db_id and name:
                names[db_id] = name
    log.info("DrugBank: parsed %d drug names", len(names))
    return names


def parse_mesh_descriptors(mesh_bin_path: str) -> dict[str, str]:
    """
    MeSH ASCII descriptor file (d20XX.bin): records separated by '*NEWRECORD',
    with 'MH = <name>' (main heading) and 'UI = <Dxxxxxxx>' (unique id).
    Returns {mesh_ui: main_heading}.
    """
    names: dict[str, str] = {}
    if not mesh_bin_path or not os.path.exists(mesh_bin_path):
        log.warning("MeSH descriptors not found at %s — diseases will use raw ids",
                    mesh_bin_path)
        return names
    mh, ui = None, None
    with open(mesh_bin_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "*NEWRECORD":
                mh, ui = None, None
            elif line.startswith("MH = "):
                mh = line[5:].strip()
            elif line.startswith("UI = "):
                ui = line[5:].strip()
                if ui and mh:
                    names[ui] = mh
    log.info("MeSH: parsed %d descriptor names", len(names))
    return names


# ── the join ──────────────────────────────────────────────────────────────────

def _humanize_fallback(entity_id: str) -> str:
    """Same cleaned-id fallback drkg_ids.humanize_drkg_id produces."""
    body = entity_id.split("::", 1)[-1]
    return body.replace(":", " ").replace("_", " ").strip()


def build_name_map(
    entities_tsv: str,
    ncbi_gene_info: str | None = None,
    drugbank_vocab: str | None = None,
    mesh_desc: str | None = None,
    out_cache: str | None = None,
) -> dict[str, str]:
    """
    Build {drkg_entity_id: english_name} by joining DRKG entity ids against
    whichever name tables are provided. Missing tables -> those entity types
    fall back to the cleaned id (graceful degradation, never crashes).
    """
    gene_names = parse_ncbi_gene(ncbi_gene_info) if ncbi_gene_info else {}
    drug_names = parse_drugbank_vocab(drugbank_vocab) if drugbank_vocab else {}
    mesh_names = parse_mesh_descriptors(mesh_desc) if mesh_desc else {}

    # load DRKG entity ids
    entity_ids: list[str] = []
    with open(entities_tsv, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                entity_ids.append(parts[0])

    resolved: dict[str, str] = {}
    stats = {"gene": 0, "drug": 0, "mesh": 0, "fallback": 0}

    for eid in entity_ids:
        etype, body = split_entity(eid)
        name = None

        gid = _ncbi_gene_id(etype, body)
        if gid and gid in gene_names:
            name = gene_names[gid]; stats["gene"] += 1

        if name is None:
            dbid = _drugbank_id(body)
            if dbid and dbid in drug_names:
                name = drug_names[dbid]; stats["drug"] += 1

        if name is None:
            mid = _mesh_id(body)
            if mid and mid in mesh_names:
                name = mesh_names[mid]; stats["mesh"] += 1

        if name is None:
            name = _humanize_fallback(eid); stats["fallback"] += 1

        resolved[eid] = name

    total = len(entity_ids)
    matched = total - stats["fallback"]
    log.info("DRKG name-join: %d/%d entities matched to real names "
             "(gene=%d drug=%d mesh=%d), %d fell back to cleaned id",
             matched, total, stats["gene"], stats["drug"], stats["mesh"],
             stats["fallback"])

    if out_cache:
        os.makedirs(os.path.dirname(out_cache) or ".", exist_ok=True)
        with open(out_cache, "wb") as f:
            pickle.dump(resolved, f)
        log.info("DRKG name map cached to %s", out_cache)

    return resolved


def load_name_map(cache_path: str) -> dict[str, str] | None:
    """Load a previously-built name map, or None if absent."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None
