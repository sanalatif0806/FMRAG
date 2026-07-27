"""
client/rag_cag/drkg_ids.py
---------------------------
Tiny, dependency-free helpers for DRKG entity IDs.

Kept separate from drkg_loader.py (which imports torch) so that code paths
needing only the string logic — e.g. DRKGHypothesisKG building its indexes
from drkg.tsv — can use it without pulling in torch/transformers.

DRKG entity IDs look like:
    Compound::DB00945       (drug, DrugBank id)
    Disease::MESH:D001351   (disease)
    Gene::2931              (gene, NCBI id)
"""

from __future__ import annotations


def humanize_drkg_id(entity_id: str) -> str:
    """
    Turn a DRKG entity ID into a human-readable token for text embedding.

    'Compound::DB00945'       -> 'DB00945'
    'Disease::MESH:D001351'   -> 'MESH D001351'
    'Gene::2931'              -> 'Gene 2931'
    'Anatomy::UBERON:0000178' -> 'UBERON 0000178'

    DRKG entity IDs are codes, not English names — DRKG ships no name
    dictionary. So the "name" we expose is the cleaned ID. For true
    English-name matching, join an external table (DrugBank names, MeSH
    descriptors); see DRKG_INTEGRATION_PLAN.md item 2.
    """
    body = entity_id.split("::", 1)[-1]
    return body.replace(":", " ").replace("_", " ").strip()


def entity_type(entity_id: str) -> str:
    """'Compound::DB00945' -> 'Compound'. Returns '' if no type prefix."""
    return entity_id.split("::", 1)[0] if "::" in entity_id else ""
