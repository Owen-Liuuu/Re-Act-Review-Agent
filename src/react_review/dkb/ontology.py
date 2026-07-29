"""DKB ontology import — the AUTHORITATIVE foundation layer (DKB-3).

Merges a JSON slice of an external ontology (LOINC / UCUM / MeSH exported to the
KnowledgeEntry shape) into the KB as ``status=authoritative`` with
``provenance.source=ontology:<name>``. Merging an existing concept UNIONs its
synonyms and promotes it if it was provisional. Plugging in a real ontology is a
data-acquisition step (download / licence); a tiny curated example ships in
``configs/ontology/`` so the mechanism is exercised end-to-end.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

from react_review.dkb.base import KnowledgeBase
from react_review.dkb.schema import KnowledgeEntry, Provenance

logger = structlog.get_logger(__name__)


def import_ontology(kb: KnowledgeBase, path: Path | str, *, source: str) -> tuple[int, int]:
    """Merge authoritative concepts from an ontology JSON. Returns (added, merged)."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    added = merged = 0
    for ft, body in data.items():
        existing = kb.entries.get(ft)
        if existing is None:
            entry = KnowledgeEntry(
                field_type=ft,
                **{k: v for k, v in body.items() if k != "field_type"},
            )
            entry.provenance = Provenance(source=f"ontology:{source}")
            entry.status = "authoritative"
            kb.add(entry)
            added += 1
        else:
            # union synonyms and lift to authoritative (ontology is a trusted source)
            existing.synonyms = list(dict.fromkeys(
                [*existing.synonyms, *body.get("synonyms", [])]))
            existing.status = "authoritative"
            merged += 1
    logger.info("dkb_ontology_import", source=source, added=added, merged=merged)
    return added, merged
