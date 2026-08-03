"""DKB ontology import — the AUTHORITATIVE foundation layer (DKB-3).

Merges a JSON slice of an external ontology (LOINC / UCUM / MeSH exported to the
KnowledgeEntry shape) into the KB as ``status=authoritative`` with
``provenance.source=ontology:<name>``. Merging an existing concept UNIONs its
synonyms and promotes it if it was provisional. Plugging in a real ontology is a
data-acquisition step (download / licence); a tiny curated example ships in
``configs/ontology/`` so the mechanism is exercised end-to-end.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import structlog

from react_review.dkb.base import KnowledgeBase
from react_review.dkb.schema import KnowledgeEntry, Provenance
from react_review.schemas.knowledge import KnowledgeConflictRecord, KnowledgeImportRecord

logger = structlog.get_logger(__name__)

_OVERRIDABLE_FIELDS = (
    "concept", "value_type", "default_unit", "domain", "scope",
    "unit_equivalences", "plausible_range", "disambiguation",
)


def import_ontology(kb: KnowledgeBase, path: Path | str, *, source: str) -> tuple[int, int]:
    """Merge curated concepts and record every override.

    Policy: values explicitly present in the curated ontology override the seed;
    omitted ontology fields retain their seed value; synonyms are unioned. The
    historical ``(added, merged)`` return shape is preserved for callers.
    """
    ontology_path = Path(path).resolve()
    raw = ontology_path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    before = len(kb.entries)
    added = merged = 0
    added_field_types: list[str] = []
    merged_field_types: list[str] = []
    conflicts: list[KnowledgeConflictRecord] = []
    for ft, body in data.items():
        entry_body = {k: v for k, v in body.items() if k != "field_type"}
        incoming = KnowledgeEntry(field_type=ft, **entry_body)
        existing = kb.entries.get(ft)
        if existing is None:
            entry = incoming
            entry.provenance = Provenance(source=f"ontology:{source}")
            entry.status = "authoritative"
            kb.add(entry)
            added += 1
            added_field_types.append(ft)
        else:
            # Explicit curated values win. Omitted values do not erase useful
            # seed metadata, and synonyms are additive rather than conflicting.
            for name in _OVERRIDABLE_FIELDS:
                if name not in body:
                    continue
                old = getattr(existing, name)
                new = getattr(incoming, name)
                if old != new:
                    conflicts.append(KnowledgeConflictRecord(
                        field_type=ft, field=name,
                        seed_value=old, ontology_value=new))
                    setattr(existing, name, new)
            existing.synonyms = list(dict.fromkeys(
                [*existing.synonyms, *body.get("synonyms", [])]))
            existing.provenance = Provenance(source=f"ontology:{source}")
            existing.status = "authoritative"
            merged += 1
            merged_field_types.append(ft)

    record = KnowledgeImportRecord(
        source=f"ontology:{source}", path=str(ontology_path),
        sha256=hashlib.sha256(raw).hexdigest(),
        concepts_before=before, concepts_after=len(kb.entries),
        added=added, merged=merged,
        added_field_types=added_field_types,
        merged_field_types=merged_field_types,
        conflicts=conflicts,
    )
    kb.imports.append(record)
    kb.version = kb.fingerprint()
    logger.info("dkb_ontology_import", source=source, added=added, merged=merged,
                conflicts=len(conflicts), fingerprint=kb.version[:12])
    return added, merged


def load_runtime_knowledge(
    seed_path: Path | str, ontology_dir: Path | str | None = None,
) -> KnowledgeBase:
    """Load the seed plus every ``*.json`` ontology slice in stable order."""
    kb = KnowledgeBase.from_json(seed_path)
    directory = Path(ontology_dir) if ontology_dir is not None else None
    if directory is not None and directory.is_dir():
        for path in sorted(directory.glob("*.json"), key=lambda p: p.name.lower()):
            import_ontology(kb, path, source=path.stem)
    kb.version = kb.fingerprint()
    return kb
