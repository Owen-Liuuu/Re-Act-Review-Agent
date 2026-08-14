"""Load review / source evidence tables from CSV for the deterministic audit.

Column-name tolerant (accepts a dedicated review.csv / source.csv, or the
benchmark's combined audit_template.csv pointed at from both sides) and always
reads as ``utf-8-sig`` (Excel may add a BOM). See docs/normalization_pipeline.md
for the long-table shape.
"""
from __future__ import annotations

import csv
from pathlib import Path

from react_review.schemas.evidence import (
    IncludedStudy,
    ReviewDataItem,
    SourceEvidenceItem,
)


def _rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _first(row: dict[str, str], *names: str, default: str = "") -> str:
    """Return the first present, non-empty column among ``names``."""
    for n in names:
        v = row.get(n)
        if v is not None and str(v).strip():
            return v
    return default


def load_review_items(path: Path | str) -> list[ReviewDataItem]:
    """Load review-side rows. Accepts ``value`` or ``review_value`` for the value."""
    items: list[ReviewDataItem] = []
    for r in _rows(path):
        study_id = _first(r, "study_id")
        field_type = _first(r, "field_type")
        if not study_id or not field_type:
            continue  # skip rows that can't be matched/audited
        items.append(ReviewDataItem(
            review_data_id=_first(r, "review_data_id", "audit_id"),
            study_id=study_id,
            group=_first(r, "group", default="-"),
            timepoint=_first(r, "timepoint", default="single"),
            field_type=field_type,
            raw_field_name=_first(r, "raw_field_name"),
            value=_first(r, "value", "review_value") or None,
            unit=_first(r, "unit", "review_unit"),
        ))
    return items


def load_included_studies(path: Path | str) -> list[IncludedStudy]:
    """Load the included-studies registry (study_id, doi, source_pdf, …)."""
    studies: list[IncludedStudy] = []
    for r in _rows(path):
        sid = _first(r, "study_id")
        if not sid:
            continue
        n = _first(r, "reported_N")
        studies.append(IncludedStudy(
            study_id=sid,
            review_citation=_first(r, "review_citation"),
            country=_first(r, "country"),
            reported_N=int(n) if n.isdigit() else None,
            measurement_tool=_first(r, "measurement_tool"),
            modality=_first(r, "modality"),
            overall_quality=_first(r, "overall_quality"),
            doi=_first(r, "doi"),
            review_ref_number=_first(r, "review_ref_number"),
            source_pdf=_first(r, "source_pdf"),
        ))
    return studies


def load_source_items(path: Path | str) -> list[SourceEvidenceItem]:
    """Load source-side rows. Accepts ``source_value`` or ``value`` for the value."""
    items: list[SourceEvidenceItem] = []
    for r in _rows(path):
        study_id = _first(r, "study_id")
        field_type = _first(r, "field_type")
        if not study_id or not field_type:
            continue
        items.append(SourceEvidenceItem(
            review_data_id=_first(r, "review_data_id", "audit_id"),
            study_id=study_id,
            group=_first(r, "group", default="-"),
            timepoint=_first(r, "timepoint", default="single"),
            field_type=field_type,
            source_value=_first(r, "source_value", "value") or None,
            source_unit=_first(r, "source_unit", "unit"),
            source_quote=_first(r, "source_quote"),
            source_location_in_paper=_first(r, "source_location_in_paper"),
        ))
    return items
