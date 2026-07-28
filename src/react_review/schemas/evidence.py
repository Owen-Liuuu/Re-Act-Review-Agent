"""Review-side and source-side evidence rows (the two extraction ground-truths).

Field names mirror the benchmark CSV columns so the eval harness can load them
directly. ``group`` uses the canonical vocabulary values (``t1dm`` / ``control``
/ ``all`` / ``-`` for study-level rows); ``field_type`` is the canonical concept
key. ``value`` / ``source_value`` are kept as the verbatim strings (e.g.
``"6.60 ± 0.71"``) — the syntax normaliser extracts the primary number at
compare time, so the raw spread is preserved for the report.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from react_review.core.enums import CollectionOutcome

Value = str | int | float | None


class ReviewDataItem(BaseModel):
    """One value the review reports (a cell of its data-extraction table).

    Mirrors ``review_ground_truth.csv``.
    """

    review_data_id: str = ""
    study_id: str
    group: str = "-"
    timepoint: str = "single"
    field_type: str
    raw_field_name: str = ""
    value: Value = None
    unit: str = ""
    source_location: str = ""


class SourceEvidenceItem(BaseModel):
    """One value located in a source paper, keyed to the review claim it answers.

    Mirrors the source side of ``audit_template.csv``.
    """

    study_id: str
    group: str = "-"
    timepoint: str = "single"
    field_type: str
    source_value: Value = None
    source_unit: str = ""
    source_quote: str = ""
    source_location_in_paper: str = ""
    collection_outcome: CollectionOutcome = CollectionOutcome.FOUND


class IncludedStudy(BaseModel):
    """A cited source paper. Mirrors ``included_studies.csv``."""

    study_id: str
    review_citation: str = ""
    country: str = ""
    reported_N: int | None = None
    measurement_tool: str = ""
    modality: str = ""
    overall_quality: str = ""
    doi: str = ""
    review_ref_number: str = ""
    source_pdf: str = ""
