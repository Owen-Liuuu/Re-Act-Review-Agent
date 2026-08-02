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
from react_review.schemas.reason import ReasonRecord

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
    # DKB resolution: "resolved" (authoritative) | "candidate" (provisional, tentative)
    # | "unresolved" (field_type unknown — kept, but not comparable / needs review).
    resolution_status: str = "resolved"
    # --- provenance back to the captured table (all optional: CSV-loaded items
    # and hand-built test items keep working unchanged) ---
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None    # (row, column) in the captured table
    column_header: str = ""                    # the header path, verbatim
    cohort_label: str = ""                     # the review's OWN word for the cohort
    timepoint_label: str = ""                  # the review's OWN word, "" if none
    origin: str = "review_table"               # review_table | checklist
    reasons: list[ReasonRecord] = Field(default_factory=list)


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
    # Back-check: source evidence (unit/value) contradicts a CANDIDATE translation
    # → the auto-classified field_type is likely wrong. Set only for candidates.
    concept_mismatch: bool = False
    concept_mismatch_reason: str = ""


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
