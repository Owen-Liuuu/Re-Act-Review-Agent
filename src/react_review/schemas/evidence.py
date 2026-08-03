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
    # How that label was placed: resolved | alias | combined | ambiguous |
    # unknown | not_applicable (a study-level field has no cohort dimension).
    # "unknown"/"ambiguous" must reach a human — never be treated as a cohort.
    cohort_status: str = "resolved"
    timepoint_label: str = ""                  # the review's OWN word, "" if none
    origin: str = "review_table"               # review_table | checklist
    # Stable identity for a checklist-authored concrete claim. Empty for normal
    # table cells and for presence/gap checks, which never enter value matching.
    checklist_id: str = ""
    # Joins this row to the run-level FieldResolutionRecord that explains how
    # ``raw_field_name`` became ``field_type``.  Empty for placeholders and
    # hand-built/CSV fixtures that never went through the Resolver.
    resolution_key: str = ""
    reasons: list[ReasonRecord] = Field(default_factory=list)


class SourceEvidenceItem(BaseModel):
    """One value located in a source paper, keyed to the review claim it answers.

    Mirrors the source side of ``audit_template.csv``.
    """

    study_id: str
    group: str = "-"
    timepoint: str = "single"
    field_type: str
    # Which review cell this evidence answers. Carried so two claims that share
    # a study/cohort/field can still be told apart when they are paired.
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    checklist_id: str = ""
    source_value: Value = None
    source_unit: str = ""
    source_quote: str = ""
    source_location_in_paper: str = ""
    # WHERE this was read from. ``source_location_in_paper`` says "Table 2" — of
    # WHICH document was never recorded, so a reader could not go and check.
    # A local run has a file; an online one has a URL; both always have a kind.
    source_file: str = ""          # absolute path, when read from disk
    source_uri: str = ""           # URL / PMC id / DOI — the online equivalent
    source_paper_id: str = ""
    source_doi: str = ""
    retriever_kind: str = ""       # local_pdf | pmc | unpaywall | openalex_pdf | …
    collection_outcome: CollectionOutcome = CollectionOutcome.FOUND
    # Back-check: source evidence (unit/value) contradicts a CANDIDATE translation
    # → the auto-classified field_type is likely wrong. Set only for candidates.
    concept_mismatch: bool = False
    concept_mismatch_reason: str = ""
    # ok | wrong_cohort | ambiguous — whether the paper's own cohort label could
    # be confirmed against the one asked for. "ambiguous" must reach a human.
    cohort_check: str = "ok"
    cohorts_seen: list[str] = Field(default_factory=list)
    reasons: list[ReasonRecord] = Field(default_factory=list)


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
