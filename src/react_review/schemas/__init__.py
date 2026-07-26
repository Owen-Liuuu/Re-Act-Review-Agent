"""Typed data contract for ReAct-Review's 4-table audit model.

Tables (see docs/normalization_pipeline.md and eval/benchmark/):
  - ReviewDataItem     — one value the review reports (review_ground_truth.csv)
  - SourceEvidenceItem — one value found in a source paper (audit source side)
  - IncludedStudy      — a cited source paper (included_studies.csv)
  - MatchResult        — the audit outcome for one (study, group, field_type)
  - ToleranceRule      — the comparison tolerance for a field_type
"""
from react_review.schemas.evidence import (
    IncludedStudy,
    ReviewDataItem,
    SourceEvidenceItem,
)
from react_review.schemas.audit import MatchResult, ToleranceRule

__all__ = [
    "ReviewDataItem",
    "SourceEvidenceItem",
    "IncludedStudy",
    "MatchResult",
    "ToleranceRule",
]
