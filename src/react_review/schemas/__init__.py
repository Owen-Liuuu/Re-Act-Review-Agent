"""Typed data contract for ReAct-Review's 4-table audit model.

Tables (see docs/normalization_pipeline.md and eval/benchmark_1/):
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
from react_review.schemas.report import (
    AuditReport,
    FinalVerification,
    HumanReviewFlag,
)
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.package import EvidencePackage
from react_review.schemas.resolution import (
    FieldResolutionRecord,
    ResolutionAttempt,
    ResolutionCellRef,
)
from react_review.schemas.knowledge import KnowledgeConflictRecord, KnowledgeImportRecord

__all__ = [
    "ReviewDataItem",
    "SourceEvidenceItem",
    "IncludedStudy",
    "MatchResult",
    "ToleranceRule",
    "AuditReport",
    "FinalVerification",
    "HumanReviewFlag",
    "AgentRun",
    "StepRecord",
    "EvidencePackage",
    "FieldResolutionRecord",
    "ResolutionAttempt",
    "ResolutionCellRef",
    "KnowledgeConflictRecord",
    "KnowledgeImportRecord",
]
