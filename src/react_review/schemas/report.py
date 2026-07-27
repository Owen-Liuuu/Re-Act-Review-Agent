"""The audit report produced by the deterministic orchestrator."""
from __future__ import annotations

from pydantic import BaseModel, Field

from react_review.core.enums import ReportVerdict
from react_review.schemas.audit import MatchResult


class AuditReport(BaseModel):
    """Aggregated outcome of an audit run.

    Attributes:
        results: per-pair comparison outcomes.
        n_match / n_mismatch / n_unit_mismatch / n_not_comparable: label counts.
        unmatched_review: keys of review claims with no source evidence.
        unmatched_source: keys of source values with no matching review claim.
        verdict: PASS / PARTIAL / FAIL / INCOMPLETE.
        flags: human-readable notes (e.g. unmatched items).
        summary: one-line human summary.
    """

    run_id: str = ""
    results: list[MatchResult] = Field(default_factory=list)
    n_match: int = 0
    n_mismatch: int = 0
    n_unit_mismatch: int = 0
    n_not_comparable: int = 0
    unmatched_review: list[str] = Field(default_factory=list)
    unmatched_source: list[str] = Field(default_factory=list)
    verdict: ReportVerdict = ReportVerdict.INCOMPLETE
    flags: list[str] = Field(default_factory=list)
    summary: str = ""


class HumanReviewFlag(BaseModel):
    """One item the Judge routes to a human (architecture: Human Review Flag)."""

    study_id: str = ""
    group: str = "-"
    field_type: str = ""
    label: str = ""          # the audit label, or "escalated"/"unmatched"
    reason: str = ""


class FinalVerification(BaseModel):
    """The Judge's adjudicated outcome over one audit run."""

    run_id: str = ""
    verdict: ReportVerdict = ReportVerdict.INCOMPLETE
    human_review_flags: list[HumanReviewFlag] = Field(default_factory=list)
    summary: str = ""
