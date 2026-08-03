"""The audit report produced by the deterministic orchestrator."""
from __future__ import annotations

from pydantic import BaseModel, Field

from react_review.core.enums import ReportVerdict
from react_review.schemas.audit import MatchResult


class UnmatchedClaim(BaseModel):
    """A claim or a source value that was NOT paired, and why.

    Was a ``"study/group/timepoint/field"`` string, which could not carry a
    reason — so "no source evidence for this" and "I refused to guess which of
    two identical keys this belongs to" looked the same to a reader. They are
    very different findings, so the reason travels with the item.
    """

    study_id: str = ""
    group: str = "-"
    timepoint: str = "single"
    field_type: str = ""
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    # no_source_evidence | ambiguous_match_key | unclaimed_source
    reason_code: str = "no_source_evidence"
    message: str = ""

    @property
    def key_text(self) -> str:
        return f"{self.study_id}/{self.group}/{self.timepoint}/{self.field_type}"


class AuditReport(BaseModel):
    """Aggregated outcome of an audit run.

    Attributes:
        results: per-pair comparison outcomes.
        n_match / n_mismatch / n_unit_mismatch / n_not_comparable: label counts.
        unmatched_review: review claims with no source evidence, each with a reason.
        unmatched_source: source values no review claim took, each with a reason.
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
    unmatched_review: list[UnmatchedClaim] = Field(default_factory=list)
    unmatched_source: list[UnmatchedClaim] = Field(default_factory=list)
    verdict: ReportVerdict = ReportVerdict.INCOMPLETE
    flags: list[str] = Field(default_factory=list)
    summary: str = ""


class HumanReviewFlag(BaseModel):
    """One item the Judge routes to a human (architecture: Human Review Flag)."""

    study_id: str = ""
    group: str = "-"
    # Carried so a flag points at ONE cell: without the timepoint and the cell
    # it came from, two rows of the same study/cohort/field are indistinguishable
    # to whoever has to check them.
    timepoint: str = "single"
    field_type: str = ""
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    label: str = ""          # the audit label, or "escalated"/"unmatched"
    reason: str = ""


class FinalVerification(BaseModel):
    """The Judge's adjudicated outcome over one audit run."""

    run_id: str = ""
    verdict: ReportVerdict = ReportVerdict.INCOMPLETE
    human_review_flags: list[HumanReviewFlag] = Field(default_factory=list)
    summary: str = ""
