"""The audit report produced by the deterministic orchestrator."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_serializer

from react_review.core.enums import ReportVerdict
from react_review.schemas.audit import MatchResult
from react_review.steps.data_extraction.schemas import DocumentScope


class UnmatchedClaim(BaseModel):
    """A claim or a source value that was NOT paired, and why.

    Was a ``"study/group/timepoint/field"`` string, which could not carry a
    reason — so "no source evidence for this" and "I refused to guess which of
    two identical keys this belongs to" looked the same to a reader. They are
    very different findings, so the reason travels with the item.
    """

    audit_id: str = ""
    study_id: str = ""
    group: str = "-"
    timepoint: str = "single"
    field_type: str = ""
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    checklist_id: str = ""
    # no_source_evidence | ambiguous_match_key | unclaimed_source
    reason_code: str = "no_source_evidence"
    message: str = ""

    @model_serializer(mode="wrap")
    def _omit_legacy_empty_audit_id(self, handler):
        body = handler(self)
        if not body.get("audit_id"):
            body.pop("audit_id", None)
        return body

    @property
    def key_text(self) -> str:
        base = f"{self.study_id}/{self.group}/{self.timepoint}/{self.field_type}"
        return f"{base} [checklist:{self.checklist_id}]" if self.checklist_id else base


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

    audit_id: str = ""
    study_id: str = ""
    group: str = "-"
    # Carried so a flag points at ONE cell: without the timepoint and the cell
    # it came from, two rows of the same study/cohort/field are indistinguishable
    # to whoever has to check them.
    timepoint: str = "single"
    field_type: str = ""
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    checklist_id: str = ""
    # Concept-level flags point back to the one run-level field decision and say
    # how many cells it affected. Ordinary audit discrepancies leave the key
    # empty and continue to point at exactly one cell.
    resolution_key: str = ""
    affected_cells: int = 1
    label: str = ""          # the audit label, or "escalated"/"unmatched"
    reason: str = ""
    # Populated for evidence-gate refusals so a reviewer sees both the decision
    # and the extent of source material it was based on.
    document_scope: DocumentScope = DocumentScope.UNKNOWN
    evidence_adequacy_status: str = ""
    evidence_adequacy_reason_codes: list[str] = Field(default_factory=list)
    review_required: bool = False

    @model_serializer(mode="wrap")
    def _omit_legacy_empty_audit_id(self, handler):
        body = handler(self)
        if not body.get("audit_id"):
            body.pop("audit_id", None)
        if self.document_scope is DocumentScope.UNKNOWN:
            body.pop("document_scope", None)
        if not body.get("evidence_adequacy_status"):
            body.pop("evidence_adequacy_status", None)
        if not body.get("evidence_adequacy_reason_codes"):
            body.pop("evidence_adequacy_reason_codes", None)
        if not body.get("review_required"):
            body.pop("review_required", None)
        return body


class FinalVerification(BaseModel):
    """The Judge's adjudicated outcome over one audit run."""

    run_id: str = ""
    verdict: ReportVerdict = ReportVerdict.INCOMPLETE
    human_review_flags: list[HumanReviewFlag] = Field(default_factory=list)
    summary: str = ""
