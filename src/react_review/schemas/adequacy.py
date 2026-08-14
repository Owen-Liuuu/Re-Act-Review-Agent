"""Typed, serialisable claim-level evidence-adequacy decisions."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from react_review.steps.data_extraction.schemas import DocumentScope


class AdequacyStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class AxisStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_REQUIRED = "not_required"


class EvidenceAnchor(BaseModel):
    """One checkable span and its bounded same-block context."""

    kind: str
    text: str
    start: int
    end: int
    context: str
    context_start: int
    context_end: int


class AxisResult(BaseModel):
    """The deterministic result for one required (or optional) axis."""

    status: AxisStatus
    reason: str = ""
    matched_phrases: list[str] = Field(default_factory=list)
    anchor_indices: list[int] = Field(default_factory=list)


class AdequacyEvaluatorIdentity(BaseModel):
    """The frozen policy, code and commit that made the decision."""

    policy_id: str = ""
    policy_sha256: str = ""
    evaluator_id: str = ""
    evaluator_version: str = ""
    evaluator_hash: str = ""
    git_commit: str = ""
    git_commit_matches_evaluator: bool = False
    evaluator_status: str = "unavailable"
    release_eligible: bool = False


class EvidenceAdequacy(BaseModel):
    """Whether one source item can enter value comparison for one claim."""

    status: AdequacyStatus
    document_scope: DocumentScope = DocumentScope.UNKNOWN
    document_sha256: str = ""
    required_axes: list[str] = Field(default_factory=list)
    axis_results: dict[str, AxisResult] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)
    evidence_anchors: list[EvidenceAnchor] = Field(default_factory=list)
    evaluator: AdequacyEvaluatorIdentity = Field(
        default_factory=AdequacyEvaluatorIdentity
    )
