"""Auditable records for one field-normalisation decision.

The review parser applies a field mapping to many table cells, but the mapping
itself is a run-level decision.  Keeping that decision in its own schema avoids
duplicating a model trace on every :class:`ReviewDataItem` while still letting a
reader follow any item back to the exact resolution question that produced it.

Raw model responses belong in the step journal.  The evidence package keeps the
parsed summary plus prompt/response hashes, which is enough to identify and
compare attempts without making ``package.json`` grow with model prose.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from react_review.schemas.reason import ReasonRecord


class ResolutionAttempt(BaseModel):
    """A compact, reproducible summary of one model classification attempt."""

    seed: int = 42
    model_id: str = ""
    field_type: str = ""
    is_new: bool = False
    value_type: str = ""
    default_unit: str = ""
    scope: str = "cohort"
    grounded_on: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    prompt_sha256: str = ""
    response_sha256: str = ""
    error: str = ""


class ResolutionCellRef(BaseModel):
    """One review-table cell affected by a field-resolution decision."""

    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    study_id: str = ""
    column_header: str = ""
    unit: str = ""
    status: str = ""
    field_type: str = ""
    reason: str = ""


class FieldResolutionRecord(BaseModel):
    """The complete run-level record for one unique resolution question."""

    resolution_key: str
    raw_field_name: str
    unit: str = ""
    modality: str = ""
    research_context_sha256: str = ""
    field_type: str | None = None
    # authoritative | candidate | unresolved | mixed.  ``mixed`` is possible
    # when a value-dependent check rejects one row but accepts another row that
    # otherwise asks the same field-resolution question.
    status: str = "unresolved"
    scope: str = "cohort"
    source: str = "none"
    grounded_on: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    checks: dict[str, bool] = Field(default_factory=dict)
    reasons: list[ReasonRecord] = Field(default_factory=list)
    attempts: list[ResolutionAttempt] = Field(default_factory=list)
    # not_checked for deterministic hits; stable when >=2 sampled attempts have
    # the same structural signature; unstable when no two attempts agree.
    stability: str = "not_checked"
    consensus_count: int = 0
    candidate_names: list[str] = Field(default_factory=list)
    # A serialised KnowledgeEntry proposal.  It is deliberately a plain mapping
    # here so the evidence schemas do not import the DKB package and form a cycle.
    proposal: dict[str, Any] | None = None
    affected_cells: list[ResolutionCellRef] = Field(default_factory=list)
    statuses_seen: list[str] = Field(default_factory=list)
    field_types_seen: list[str] = Field(default_factory=list)
    cache_hits: int = 0
