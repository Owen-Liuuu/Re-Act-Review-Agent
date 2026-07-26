"""The Evidence Package: everything one audit run produces, for persistence.

Maps to the architecture's Evidence Package Store contents:
  review_items        → Data Reference List (Claim)
  source_items        → Cited Data Source List
  report              → the audit result (Preliminary/Final)
  processing_records  → Processing Record (agent trajectories)

(The Conclusion List is deferred to P4.)
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from react_review.schemas.agent import AgentRun
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.report import AuditReport


class EvidencePackage(BaseModel):
    """The full, serialisable state of one audit run."""

    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    review_items: list[ReviewDataItem] = Field(default_factory=list)
    source_items: list[SourceEvidenceItem] = Field(default_factory=list)
    report: AuditReport | None = None
    processing_records: list[AgentRun] = Field(default_factory=list)
