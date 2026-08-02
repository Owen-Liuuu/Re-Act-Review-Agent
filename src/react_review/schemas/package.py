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
from react_review.schemas.report import AuditReport, FinalVerification
from react_review.schemas.table import CapturedTableSet


class EvidencePackage(BaseModel):
    """The full, serialisable state of one audit run."""

    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    review_items: list[ReviewDataItem] = Field(default_factory=list)
    source_items: list[SourceEvidenceItem] = Field(default_factory=list)
    report: AuditReport | None = None
    final_verification: FinalVerification | None = None
    processing_records: list[AgentRun] = Field(default_factory=list)
    # The verbatim tables the review claims were read from, as approved at the
    # capture checkpoint — so any audited value can be traced back to its cell.
    captured_tables: CapturedTableSet = Field(default_factory=CapturedTableSet)
    # How the run ended: complete | stopped_by_user | interrupted | error.
    # A partial package is still evidence — it records what HAD been checked.
    status: str = "complete"
    stopped_at_stage: str = ""
