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

from pydantic import BaseModel, Field, model_serializer

from react_review.checklist.schema import ChecklistApplication
from react_review.normalize.cohorts import CohortRegistry
from react_review.schemas.agent import AgentRun
from react_review.schemas.batch import BatchReadingRecord
from react_review.schemas.telemetry import RunTelemetry
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.knowledge import KnowledgeImportRecord
from react_review.schemas.report import AuditReport, FinalVerification
from react_review.schemas.resolution import FieldResolutionRecord
from react_review.schemas.run_manifest import RunManifest
from react_review.schemas.table import CapturedTableSet


class EvidencePackage(BaseModel):
    """The full, serialisable state of one audit run."""

    run_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    # WHICH RULES produced this, and how it was executed. Absent on packages
    # written before run contracts existed; a partial package carries the
    # manifest without its cache hashes, because a file still being appended to
    # has no content hash worth recording.
    run_manifest: RunManifest | None = None
    review_items: list[ReviewDataItem] = Field(default_factory=list)
    source_items: list[SourceEvidenceItem] = Field(default_factory=list)
    report: AuditReport | None = None
    final_verification: FinalVerification | None = None
    processing_records: list[AgentRun] = Field(default_factory=list)
    #: One entry per BATCH reading, not per claim. Every claim answered by a
    #: batch names its execution id, and this is where that reference resolves;
    #: without it the reference points at nothing and the claim that several
    #: answers came from one act of reading is unverifiable.
    batch_records: list[BatchReadingRecord] = Field(default_factory=list)
    #: What this run cost. Absent from every package written before it existed,
    #: and from any run that measured nothing — a production run should be able
    #: to say what it spent, and until now only the eval harness could.
    telemetry: RunTelemetry | None = None

    @model_serializer(mode="wrap")
    def _omit_unused_batch_records(self, handler):
        """A package with no batch readings gains no key for them.

        Same rule as the evidence row, for the same reason and by the same
        mechanism: a replay compares BYTES, and an empty list written into every
        package ever recorded is a changed artifact for a fact nothing had.
        """
        body = handler(self)
        if not body.get("batch_records"):
            body.pop("batch_records", None)
        # By the COUNTS, not by the object. A zero RunTelemetry serialises to a
        # full dictionary of zeroes and is therefore truthy, so testing the body
        # decided nothing and every package gained the key regardless.
        if self.telemetry is None or not self.telemetry.has_measurements():
            body.pop("telemetry", None)
        # Only a run that ended early has a reason to give. Writing an empty one
        # into every completed package would change bytes that nothing changed.
        if not body.get("stop_reason"):
            body.pop("stop_reason", None)
        if not body.get("finalisation_errors"):
            body.pop("finalisation_errors", None)
        if not body.get("superseded_by"):
            body.pop("superseded_by", None)
        return body
    # The verbatim tables the review claims were read from, as approved at the
    # capture checkpoint — so any audited value can be traced back to its cell.
    captured_tables: CapturedTableSet = Field(default_factory=CapturedTableSet)
    # The cohorts this review was found to report — so a reader can see what the
    # arms were called and how each claim's group was arrived at.
    cohorts: CohortRegistry = Field(default_factory=CohortRegistry)
    # Every field mapping applied to ``review_items``.  Review rows carry only a
    # ``resolution_key``; the full attempts/checks/reasons live once here.
    field_resolutions: list[FieldResolutionRecord] = Field(default_factory=list)
    # Exact ontology files and conflict policy that produced the runtime KB.
    knowledge_imports: list[KnowledgeImportRecord] = Field(default_factory=list)
    # Snapshot identity for the *effective* KB after seed + ontology merging.
    # Import records alone cannot prove which seed version was used.
    knowledge_fingerprint: str = ""
    knowledge_concept_count: int = 0
    # Clinician-editable coverage questions and every required gap found.
    checklist: ChecklistApplication | None = None
    # How the run ended: complete | stopped_by_user | interrupted | error.
    # A partial package is still evidence — it records what HAD been checked.
    status: str = "complete"
    stopped_at_stage: str = ""
    #: WHY it ended there, in the artifact rather than only on the terminal.
    #: The reason was printed and then lost, so a stopped run's file could say
    #: where it stopped but never what stopped it — which is the part a reader
    #: coming back to the directory a day later actually needs.
    stop_reason: str = ""
    #: Failures encountered while making the outcome durable. Kept beside the
    #: original stop/error reason so a cache or disk failure cannot replace the
    #: event that caused finalisation in the first place.
    finalisation_errors: list[dict[str, str]] = Field(default_factory=list)
    #: A completed final package may supersede a progress artifact when the
    #: latter cannot be removed. This is omitted from ordinary packages.
    superseded_by: str = ""
