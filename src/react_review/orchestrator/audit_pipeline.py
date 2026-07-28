"""Full audit pipeline: Collector → Auditor → Judge → EvidencePackage.

Sequences the P2 agents deterministically (the orchestrator owns control flow):
for each review claim the Collector finds the source value; the Auditor
(AuditOrchestrator) matches + compares review vs source; the Judge adjudicates
and flags items for human review. The whole run is persisted as an
EvidencePackage. The review items themselves come from the ReviewParser (run by
the caller / CLI) or from a CSV.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Callable

import structlog

from react_review.orchestrator.judge import Judge
from react_review.orchestrator.pipeline import AuditOrchestrator
from react_review.schemas.agent import AgentRun
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.package import EvidencePackage
from react_review.store.evidence_package import EvidencePackageStore

if TYPE_CHECKING:  # avoid an agents <-> orchestrator import cycle at runtime
    from react_review.agents.collector import Collector
    from react_review.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)

ReferenceResolver = Callable[[str], "ReferenceEntry"]


class AuditPipeline:
    """Collect source evidence, audit it, adjudicate, and persist."""

    def __init__(
        self,
        collector: Collector,
        auditor: AuditOrchestrator,
        judge: Judge,
        *,
        store: EvidencePackageStore | None = None,
    ) -> None:
        self._collector = collector
        self._auditor = auditor
        self._judge = judge
        self._store = store

    async def run(
        self,
        review_items: list[ReviewDataItem],
        reference_for: ReferenceResolver,
        *,
        research_context: str = "",
        run_id: str | None = None,
        parser_record: AgentRun | None = None,
    ) -> EvidencePackage:
        run_id = run_id or uuid.uuid4().hex[:12]
        logger.info("audit_pipeline_start", run_id=run_id, n_review=len(review_items))

        source_items = []
        records: list[AgentRun] = [parser_record] if parser_record else []
        for item in review_items:
            result = await self._collector.collect(
                item, reference_for(item.study_id), research_context=research_context
            )
            source_items.append(result.source_item)
            records.append(result.record)

        report = await self._auditor.run(review_items, source_items, run_id=run_id)
        final = self._judge.adjudicate(report, source_items)

        package = EvidencePackage(
            run_id=run_id,
            review_items=review_items,
            source_items=source_items,
            report=report,
            final_verification=final,
            processing_records=records,
        )
        if self._store is not None:
            self._store.save(package)

        logger.info(
            "audit_pipeline_done", run_id=run_id, verdict=final.verdict.value,
            flags=len(final.human_review_flags),
        )
        return package
