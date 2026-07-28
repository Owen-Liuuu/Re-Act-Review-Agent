"""Agent 1 — Evidence Collector: review-driven directed source extraction.

Given ONE review claim (study, group, field_type), fetch the source paper's full
text and extract that specific value with a verbatim quote, producing a
:class:`SourceEvidenceItem`. A bounded, reflection-driven retry loop handles
"not found" (retry, then escalate); the whole trajectory is recorded as a
ProcessingRecord (AgentRun).

Design note: the fetched document (up to ~60k chars) is held in Python, not
passed through an LLM trajectory — so this Collector orchestrates the tools
deterministically at the document level while the LLM does the directed
extraction inside ``extract_source_value``. (Dual-LLM cross-checking needs a
second backend / ``llm2`` and is deferred.)
"""
from __future__ import annotations

from pydantic import BaseModel

from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.normalize.vocabulary import Vocabulary
from react_review.orchestrator.reflection import ReflectionDecider, ReflectionSignals
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract_source import ExtractSourceValueInput, SourceValueResult
from react_review.tools.registry import ToolRegistry


class CollectResult(BaseModel):
    source_item: SourceEvidenceItem
    record: AgentRun
    decision: ReflectionDecision


class Collector:
    """Collect the source value for review claims via fetch + directed extract."""

    def __init__(
        self,
        catalogue: ToolRegistry,
        *,
        vocabulary: Vocabulary | None = None,
        decider: ReflectionDecider | None = None,
        max_attempts: int = 3,
    ) -> None:
        self._fetch = catalogue.get("fetch_fulltext")
        self._extract = catalogue.get("extract_source_value")
        self._vocab = vocabulary
        self._max_attempts = max(1, max_attempts)
        self._decider = decider or ReflectionDecider(max_attempts=self._max_attempts)

    def _concept_for(self, field_type: str) -> str:
        if self._vocab and field_type in self._vocab.entries:
            return self._vocab.entries[field_type].concept
        return ""

    async def collect(
        self,
        review_item: ReviewDataItem,
        reference: ReferenceEntry,
        *,
        research_context: str = "",
    ) -> CollectResult:
        steps: list[StepRecord] = []

        # 1. Fetch the source paper (deterministic; document held in Python).
        fetched = await self._fetch.run(reference)
        steps.append(StepRecord(
            index=0, thought="fetch source full text", tool="fetch_fulltext",
            args={"doi": reference.doi or "", "title": reference.title[:60]},
            observation={
                "retrieved": fetched.retrieved,
                "source": (fetched.document.metadata.get("source")
                           if fetched.document else None),
            },
        ))
        if not fetched.retrieved or fetched.document is None:
            out = self._decider.decide(ReflectionSignals(retrieval_ok=False, attempt=0))
            return self._result(review_item, SourceValueResult(found=False),
                                 steps, out.decision,
                                 CollectionOutcome.SOURCE_ACCESS_FAILED)

        # 2. Directed extraction, with a bounded reflection-driven retry loop.
        concept = self._concept_for(review_item.field_type)
        result = SourceValueResult(found=False)
        decision = ReflectionDecision.ESCALATE
        for attempt in range(self._max_attempts):
            result = await self._extract.run(ExtractSourceValueInput(
                document=fetched.document,
                field_type=review_item.field_type,
                group=review_item.group,
                concept=concept,
                unit_hint=review_item.unit,
                research_context=research_context,
            ))
            steps.append(StepRecord(
                index=len(steps),
                thought=f"extract {review_item.field_type} for {review_item.group} "
                        f"(attempt {attempt + 1})",
                tool="extract_source_value",
                args={"field_type": review_item.field_type, "group": review_item.group},
                observation=result.model_dump(),
            ))
            # Reflection: a found value is accepted; otherwise retry then escalate.
            decision = self._decider.decide(
                ReflectionSignals(retrieval_ok=result.found, attempt=attempt)
            ).decision
            if result.found or decision != ReflectionDecision.RETRY:
                break

        # Retrieved but the value was never located → potential fabrication.
        outcome = (CollectionOutcome.FOUND if result.found
                   else CollectionOutcome.MISSING_SOURCE)
        return self._result(review_item, result, steps, decision, outcome)

    def _result(
        self,
        review_item: ReviewDataItem,
        result: SourceValueResult,
        steps: list[StepRecord],
        decision: ReflectionDecision,
        outcome: CollectionOutcome = CollectionOutcome.FOUND,
    ) -> CollectResult:
        source_item = SourceEvidenceItem(
            study_id=review_item.study_id,
            group=review_item.group,
            timepoint=review_item.timepoint,
            field_type=review_item.field_type,
            source_value=result.value,
            source_unit=result.unit,
            source_quote=result.quote,
            source_location_in_paper=result.location,
            collection_outcome=outcome,
        )
        record = AgentRun(
            agent="collector",
            task={"study_id": review_item.study_id, "group": review_item.group,
                  "field_type": review_item.field_type},
            steps=steps,
            status="finished",
            final=source_item.model_dump(mode="json"),
        )
        return CollectResult(source_item=source_item, record=record, decision=decision)
