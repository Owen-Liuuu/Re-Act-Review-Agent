"""B1 recording and B2: a failed call is not retried and is not missing_source."""
from __future__ import annotations

import json

import pytest

from react_review.agents.collector import Collector
from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.core.exceptions import LLMError
from react_review.llm.base import LLMBackend
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.telemetry import RunTelemetry
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry
from react_review.tools.retry_reason import RETRY_REASONS, bind_retry_attribution

_REVIEW = ReviewDataItem(
    study_id="ahmad_2022", group="t1dm", field_type="eat_thickness",
    value="6.60 ± 0.71", unit="mm")
_REF = ReferenceEntry(title="Ahmad 2022", doi="10.1/x")


class _DocRetriever(PaperRetriever):
    async def retrieve(self, reference):
        return PaperDocument(
            paper_id=reference.doi or "x", reference=reference,
            full_text="Table 2 reports EFT of 6.60 ± 0.71 mm in diabetic children.",
        )


class _JsonBackend(LLMBackend):
    def __init__(self, payload) -> None:
        super().__init__()
        self._payload = payload
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return json.dumps(self._payload)


class _FailBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        raise LLMError(
            "OpenAI API error after 1 attempts: network error: ReadError('')")


def _collector(backend, telemetry: RunTelemetry) -> Collector:
    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_DocRetriever()))
    registry.register(ExtractSourceValueTool(backend, telemetry=telemetry))
    return bind_retry_attribution(
        Collector(registry, telemetry=telemetry, max_attempts=3))


@pytest.mark.asyncio
async def test_not_found_retries_are_attributed_and_sum_to_repeated_attempts():
    telemetry = RunTelemetry()
    backend = _JsonBackend({
        "found": False, "value": None,
        "not_found_reason": "the paper does not report this value",
    })
    result = await _collector(backend, telemetry).collect(_REVIEW, _REF)
    assert backend.calls == 3
    assert result.source_item.collection_outcome is CollectionOutcome.MISSING_SOURCE
    assert telemetry.repeated_attempts == 2
    assert telemetry.retry_reason == {
        "call_failed": 0, "not_found": 2, "low_confidence": 0, "disagreement": 0,
    }
    assert sum(telemetry.retry_reason.values()) == telemetry.repeated_attempts
    assert "unknown" not in telemetry.retry_reason
    assert set(telemetry.retry_reason) == set(RETRY_REASONS)


@pytest.mark.asyncio
async def test_call_failed_is_not_retried_and_is_not_missing_source():
    telemetry = RunTelemetry()
    backend = _FailBackend()
    result = await _collector(backend, telemetry).collect(_REVIEW, _REF)
    assert backend.calls == 1
    assert result.decision is ReflectionDecision.ESCALATE
    assert result.source_item.collection_outcome is CollectionOutcome.EXTRACTION_FAILED
    assert telemetry.repeated_attempts == 0
    assert telemetry.retry_reason is None
    assert "missing_source" not in {
        r.code for r in result.source_item.reasons}
