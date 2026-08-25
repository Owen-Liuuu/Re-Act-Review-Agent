"""C/D: a billing refusal is not a missing value, and it must stop the run."""
from __future__ import annotations

import pytest

from react_review.agents.split_collector import SplitAwareCollector
from react_review.core.exceptions import LLMError, PermanentProviderError
from react_review.llm.base import LLMBackend
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract_source import ExtractSourceValueInput, ExtractSourceValueTool
from tests.agents.test_collector import _REVIEW, _REF, _catalogue


class _Paywall(LLMBackend):
    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "paywall"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        raise LLMError(
            f"OpenAI API error (HTTP {self.status}): "
            + ("Insufficient Balance" if self.status == 402 else "Too Many Requests"))


class _CountingRetriever(PaperRetriever):
    def __init__(self) -> None:
        self.opens = 0

    async def retrieve(self, reference):
        self.opens += 1
        return PaperDocument(
            paper_id=reference.doi or "x", reference=reference,
            full_text="Table 2 reports EFT of 6.60 ± 0.71 mm in diabetic children.")


@pytest.mark.asyncio
async def test_extract_tool_raises_on_http_402_with_the_real_reason():
    backend = _Paywall(402)
    with pytest.raises(PermanentProviderError) as caught:
        await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text="EFT 6.60"),
            field_type="eat_thickness", group="t1dm"))
    assert caught.value.status == 402
    assert "HTTP 402" in caught.value.not_found_reason
    assert "Insufficient Balance" in caught.value.not_found_reason
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_http_429_is_not_a_permanent_failure():
    backend = _Paywall(429)
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF, full_text="EFT 6.60"),
        field_type="eat_thickness", group="t1dm"))
    assert out.found is False
    assert out.permanent_failure is False
    assert "HTTP 429" in out.not_found_reason
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_collector_does_not_retry_a_402():
    backend = _Paywall(402)
    collector = SplitAwareCollector(_catalogue(_CountingRetriever(), backend),
                                    max_attempts=3)
    with pytest.raises(PermanentProviderError, match="HTTP 402"):
        await collector.collect(_REVIEW, _REF)
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_tamper_429_restores_retries():
    backend = _Paywall(429)
    collector = SplitAwareCollector(_catalogue(_CountingRetriever(), backend),
                                    max_attempts=3)
    res = await collector.collect(_REVIEW, _REF)
    assert backend.calls == 3
    messages = " ".join(r.message for r in res.source_item.reasons)
    assert "HTTP 429" in messages
    assert res.source_item.collection_outcome.value == "missing_source"


@pytest.mark.asyncio
async def test_collect_study_aborts_before_the_second_claim():
    backend = _Paywall(402)
    second = ReviewDataItem(study_id="ahmad_2022", group="control",
                            field_type="eat_thickness", value="3.83", unit="mm")
    collector = SplitAwareCollector(_catalogue(_CountingRetriever(), backend),
                                    max_attempts=3)
    with pytest.raises(PermanentProviderError, match="cannot succeed"):
        await collector.collect_study([_REVIEW, second], _REF)
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_402_is_not_reported_as_a_paper_omission():
    backend = _Paywall(402)
    collector = SplitAwareCollector(_catalogue(_CountingRetriever(), backend))
    with pytest.raises(PermanentProviderError) as caught:
        await collector.collect(_REVIEW, _REF)
    blob = str(caught.value) + caught.value.not_found_reason
    assert "HTTP 402" in blob
    assert "not stated" not in blob
    assert "fabrication" not in blob
