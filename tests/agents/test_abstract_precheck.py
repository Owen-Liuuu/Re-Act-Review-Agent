"""E: abstract-only claims skip the model when no concept form is present."""
from __future__ import annotations

import pytest

from react_review.agents.split_collector import SplitAwareCollector
from react_review.core.enums import CollectionOutcome
from react_review.dkb import KnowledgeBase, KnowledgeEntry
from react_review.llm.base import LLMBackend
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry

_LI_ABSTRACT = (
    "Background: Minimally invasive esophagectomy (MIE) has been increasingly "
    "used for esophageal cancer. Methods: We compared 98 MIE patients with 87 "
    "open esophagectomy patients between 2009 and 2013. Results: MIE had less "
    "blood loss and shorter hospital stay. Five-year survival did not differ. "
    "Conclusions: MIE is a safe alternative to open surgery."
)
_REF = ReferenceEntry(title="Li 2015", doi="10.1/li")
_BMI = ReviewDataItem(study_id="li_2015", group="mie", field_type="bmi",
                      raw_field_name="BMI", value="24.1", unit="kg/m2")


class _Stub(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return '{"found": true, "value": "24.1", "unit": "kg/m2", "quote": "BMI 24.1"}'


class _Abstract(PaperRetriever):
    def __init__(self, text: str, scope: DocumentScope) -> None:
        self.text = text
        self.scope = scope

    async def retrieve(self, reference):
        return PaperDocument(
            paper_id=reference.doi or "li_2015", reference=reference,
            full_text=self.text, document_scope=self.scope)


def _kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry(
        field_type="bmi", concept="body mass index", default_unit="kg/m2",
        synonyms=["BMI", "body mass index"]))
    return kb


def _collector(retriever, backend) -> SplitAwareCollector:
    registry = ToolRegistry()
    registry.register(FetchFullTextTool(retriever))
    registry.register(ExtractSourceValueTool(backend))
    return SplitAwareCollector(registry, knowledge=_kb())


@pytest.mark.asyncio
async def test_abstract_without_any_variant_skips_the_call():
    backend = _Stub()
    collector = _collector(_Abstract(_LI_ABSTRACT, DocumentScope.ABSTRACT_ONLY), backend)
    res = await collector.collect(_BMI, _REF)
    assert backend.calls == 0
    assert res.source_item.collection_outcome is CollectionOutcome.MISSING_SOURCE
    reason = " ".join(r.message for r in res.source_item.reasons)
    assert "the abstract does not mention this concept in any form" in reason
    assert "bmi" in reason.lower()
    assert "body mass index" in reason.lower()


@pytest.mark.asyncio
async def test_abstract_with_a_variant_still_calls():
    backend = _Stub()
    text = _LI_ABSTRACT + " Mean body mass index was 24.1 kg/m2."
    collector = _collector(_Abstract(text, DocumentScope.ABSTRACT_ONLY), backend)
    await collector.collect(_BMI, _REF)
    assert backend.calls >= 1


@pytest.mark.asyncio
async def test_full_text_does_not_precheck():
    backend = _Stub()
    collector = _collector(_Abstract(_LI_ABSTRACT, DocumentScope.FULL_TEXT), backend)
    await collector.collect(_BMI, _REF)
    assert backend.calls >= 1
