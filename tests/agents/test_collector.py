"""Tests for extract_source_value + the Collector agent (stub backend, no net)."""
from __future__ import annotations

import json

import pytest

from react_review.agents.collector import Collector
from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.dkb import KnowledgeBase, KnowledgeEntry
from react_review.llm.base import LLMBackend
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
    _group_mismatch,
)
from react_review.tools.registry import ToolRegistry


class StubBackend(LLMBackend):
    def __init__(self, payload) -> None:
        super().__init__()
        self._payload = payload
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)


class _DocRetriever(PaperRetriever):
    async def retrieve(self, reference):
        return PaperDocument(
            paper_id=reference.doi or "x", reference=reference,
            full_text="Table 2 reports EFT of 6.60 ± 0.71 mm in diabetic children.",
        )


class _NoneRetriever(PaperRetriever):
    async def retrieve(self, reference):
        return None


def _catalogue(retriever, extract_backend) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(retriever))
    reg.register(ExtractSourceValueTool(extract_backend))
    return reg


_REVIEW = ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                         field_type="eat_thickness", value="6.60 ± 0.71", unit="mm")
_REF = ReferenceEntry(title="Ahmad 2022", doi="10.1/x")


# --- extract_source_value tool ---

@pytest.mark.asyncio
async def test_extract_tool_found():
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "source_field_name": "EFT",
                           "location": "Table 2"})
    tool = ExtractSourceValueTool(backend)
    doc = PaperDocument(paper_id="x", reference=_REF, full_text="…")
    out = await tool.run(ExtractSourceValueInput(document=doc, field_type="eat_thickness", group="t1dm"))
    assert out.found is True and out.value == "6.60 ± 0.71" and out.unit == "mm"


@pytest.mark.asyncio
async def test_extract_tool_not_found_nullifies_value():
    backend = StubBackend({"found": False, "value": "null"})
    out = await ExtractSourceValueTool(backend).run(
        ExtractSourceValueInput(document=PaperDocument(paper_id="x", reference=_REF),
                                field_type="bmi", group="control"))
    assert out.found is False and out.value is None


@pytest.mark.asyncio
async def test_extract_tool_unparseable_is_not_found():
    out = await ExtractSourceValueTool(StubBackend("not json")).run(
        ExtractSourceValueInput(document=PaperDocument(paper_id="x", reference=_REF),
                                field_type="bmi"))
    assert out.found is False


# --- group-confusion guard ---

@pytest.mark.parametrize("target, label, mismatch", [
    ("control", "Diabetic children", True),    # read the T1DM column for a control ask
    ("control", "Controls", False),
    ("control", "healthy controls", False),
    ("control", "control patients", False),     # "control" wins over generic words
    ("t1dm", "Control group", True),
    ("t1dm", "T1DM patients", False),
    ("t1dm", "Diabetic children", False),
    ("control", "", False),                      # no reported label → no guard
    ("all", "Diabetic children", False),         # non-split group → skip
])
def test_group_mismatch(target, label, mismatch):
    assert _group_mismatch(target, label) is mismatch


@pytest.mark.asyncio
async def test_extract_tool_rejects_wrong_cohort_value():
    backend = StubBackend({"found": True, "value": "12.90 ± 1.30", "unit": "years",
                           "group_label_in_paper": "Diabetic children",
                           "quote": "Age 12.90 ± 1.30", "location": "Table 1"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="control"))
    assert out.found is False and out.value is None      # rejected as wrong cohort
    assert out.group_label_in_paper == "Diabetic children"


@pytest.mark.asyncio
async def test_extract_tool_accepts_matching_cohort():
    backend = StubBackend({"found": True, "value": "12.96 ± 1.12", "unit": "years",
                           "group_label_in_paper": "Healthy controls",
                           "quote": "Controls 12.96 ± 1.12", "location": "Table 1"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="control"))
    assert out.found is True and out.value == "12.96 ± 1.12"


@pytest.mark.asyncio
async def test_extract_tool_rejects_when_quote_names_wrong_cohort():
    # The model faked a matching label but its quote is a diabetic-only sentence
    # (the real Ahmad failure: inferring control from "no significant difference").
    backend = StubBackend({"found": True, "value": "12.90 ± 1.30", "unit": "years",
                           "group_label_in_paper": "healthy controls",
                           "quote": "Regarding diabetic children, the mean age was "
                                    "12.90 ± 1.30 years.", "location": "Results"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="control"))
    assert out.found is False and out.value is None


# --- Collector ---

@pytest.mark.asyncio
async def test_collector_produces_source_item():
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "source_field_name": "EFT",
                           "location": "Table 2"})
    collector = Collector(_catalogue(_DocRetriever(), backend))
    res = await collector.collect(_REVIEW, _REF, research_context="EAT in T1DM")
    assert res.decision == ReflectionDecision.ACCEPT
    assert res.source_item.source_value == "6.60 ± 0.71"
    assert res.source_item.source_unit == "mm"
    assert res.source_item.study_id == "ahmad_2022"
    assert res.source_item.collection_outcome == CollectionOutcome.FOUND
    assert backend.calls == 1  # found on first attempt, no retry
    # trajectory recorded: fetch + one extract
    assert [s.tool for s in res.record.steps] == ["fetch_fulltext", "extract_source_value"]


@pytest.mark.asyncio
async def test_collector_retries_then_escalates_when_not_found():
    backend = StubBackend({"found": False, "value": None})
    collector = Collector(_catalogue(_DocRetriever(), backend), max_attempts=3)
    res = await collector.collect(_REVIEW, _REF)
    assert res.decision == ReflectionDecision.ESCALATE
    assert res.source_item.source_value is None
    # retrieved the paper but never located the value → potential fabrication
    assert res.source_item.collection_outcome == CollectionOutcome.MISSING_SOURCE
    assert backend.calls == 3  # tried max_attempts times


def _kb_thickness() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry(field_type="eat_thickness",
                          concept="epicardial fat thickness", default_unit="mm"))
    return kb


@pytest.mark.asyncio
async def test_collector_back_check_flags_contradicting_concept():
    # A CANDIDATE mapping (eat_thickness, expects mm) but the source reports a
    # volume (cm3) → the source evidence refutes the parse-time translation.
    backend = StubBackend({"found": True, "value": "52.3", "unit": "cm3",
                           "quote": "EAT volume 52.3 cm3", "location": "Table 1"})
    review = ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                            field_type="eat_thickness", raw_field_name="EAT",
                            value="52.3", unit="cm3", resolution_status="candidate")
    collector = Collector(_catalogue(_DocRetriever(), backend), knowledge=_kb_thickness())
    res = await collector.collect(review, _REF)
    assert res.source_item.concept_mismatch is True
    assert "different kind" in res.source_item.concept_mismatch_reason


@pytest.mark.asyncio
async def test_collector_no_back_check_for_resolved_items():
    # A trusted (resolved) concept is never second-guessed by the back-check.
    backend = StubBackend({"found": True, "value": "52.3", "unit": "cm3",
                           "quote": "x", "location": "y"})
    review = ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                            field_type="eat_thickness", value="52.3", unit="cm3",
                            resolution_status="resolved")
    collector = Collector(_catalogue(_DocRetriever(), backend), knowledge=_kb_thickness())
    res = await collector.collect(review, _REF)
    assert res.source_item.concept_mismatch is False


@pytest.mark.asyncio
async def test_collector_escalates_when_paper_not_retrieved():
    backend = StubBackend({"found": True, "value": "x"})
    collector = Collector(_catalogue(_NoneRetriever(), backend))
    res = await collector.collect(_REVIEW, _REF)
    assert res.decision == ReflectionDecision.RETRY or res.decision == ReflectionDecision.ESCALATE
    assert res.source_item.source_value is None
    # never got the paper → access failure, NOT a fabrication signal
    assert res.source_item.collection_outcome == CollectionOutcome.SOURCE_ACCESS_FAILED
    assert backend.calls == 0  # never reached extraction
    assert res.record.steps[0].observation["retrieved"] is False
