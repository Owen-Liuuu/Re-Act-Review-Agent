"""P1: groups of one paper overlap; claim order is rebuilt from positions."""
from __future__ import annotations

import asyncio
import json

from react_review.agents.split_collector import SplitAwareCollector
from react_review.core.enums import CollectionOutcome
from react_review.llm.base import LLMBackend
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.base import Tool, ToolStage
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry

REFERENCE = ReferenceEntry(study_id="s", title="A trial", doi="10.1/x")
PAPER = ("Age in group A was 31 years. BMI in group A was 22.1 kg/m2.")


class _Fetch(Tool):
    name = "fetch_fulltext"
    stage = ToolStage.EXTRACT

    async def run(self, reference):
        class _Fetched:
            retrieved = True
            document = PaperDocument(paper_id="p1", reference=reference,
                                     full_text=PAPER)
        return _Fetched()


class _SlowBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.05)
        self.in_flight -= 1
        head = prompt.split("## PAPER")[0]
        if "bmi" in head.lower():
            return json.dumps({
                "found": True, "value": "22.1", "unit": "kg/m2",
                "quote": "BMI in group A was 22.1 kg/m2."})
        return json.dumps({
            "found": True, "value": "31", "unit": "years",
            "quote": "Age in group A was 31 years."})


def _claims():
    return [
        ReviewDataItem(study_id="s", group="a", field_type="age",
                       raw_field_name="Age", column_header="Age", value="31"),
        ReviewDataItem(study_id="s", group="a", field_type="bmi",
                       raw_field_name="BMI", column_header="BMI", value="22.1"),
    ]


def _collector(backend):
    registry = ToolRegistry()
    registry.register(_Fetch())
    registry.register(ExtractSourceValueTool(backend))
    return SplitAwareCollector(registry)


def test_two_groups_overlap_and_come_back_in_input_order():
    backend = _SlowBackend()
    result = asyncio.run(_collector(backend).collect_study(_claims(), REFERENCE))
    assert backend.max_in_flight >= 2
    assert [r.source_item.field_type for r in result.claim_results] == ["age", "bmi"]
    assert [r.source_item.source_value for r in result.claim_results] == ["31", "22.1"]


def test_one_group_raising_does_not_cancel_the_other():
    orig = SplitAwareCollector.collect

    async def exploding(claim, *args, **kwargs):
        if claim.field_type == "bmi":
            raise RuntimeError("boom")
        return await orig(collector, claim, *args, **kwargs)

    backend = _SlowBackend()
    collector = _collector(backend)
    collector.collect = exploding
    result = asyncio.run(collector.collect_study(_claims(), REFERENCE))
    by_field = {r.source_item.field_type: r for r in result.claim_results}
    assert by_field["age"].source_item.source_value == "31"
    assert by_field["bmi"].source_item.source_value is None
    assert by_field["bmi"].source_item.collection_outcome is (
        CollectionOutcome.EXTRACTION_FAILED)
    assert any(r.code == "extraction_error" or "boom" in (r.message or "")
               for r in by_field["bmi"].source_item.reasons)
