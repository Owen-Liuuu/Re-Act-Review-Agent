"""Same (paper, group, field, outcome) question is asked once; different outcomes are not merged."""
from __future__ import annotations

import json

import pytest

from react_review.llm.base import LLMBackend
from react_review.parser.review_extraction.schemas import ReviewClaim
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
    source_query_from_claim,
)
from react_review.tools.extraction_profile import TARGETED_V7, prompt_version


class _CountBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return json.dumps({"found": False, "not_found_reason": "probe"})


def _payload(**overrides) -> ExtractSourceValueInput:
    doc = PaperDocument(
        paper_id="li_2015",
        reference=ReferenceEntry(title="Li J 2015"),
        full_text="Events 22 of 58 in the MIE arm.",
    )
    fixture = dict(
        document=doc, field_type="event_count", group="mie",
        concept="events", raw_field_name="Events", unit_hint="n",
        cohort_display="MIE", attempt=0)
    return ExtractSourceValueInput(**{**fixture, **overrides})


@pytest.mark.asyncio
async def test_same_key_is_extracted_once():
    backend = _CountBackend()
    tool = ExtractSourceValueTool(backend)
    first = await tool.run(_payload(outcome="overall complications"))
    second = await tool.run(_payload(outcome="overall complications"))
    assert first.not_found_reason == second.not_found_reason == "probe"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_three_forest_outcomes_are_three_queries():
    backend = _CountBackend()
    tool = ExtractSourceValueTool(backend)
    for outcome in ("overall complications", "mortality", "pulmonary complications"):
        await tool.run(_payload(outcome=outcome))
    assert backend.calls == 3


@pytest.mark.asyncio
async def test_attempt_above_zero_is_not_served_from_the_reuse_cache():
    backend = _CountBackend()
    tool = ExtractSourceValueTool(backend)
    await tool.run(_payload(outcome="overall complications", attempt=0))
    await tool.run(_payload(outcome="overall complications", attempt=1))
    assert backend.calls == 2


def test_source_query_copies_claim_outcome():
    claim = ReviewClaim(
        study_id="li_2015", group="mie", field_type="event_count",
        outcome="overall complications")
    query = source_query_from_claim(claim)
    assert query.outcome == "overall complications"


def test_targeted_v7_version_string():
    assert prompt_version("targeted_v7") == TARGETED_V7
