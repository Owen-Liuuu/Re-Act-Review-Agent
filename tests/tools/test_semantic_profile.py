"""The semantic contract is versioned, and the old one is frozen.

The semantic cache is keyed on the prompt version, so a recorded judgement is
reachable only while that version — and the prompt it names — stay put. Asking
the model one more question is therefore a new PROFILE, not an edit: Phase 6's
recordings must keep replaying exactly as they were made.

Phase 7D-1 only changes what is ASKED and recorded. Acting on the answer (a
relation that contradicts its own direction) is 7D-2.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from react_review.audit import ToleranceTable
from react_review.audit.semantic_cache import SemanticCache
from react_review.llm.base import LLMBackend
from react_review.tools.compare import CompareValuesTool
from react_review.tools.models import CompareInput
from react_review.tools.semantic_compare import (
    PROMPT_VERSION,
    SPECIFICITY_VERSION,
    SemanticCompareTool,
    semantic_prompt_version,
)

# Computed from the prompt as Phase 6 recorded it. Not to be updated to match a
# new rendering: that is the defect this pins.
V1_PROMPT_SHA256 = "55644acf41a8ed2e8c99d26ca28269b3a903e30654abe58d03b085b4dcfc8b4c"


class _RecordingBackend(LLMBackend):
    def __init__(self, response: dict | None = None) -> None:
        super().__init__()
        self.prompts: list[str] = []
        self._response = response or {"relation": "same", "equivalent": True,
                                      "confidence": 0.9, "rationale": "r"}

    @property
    def model_id(self) -> str:
        return "glm-4.5-flash"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return json.dumps(self._response)


async def _judge(backend, profile, **kw):
    tool = SemanticCompareTool(backend, profile=profile)
    return await tool.judge(
        field_type=kw.get("field_type", "study_design"),
        column_header=kw.get("column_header", "Design"),
        research_context=kw.get("research_context", "melanoma trials"),
        review_value=kw.get("review_value", "A"),
        source_value=kw.get("source_value", "B"),
        source_quote=kw.get("source_quote", "q"))


@pytest.mark.asyncio
async def test_the_legacy_prompt_bytes_are_unchanged():
    backend = _RecordingBackend()
    await _judge(backend, "semantic_v1")
    digest = hashlib.sha256(backend.prompts[0].encode("utf-8")).hexdigest()
    assert digest == V1_PROMPT_SHA256, (
        "the semantic_v1 prompt changed: every recorded judgement is now "
        "unreachable. A new question belongs in a new profile.")


def test_version_strings_are_distinct_and_stable():
    assert PROMPT_VERSION == "semantic-v1"
    assert SPECIFICITY_VERSION == "semantic-v2-specificity"
    assert semantic_prompt_version("semantic_v1") == PROMPT_VERSION
    assert semantic_prompt_version("semantic_v2_specificity") == SPECIFICITY_VERSION
    assert semantic_prompt_version("") == PROMPT_VERSION


def test_an_unknown_semantic_profile_is_refused():
    with pytest.raises(ValueError, match="unknown semantic profile"):
        semantic_prompt_version("semantic_v9")


@pytest.mark.asyncio
async def test_v1_does_not_ask_for_the_direction():
    backend = _RecordingBackend()
    verdict = await _judge(backend, "semantic_v1")
    assert "more_specific_side" not in backend.prompts[0]
    # Not asked is not the same as answered "unknown".
    assert verdict.more_specific_side == ""
    assert verdict.provenance["prompt_version"] == PROMPT_VERSION


@pytest.mark.asyncio
async def test_v2_asks_for_the_direction_and_records_it():
    backend = _RecordingBackend({"relation": "source_broader",
                                 "more_specific_side": "Review",
                                 "equivalent": False, "confidence": 0.9,
                                 "rationale": "the review adds the dose"})
    verdict = await _judge(backend, "semantic_v2_specificity")
    prompt = backend.prompts[0]
    assert "more_specific_side" in prompt
    assert "WHICH SIDE SAYS MORE" in prompt
    assert verdict.more_specific_side == "review"
    assert verdict.relation == "source_broader"
    assert verdict.provenance["prompt_version"] == SPECIFICITY_VERSION


@pytest.mark.asyncio
async def test_v2_output_skeleton_is_valid_json():
    backend = _RecordingBackend()
    await _judge(backend, "semantic_v2_specificity")
    prompt = backend.prompts[0]
    body = prompt[prompt.index("{"):]
    parsed = json.loads(body.replace("true or false", "true"))
    assert "more_specific_side" in parsed and "relation" in parsed


@pytest.mark.asyncio
async def test_a_v2_judgement_is_not_served_from_a_v1_recording(tmp_path):
    """Two contracts, two keys: the answer to a different question is not reused."""
    cache = SemanticCache(tmp_path / "semantic.json")
    payload = CompareInput(
        field_type="study_design", column_header="Design",
        review_value="Randomized controlled double-blinded Phase III study",
        source_value="randomized, double-blind, phase 3 study",
        source_quote="In this randomized, double-blind, phase 3 study",
        research_context="melanoma trials")

    v1 = CompareValuesTool(
        ToleranceTable(), semantic=SemanticCompareTool(_RecordingBackend()),
        semantic_mode="on", semantic_cache=cache, semantic_profile="semantic_v1")
    await v1.run(payload)
    assert len(cache) == 1

    v2 = CompareValuesTool(
        ToleranceTable(),
        semantic=SemanticCompareTool(_RecordingBackend(),
                                     profile="semantic_v2_specificity"),
        semantic_mode="on", semantic_cache=cache,
        semantic_profile="semantic_v2_specificity")
    await v2.run(payload)
    assert len(cache) == 2, "the v2 question must not read the v1 answer"


@pytest.mark.asyncio
async def test_7d1_does_not_yet_act_on_the_direction():
    """A self-contradicting verdict still passes through — 7D-2 is the gate."""
    backend = _RecordingBackend({"relation": "review_broader",
                                 "more_specific_side": "review",
                                 "equivalent": False, "confidence": 1.0,
                                 "rationale": "the review specifies the dose"})
    tool = CompareValuesTool(
        ToleranceTable(),
        semantic=SemanticCompareTool(backend, profile="semantic_v2_specificity"),
        semantic_mode="on", semantic_profile="semantic_v2_specificity")
    result = await tool.run(CompareInput(
        field_type="treatment_arm", column_header="Arm",
        review_value="Ipilimumab (3 mg/kg) + placebo",
        source_value="ipilimumab group", source_quote="315 to the ipilimumab group"))
    assert result.semantic_relation == "review_broader"
    assert result.review_required is True          # broader always reaches a human
    assert result.match_mode == "semantic"
