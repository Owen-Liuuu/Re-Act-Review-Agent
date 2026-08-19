"""Every single-target extraction contract is pinned, byte for byte.

Phase 6's recorded caches are keyed on ``prompt_version`` plus the SHA-256 of
the prompt itself, so any drift in either — a renamed version string, an extra
line in the output skeleton, one changed space — silently invalidates every
recorded response and turns a frozen replay into a live run. These tests fail on
that drift instead of discovering it during a paid recording.

``targeted_v6`` is pinned here for the same reason and BEFORE it has recordings,
so its first recording is made against agreed bytes rather than whatever the
template happened to say that day. It is also the file that states what v6 is
for: v4's question with its cohort examples written as placeholders. That is
checked as an invariant — the two prompts must differ in the examples and
NOWHERE else — because a hash alone would still pass if someone changed a rule
in one profile and not the other.

The prompt is captured from the REAL tool call rather than rebuilt here: a copy
of the formatting logic in the test would drift with the code it is meant to
pin.
"""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract_source import (
    PROMPT_VERSION,
    ExtractSourceValueInput,
    ExtractSourceValueTool,
)
from react_review.tools.extraction_cache import ExtractionCache
from react_review.tools.extraction_profile import (
    LEGACY_V3,
    TARGETED_V4,
    TARGETED_V6,
    TARGETED_V7,
    BATCH_SPLIT_V1,
    is_batch_route,
    prompt_profile,
    prompt_version,
    uses_targeted_sections,
)
from react_review.llm.base import LLMBackend

# Computed from the code as it stood when Phase 6E was recorded. These constants
# must NOT be updated to match new output: a legacy-profile change is the defect.
LEGACY_PROMPT_SHA256 = "0c9c640c16953a21ee3519322b1f4b49532733bc2fb8e14aa3ac042ced0cb598"
LEGACY_CACHE_KEY = "8b1d5b403d0342907084c5160207cf97cd1ba93becb4a37356f67b769e347c69"
# targeted_v4 carries the melanoma benchmark's recordings; v6 is the neutral
# candidate. Same rule: a changed hash is a new profile, not a new constant here.
TARGETED_V4_PROMPT_SHA256 = "7c3ab12261b4a352e2e103d1edf61029400f7ce27468a8bfefe297351294e3f8"
TARGETED_V6_PROMPT_SHA256 = "33cdd50fe6691e81f5a3de6cdddc61181fad906f79065937f7289beb3d970d05"
TARGETED_V6_CACHE_KEY = "414c0bc9f8a6cb0540f2db6e4ef495c1acff164848c052503d32b76ad33a879f"

# The block the two targeted profiles are allowed to differ in, located by the
# text around it so the test does not restate the rules themselves.
_RULES_START = "- PREFER the data table"
_RULES_END = "- First list every cohort/column"
# Present only when the enumerate-then-assign sections are rendered.
_TARGETED_MARKER = "- Do NOT pick the target arm yourself."


class _RecordingBackend(LLMBackend):
    """Captures the prompt the tool actually sent."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "glm-4.5-flash"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return json.dumps({"found": False, "not_found_reason": "pinned probe"})


def _pinned_payload(**overrides) -> ExtractSourceValueInput:
    """One fixed request. Every field that reaches the prompt is spelled out.

    Overrides are merged over the fixture rather than passed alongside it, so a
    test can vary one field — the cohorts, say — without a duplicate-argument
    error. With no overrides the request is byte-identical to the one the pinned
    hashes were computed from.
    """
    document = PaperDocument(
        paper_id="pin", reference=ReferenceEntry(title="pinned paper"),
        full_text="Table 1. Age (years) 12.90 +/- 1.30 12.96 +/- 1.12")
    fixture = dict(
        document=document, field_type="age", group="control", concept="age",
        concept_variants=["age", "age (years)"], raw_field_name="Age (years)",
        unit_hint="years", research_context="pinned context",
        cohort_display="Controls",
        cohorts={"t1dm": ["T1DM"], "control": ["Controls"]},
        attempt=0)
    return ExtractSourceValueInput(**{**fixture, **overrides})


@pytest.mark.asyncio
async def test_legacy_prompt_bytes_are_unchanged():
    backend = _RecordingBackend()
    await ExtractSourceValueTool(backend).run(_pinned_payload())
    assert len(backend.prompts) == 1
    digest = hashlib.sha256(backend.prompts[0].encode("utf-8")).hexdigest()
    assert digest == LEGACY_PROMPT_SHA256, (
        "the legacy extraction prompt changed: every Phase 6 recording is now "
        "unreachable. A new contract belongs in a new profile.")


@pytest.mark.asyncio
async def test_legacy_cache_key_is_unchanged(tmp_path):
    """The key the TOOL writes, not one this test recomputes."""
    cache = ExtractionCache(tmp_path / "record.json")
    tool = ExtractSourceValueTool(_RecordingBackend(), cache=cache,
                                  cache_mode="record")
    await tool.run(_pinned_payload())
    body = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert list(body["entries"]) == [LEGACY_CACHE_KEY]


def test_legacy_version_string_is_unchanged():
    assert PROMPT_VERSION == LEGACY_V3 == "extract-source-v3-scoped-cohort-counts"
    assert prompt_version("legacy_v3") == LEGACY_V3
    assert prompt_version("targeted_v4") == TARGETED_V4
    assert LEGACY_V3 != TARGETED_V4


def test_profile_defaults_to_legacy_and_is_explicit():
    assert prompt_profile(_pinned_payload()) == "legacy_v3"
    assert prompt_profile(
        _pinned_payload(extraction_profile="targeted_v4")) == "targeted_v4"


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="unknown extraction profile"):
        prompt_profile(_pinned_payload(extraction_profile="v5_experiment"))
    with pytest.raises(ValueError, match="unknown extraction profile"):
        prompt_version("v5_experiment")


@pytest.mark.asyncio
async def test_profile_does_not_leak_into_the_legacy_prompt(tmp_path):
    """Carrying a profile field must not, by itself, change what is sent."""
    backend = _RecordingBackend()
    await ExtractSourceValueTool(backend).run(
        _pinned_payload(extraction_profile="legacy_v3"))
    assert hashlib.sha256(
        backend.prompts[0].encode("utf-8")).hexdigest() == LEGACY_PROMPT_SHA256


# --- targeted_v6: v4's question, without one review's disease in the examples --

async def _rendered(profile: str, **overrides) -> str:
    """The prompt the tool really sends under a profile."""
    backend = _RecordingBackend()
    await ExtractSourceValueTool(backend).run(
        _pinned_payload(extraction_profile=profile, **overrides))
    assert len(backend.prompts) == 1
    return backend.prompts[0]


def _rules_block(prompt: str) -> str:
    return prompt[prompt.index(_RULES_START):prompt.index(_RULES_END)]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile, expected", [
    ("targeted_v4", TARGETED_V4_PROMPT_SHA256),
    ("targeted_v6", TARGETED_V6_PROMPT_SHA256),
])
async def test_targeted_prompt_bytes_are_unchanged(profile, expected):
    digest = hashlib.sha256((await _rendered(profile)).encode("utf-8")).hexdigest()
    assert digest == expected, (
        f"the {profile} extraction prompt changed: recordings made under it are "
        "now unreachable. A new contract belongs in a new profile")


@pytest.mark.asyncio
async def test_targeted_v6_cache_key_is_unchanged(tmp_path):
    cache = ExtractionCache(tmp_path / "record.json")
    tool = ExtractSourceValueTool(_RecordingBackend(), cache=cache,
                                  cache_mode="record")
    await tool.run(_pinned_payload(extraction_profile="targeted_v6"))
    body = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert list(body["entries"]) == [TARGETED_V6_CACHE_KEY]


@pytest.mark.asyncio
async def test_v6_is_v4_with_only_the_cohort_examples_replaced():
    """The point of v6, stated as an invariant rather than as two hashes.

    Two independent hashes would stay green if a rule were tightened in v6 and
    not v4, which would make an A/B between them measure two changes at once.
    """
    v4, v6 = await _rendered("targeted_v4"), await _rendered("targeted_v6")
    assert _rules_block(v4) != _rules_block(v6)
    assert v4.replace(_rules_block(v4), _rules_block(v6)) == v6, (
        "v4 and v6 differ somewhere other than the cohort examples, so an A/B "
        "between them would not isolate the wording change")


@pytest.mark.asyncio
async def test_v6_names_no_disease_while_v4_still_does():
    """v4 keeps its recorded wording; only v6 is neutral.

    The cohorts come from the request, so a neutral request is used: what must
    contain no disease is the TEMPLATE, not a review's own labels.
    """
    neutral = {"cohorts": {"arm_a": ["Arm A"], "control": ["Controls"]}}
    v4 = (await _rendered("targeted_v4", **neutral)).lower()
    v6 = (await _rendered("targeted_v6", **neutral)).lower()

    def names(term: str, prompt: str) -> bool:
        # Whole words only: "eft" occurs inside "left_label" and "eat" inside
        # "repeated", and neither is this domain leaking into the prompt.
        return re.search(rf"\b{term}\b", prompt) is not None

    for term in ("diabetic", "diabetes", "t1dm", "eat", "eft"):
        assert not names(term, v6), f"targeted_v6 still names {term!r}"
    assert names("diabetic", v4), (
        "v4 was edited to be neutral — that silently invalidates the melanoma "
        "recordings; neutral wording is what v6 is for")


@pytest.mark.asyncio
async def test_v6_asks_the_model_to_enumerate_arms_like_v4():
    """A neutral prompt that lost the enumerate-then-assign sections would be a
    different contract wearing a wording change's name."""
    assert uses_targeted_sections("targeted_v6")
    assert uses_targeted_sections("targeted_v4")
    assert not uses_targeted_sections("legacy_v3")
    assert _TARGETED_MARKER in await _rendered("targeted_v6")


def test_v6_version_string_is_registered_and_distinct():
    assert prompt_version("targeted_v6") == TARGETED_V6
    assert prompt_version("targeted_v7") == TARGETED_V7
    assert len({LEGACY_V3, TARGETED_V4, TARGETED_V6, TARGETED_V7}) == 4


@pytest.mark.asyncio
async def test_v7_without_outcome_matches_v6_bytes():
    """v7 is v6 plus an outcome clause that is empty when none is supplied."""
    assert await _rendered("targeted_v6") == await _rendered("targeted_v7")


@pytest.mark.asyncio
async def test_v7_interpolates_outcome_and_frozen_profiles_do_not():
    outcome = "overall complications"
    v4 = await _rendered("targeted_v4", outcome=outcome)
    v6 = await _rendered("targeted_v6", outcome=outcome)
    v7 = await _rendered("targeted_v7", outcome=outcome)
    assert outcome not in v4
    assert outcome not in v6
    assert outcome in v7
    assert hashlib.sha256(v4.encode("utf-8")).hexdigest() == TARGETED_V4_PROMPT_SHA256
    assert hashlib.sha256(v6.encode("utf-8")).hexdigest() == TARGETED_V6_PROMPT_SHA256
    assert uses_targeted_sections("targeted_v7")


def test_targeted_v7_contract_pins_the_rendered_prompt_with_outcome():
    """The new profile has a contract file; its hash is of the rendered prompt."""
    from react_review.contracts import repo_root

    body = json.loads(
        (repo_root() / "configs/prompt_contracts/targeted_v7.json"
         ).read_text(encoding="utf-8"))
    assert body["extraction_profile"] == "targeted_v7"
    assert body["prompt_version"] == TARGETED_V7
    assert body["rendered_prompt_sha256"] != "PENDING"


@pytest.mark.asyncio
async def test_targeted_v7_contract_hash_matches_what_the_tool_sends():
    from react_review.contracts import repo_root

    body = json.loads(
        (repo_root() / "configs/prompt_contracts/targeted_v7.json"
         ).read_text(encoding="utf-8"))
    fixture = body["fixture_inputs"]
    prompt = await _rendered(
        "targeted_v7",
        outcome=fixture["outcome"],
        research_context=fixture["context"],
    )
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()
    assert digest == body["rendered_prompt_sha256"]


def test_an_unknown_profile_cannot_be_read_as_not_targeted():
    """Returning False would render the legacy body under an undefined name."""
    with pytest.raises(ValueError, match="unknown extraction profile"):
        uses_targeted_sections("targeted_v9_missing")


def test_batch_split_is_a_batch_route_and_v5_stays_one():
    """v5 one-shot stays frozen; split is a second batch route, not an edit."""
    assert is_batch_route("targeted_v5_batch")
    assert is_batch_route("batch_split_v1")
    assert not is_batch_route("targeted_v4")
    assert prompt_version("batch_split_v1") == BATCH_SPLIT_V1
    assert BATCH_SPLIT_V1 != prompt_version("targeted_v5_batch")
