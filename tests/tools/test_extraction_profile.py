"""The legacy extraction contract is pinned, byte for byte.

Phase 6's recorded caches are keyed on ``prompt_version`` plus the SHA-256 of
the prompt itself, so any drift in either — a renamed version string, an extra
line in the output skeleton, one changed space — silently invalidates every
recorded response and turns a frozen replay into a live run. These tests fail on
that drift instead of discovering it during a paid recording.

The prompt is captured from the REAL tool call rather than rebuilt here: a copy
of the formatting logic in the test would drift with the code it is meant to
pin.
"""
from __future__ import annotations

import hashlib
import json

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
    prompt_profile,
    prompt_version,
)
from react_review.llm.base import LLMBackend

# Computed from the code as it stood when Phase 6E was recorded. These constants
# must NOT be updated to match new output: a legacy-profile change is the defect.
LEGACY_PROMPT_SHA256 = "0c9c640c16953a21ee3519322b1f4b49532733bc2fb8e14aa3ac042ced0cb598"
LEGACY_CACHE_KEY = "8b1d5b403d0342907084c5160207cf97cd1ba93becb4a37356f67b769e347c69"


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
    """One fixed request. Every field that reaches the prompt is spelled out."""
    document = PaperDocument(
        paper_id="pin", reference=ReferenceEntry(title="pinned paper"),
        full_text="Table 1. Age (years) 12.90 +/- 1.30 12.96 +/- 1.12")
    return ExtractSourceValueInput(
        document=document, field_type="age", group="control", concept="age",
        concept_variants=["age", "age (years)"], raw_field_name="Age (years)",
        unit_hint="years", research_context="pinned context",
        cohort_display="Controls",
        cohorts={"t1dm": ["T1DM"], "control": ["Controls"]},
        attempt=0, **overrides)


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
