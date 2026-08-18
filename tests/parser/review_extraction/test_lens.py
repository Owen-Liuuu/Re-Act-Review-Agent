"""Lens compression: short fields, no invented content, later steps never see the abstract."""
from __future__ import annotations

import json

import pytest

from react_review.llm.base import LLMBackend
from react_review.parser.review_extraction.lens import read_lens
from react_review.parser.review_extraction.prompts import ExtractionPromptContract
from react_review.parser.review_extraction.windows import clip_words, front_matter


class QueueBackend(LLMBackend):
    def __init__(self, responses: list) -> None:
        super().__init__()
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else "{}"


DOC05_FRONT = """
Minimally invasive esophagectomy in elderly patients

Abstract
Background: Elderly patients aged 70 years and over with resectable ESCC.
We compared MIE versus open esophagectomy for postoperative complications.

Introduction
Esophageal cancer surgery is common.
"""


def test_clip_words_enforces_the_hard_limit():
    assert clip_words("one two three four five", 3) == "one two three"
    assert clip_words("short", 40) == "short"


def test_front_matter_stops_at_introduction():
    window = front_matter(DOC05_FRONT)
    assert "Abstract" in window and "MIE" in window
    assert "Esophageal cancer surgery is common" not in window


def test_lens_contract_does_not_drift():
    assert ExtractionPromptContract.load("review_lens_v1").drifts() == []


@pytest.mark.asyncio
async def test_read_lens_clips_fields_and_keeps_stated_terms():
    backend = QueueBackend([{
        "lens_one_line": " ".join(["word"] * 50),
        "domain": "esophageal cancer surgery extra words that overflow",
        "population": "elderly / ≥70, resectable ESCC",
        "comparison": "MIE vs open esophagectomy",
        "outcomes": ["overall complications", "pulmonary", "30-day mortality",
                     "anastomotic leak"],
        "not_audit_focus": ["pooled GRADE"],
    }])
    lens = await read_lens(backend, DOC05_FRONT)
    assert len(lens.lens_one_line.split()) == 40
    assert len(lens.domain.split()) <= 12
    assert "elderly" in lens.population.lower() or "70" in lens.population
    assert "MIE" in lens.comparison
    assert "overall complications" in lens.outcomes
    assert "FRONT MATTER" in backend.prompts[0]
    assert "Do not invent" in backend.prompts[0]
