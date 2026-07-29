"""Tests for DKB retrieval + the grounded classification agent (DKB-2a)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.dkb import KeywordRetriever, KnowledgeAgent, KnowledgeBase
from react_review.llm.base import LLMBackend

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_json(SEED)


class _Stub(LLMBackend):
    def __init__(self, payload: dict) -> None:
        super().__init__()
        self._payload = payload
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return json.dumps(self._payload)


def test_keyword_retriever_ranks_relevant_first(kb):
    assert KeywordRetriever(kb).retrieve("body mass index")[0].field_type == "bmi"


def test_keyword_retriever_falls_back_to_all(kb):
    # nothing overlaps → the agent still gets the whole (small) KB to choose from
    assert len(KeywordRetriever(kb).retrieve("zzz nonsense", k=50)) == len(kb.entries)


@pytest.mark.asyncio
async def test_agent_picks_existing_candidate(kb):
    backend = _Stub({"field_type": "bmi", "is_new": False,
                     "grounded_on": ["bmi"], "confidence": 0.9})
    result = await KnowledgeAgent(backend, KeywordRetriever(kb)).classify("body mass index")
    assert result.field_type == "bmi"
    assert result.is_new is False and result.entry is None
    assert result.grounded_on == ["bmi"]


@pytest.mark.asyncio
async def test_agent_proposes_new_as_provisional(kb):
    backend = _Stub({"field_type": "hba1c", "concept": "glycated haemoglobin",
                     "value_type": "numeric", "default_unit": "%", "is_new": True,
                     "grounded_on": [], "confidence": 0.7})
    result = await KnowledgeAgent(backend, KeywordRetriever(kb)).classify("HbA1c", unit="%")
    assert result.field_type == "hba1c" and result.is_new is True
    assert result.entry is not None
    assert result.entry.status == "provisional"        # never silently authoritative
    assert result.entry.provenance.source == "llm" and result.entry.provenance.confidence == 0.7
