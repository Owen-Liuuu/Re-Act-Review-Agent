"""Tests for the DKB Resolver — status semantics + read-only audit mode."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.dkb import FieldResolver, KnowledgeAgent, KeywordRetriever, KnowledgeBase
from react_review.llm.base import LLMBackend

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"


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


def _kb() -> KnowledgeBase:
    return KnowledgeBase.from_json(SEED)


def _agent(kb, payload):
    return KnowledgeAgent(_Stub(payload), KeywordRetriever(kb))


@pytest.mark.asyncio
async def test_deterministic_hit_is_authoritative():
    kb = _kb()
    r = FieldResolver(kb)
    out = await r.resolve("BMI")
    assert out.field_type == "bmi" and out.status == "authoritative"
    assert out.source == "deterministic" and out.provisional is False


@pytest.mark.asyncio
async def test_modality_disambiguation_through_resolver():
    r = FieldResolver(_kb())
    out = await r.resolve("EFT/ EAT", modality="Coronary CT angiography (CT)")
    assert out.field_type == "eat_volume" and out.status == "authoritative"


@pytest.mark.asyncio
async def test_miss_without_agent_is_unresolved():
    out = await FieldResolver(_kb()).resolve("Wongabongo score")
    assert out.resolved is False and out.status == "unresolved" and out.field_type is None


@pytest.mark.asyncio
async def test_miss_with_agent_is_candidate_and_proposed_not_written_back():
    kb = _kb()
    payload = {"field_type": "hba1c", "concept": "glycated haemoglobin",
               "value_type": "numeric", "default_unit": "%", "is_new": True,
               "grounded_on": [], "confidence": 0.6}
    r = FieldResolver(kb, _agent(kb, payload), write_back=False)   # audit mode
    out = await r.resolve("HbA1c", unit="%")
    assert out.field_type == "hba1c" and out.status == "candidate" and out.provisional is True
    assert out.source == "retrieval_llm"
    # audit mode: KB is NOT mutated; the candidate is collected as a proposal
    assert "hba1c" not in kb.entries
    assert [e.field_type for e in r.proposals] == ["hba1c"]


@pytest.mark.asyncio
async def test_write_back_mode_merges_into_kb():
    kb = _kb()
    payload = {"field_type": "hba1c", "concept": "x", "is_new": True, "confidence": 0.6}
    r = FieldResolver(kb, _agent(kb, payload), write_back=True)    # developer/learn mode
    await r.resolve("HbA1c")
    assert "hba1c" in kb.entries and kb.entries["hba1c"].status == "provisional"


@pytest.mark.asyncio
async def test_cache_dedupes_without_second_llm_call():
    kb = _kb()
    agent = _agent(kb, {"field_type": "hba1c", "is_new": True, "confidence": 0.5})
    r = FieldResolver(kb, agent, write_back=False)
    await r.resolve("HbA1c")
    await r.resolve("HbA1c")                # same name → cache, no 2nd LLM call
    assert agent._backend.calls == 1
