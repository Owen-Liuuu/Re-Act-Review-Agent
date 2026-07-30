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


@pytest.mark.asyncio
async def test_ungrounded_low_confidence_guess_is_rejected_to_unresolved():
    kb = _kb()
    payload = {"field_type": "made_up_thing", "is_new": True,
               "grounded_on": [], "confidence": 0.1}          # blind low-confidence guess
    r = FieldResolver(kb, _agent(kb, payload), write_back=False)
    out = await r.resolve("Wongabongo score")
    assert out.status == "unresolved" and out.field_type is None
    assert r.proposals == []                                  # a rejected guess is not learned


@pytest.mark.asyncio
async def test_unit_kind_conflict_is_rejected_to_unresolved():
    # LLM maps an unknown name to eat_thickness (mm) but the column unit is cm3
    # (a volume) — a concept confusion the deterministic gate catches.
    kb = _kb()
    payload = {"field_type": "eat_thickness", "is_new": False,
               "grounded_on": ["eat_thickness"], "confidence": 0.9}
    r = FieldResolver(kb, _agent(kb, payload), write_back=False)
    out = await r.resolve("mysterious fat measure", unit="cm3")
    assert out.status == "unresolved" and out.field_type is None


@pytest.mark.asyncio
async def test_range_conflict_is_not_cached():
    # A value-dependent rejection must NOT poison a later row with a valid value.
    kb = _kb()
    kb.entries["eat_thickness"].plausible_range = [1.0, 30.0]
    payload = {"field_type": "eat_thickness", "is_new": False,
               "grounded_on": ["eat_thickness"], "confidence": 0.9}
    agent = _agent(kb, payload)
    r = FieldResolver(kb, agent, write_back=False)
    bad = await r.resolve("eft-ish", unit="mm", value="900")   # out of range → rejected
    assert bad.status == "unresolved"
    ok = await r.resolve("eft-ish", unit="mm", value="6.6")    # valid value → re-tried, accepted
    assert ok.field_type == "eat_thickness"
    assert agent._backend.calls == 2                            # not cached, so LLM ran twice
