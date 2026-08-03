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


class _SeedStub(LLMBackend):
    def __init__(self, payloads: dict[int, dict]) -> None:
        super().__init__()
        self._payloads = payloads
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "seed-stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return json.dumps(self._payloads[seed])


def _kb() -> KnowledgeBase:
    return KnowledgeBase.from_json(SEED)


def _agent(kb, payload):
    return KnowledgeAgent(_Stub(payload), KeywordRetriever(kb))


def _seed_agent(kb, payloads):
    return KnowledgeAgent(_SeedStub(payloads), KeywordRetriever(kb))


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
    assert out.reasons[0].code == "concept_unresolved"


@pytest.mark.asyncio
async def test_failed_agent_call_keeps_attempt_hash_and_reason():
    kb = _kb()
    r = FieldResolver(kb, _agent(kb, {}), write_back=False)
    out = await r.resolve("Wongabongo score")
    assert out.status == "unresolved"
    assert out.reasons[0].code == "concept_resolution_exception"
    assert len(out.attempts) == 3
    assert out.attempts[0].prompt_sha256
    assert out.attempts[0].response_sha256
    assert "no field_type" in out.attempts[0].error


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
async def test_cache_does_not_leak_one_rows_modality_to_the_next():
    # "EFT/ EAT" is a thickness in an echo study and a volume in a CT one. The
    # per-run cache must key on the modality too, or every later row silently
    # inherits the first row's answer.
    r = FieldResolver(_kb())
    echo = await r.resolve("EFT/ EAT", modality="Echocardiography (Echo)")
    ct = await r.resolve("EFT/ EAT", modality="Coronary computed tomography (CT)")
    assert echo.field_type == "eat_thickness"
    assert ct.field_type == "eat_volume"


@pytest.mark.asyncio
async def test_cache_dedupes_without_second_llm_call():
    kb = _kb()
    agent = _agent(kb, {"field_type": "hba1c", "is_new": True, "confidence": 0.5})
    r = FieldResolver(kb, agent, write_back=False)
    first = await r.resolve("HbA1c")
    second = await r.resolve("HbA1c")
    assert agent._backend.calls == 2
    assert second.resolution_key == first.resolution_key
    assert second.source == "retrieval_llm"
    assert second.checks == first.checks
    assert second.attempts == first.attempts
    assert r.records[0].cache_hits == 1


@pytest.mark.asyncio
async def test_candidate_key_is_concept_level_not_one_key_per_row_context():
    kb = _kb()
    payload = {
        "field_type": "novel_marker", "concept": "novel marker",
        "value_type": "numeric", "default_unit": "%", "scope": "cohort",
        "is_new": True, "grounded_on": [], "confidence": 1.0,
    }
    agent = _agent(kb, payload)
    resolver = FieldResolver(kb, agent, write_back=False)
    first = await resolver.resolve(
        "Novel marker", unit="%", modality="Smith 2020 Echo 7.1", value="7.1")
    second = await resolver.resolve(
        "Novel marker", unit="%", modality="Jones 2021 CT 8.2", value="8.2")

    assert first.resolution_key == second.resolution_key
    assert agent._backend.calls == 2                 # two seed samples total, not per row
    assert resolver.records[0].cache_hits == 1


@pytest.mark.asyncio
async def test_low_confidence_does_not_reject_a_stable_self_consistent_candidate():
    kb = _kb()
    payload = {"field_type": "made_up_thing", "is_new": True,
               "grounded_on": [], "confidence": 0.1}          # blind low-confidence guess
    r = FieldResolver(kb, _agent(kb, payload), write_back=False)
    out = await r.resolve("Wongabongo score")
    cached = await r.resolve("Wongabongo score")
    assert out.status == "candidate" and out.field_type == "made_up_thing"
    assert [p.field_type for p in r.proposals] == ["made_up_thing"]
    assert r._agent._backend.calls == 2
    assert cached.reasons == out.reasons
    assert cached.attempts == out.attempts


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
    assert agent._backend.calls == 4                            # two samples per resolution


@pytest.mark.asyncio
async def test_new_concept_name_drift_is_structurally_stable():
    kb = _kb()
    common = {
        "concept": "EAT area indexed to body surface area", "value_type": "numeric",
        "default_unit": "cm2/m2", "scope": "cohort", "is_new": True,
        "grounded_on": [], "confidence": 1.0,
    }
    agent = _seed_agent(kb, {
        42: {**common, "field_type": "eat_area_bsa_index"},
        43: {**common, "field_type": "eat_area_to_bsa_ratio"},
    })
    resolver = FieldResolver(kb, agent, write_back=False)
    out = await resolver.resolve(
        "EAT/BSA index", unit="cm2/m2", value="12.4")

    assert out.status == "candidate"              # stable never means authoritative
    assert out.field_type == "eat_area_bsa_index"  # deterministic canonical spelling
    assert out.stability == "stable" and out.consensus_count == 2
    assert out.candidate_names == ["eat_area_bsa_index", "eat_area_to_bsa_ratio"]
    assert "eat_area_to_bsa_ratio" in out.proposal["synonyms"]
    assert agent._backend.calls == 2


@pytest.mark.asyncio
async def test_three_structurally_different_answers_are_unstable():
    kb = _kb()
    base = {"concept": "x", "is_new": True, "grounded_on": [], "confidence": 1.0}
    agent = _seed_agent(kb, {
        42: {**base, "field_type": "x_numeric", "value_type": "numeric",
             "default_unit": "%", "scope": "cohort"},
        43: {**base, "field_type": "x_text", "value_type": "text",
             "default_unit": "", "scope": "cohort"},
        44: {**base, "field_type": "x_study", "value_type": "numeric",
             "default_unit": "%", "scope": "study"},
    })
    resolver = FieldResolver(kb, agent, write_back=False)
    out = await resolver.resolve("X", unit="%", value="7")

    assert out.status == "unresolved" and out.field_type is None
    assert out.stability == "unstable" and len(out.attempts) == 3
    assert out.checks["resampling_consistency"] is False
    assert out.reasons[0].code == "candidate_unstable"
    assert resolver.proposals == []
    assert agent._backend.calls == 3
