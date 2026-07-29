"""Tests for the DKB knowledge base — deterministic resolution + scope (DKB-1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.dkb import KnowledgeBase

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_json(SEED)


def test_load_seed(kb):
    assert "eat_thickness" in kb.entries and "subgroup_n" in kb.entries
    assert kb.entries["sample_size"].scope == "study"


def test_resolve_single_candidate(kb):
    assert kb.resolve("BMI") == "bmi"
    assert kb.resolve("N") == "sample_size"        # bare "N" is sample_size only


def test_resolve_by_unit(kb):
    assert kb.resolve("EFT/ EAT", unit="mm") == "eat_thickness"
    assert kb.resolve("EFT/ EAT", unit="cm3") == "eat_volume"


def test_resolve_by_modality(kb):
    # the A2 fix as DATA: modality disambiguates EAT thickness vs volume
    assert kb.resolve("EFT/ EAT",
                      modality="Coronary computed tomography angiography (CT)") == "eat_volume"
    assert kb.resolve("EFT/ EAT",
                      modality="Transthoracic echocardiography (Echo)") == "eat_thickness"


def test_resolve_ambiguous_is_none(kb):
    assert kb.resolve("EFT/ EAT") is None          # no unit, no modality → LLM fallback
    assert kb.resolve("totally unknown field") is None


def test_scope_of(kb):
    assert kb.scope_of("country") == "study"
    assert kb.scope_of("sample_size") == "study"
    assert kb.scope_of("age") == "cohort"
    assert kb.scope_of("unknown_field") == "cohort"   # default


def test_provenance_and_status_defaults(kb):
    e = kb.entries["eat_thickness"]
    assert e.provenance.source == "curated" and e.status == "authoritative"


def test_save_round_trip(kb, tmp_path):
    p = tmp_path / "kb.json"
    kb.save(p)
    kb2 = KnowledgeBase.from_json(p)
    assert kb2.entries["eat_volume"].disambiguation == kb.entries["eat_volume"].disambiguation
    assert kb2.scope_of("sample_size") == "study"
