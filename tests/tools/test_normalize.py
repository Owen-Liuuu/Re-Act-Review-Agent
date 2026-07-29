"""Tests for the Tier-2 vocabulary and the normalize_field tool.

Deterministic paths (cache / vocabulary + unit disambiguation) are tested
without an LLM; the LLM fallback is tested with a tiny stub backend so no
network is needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.dkb import KnowledgeBase
from react_review.llm.base import LLMBackend
from react_review.tools.normalize import NormalizeFieldTool
from react_review.tools.models import NormalizeInput

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"


@pytest.fixture
def vocab() -> KnowledgeBase:
    return KnowledgeBase.from_json(SEED)


# --- Vocabulary (deterministic resolution) ---

@pytest.mark.parametrize(
    "raw, unit, expected",
    [
        ("N", "", "sample_size"),
        ("Country", "", "country"),
        ("Measurement tool", "", "measurement_tool"),
        ("Overall quality", "", "overall_quality"),
        ("Age", "", "age"),
        ("BMI Kg/m2", "", "bmi"),
        ("EFT/ EAT", "mm", "eat_thickness"),      # unit disambiguates
        ("EFT/ EAT", "cm3", "eat_volume"),        # unit disambiguates
        ("EAT thickness average", "", "eat_thickness"),
    ],
)
def test_vocabulary_resolves_benchmark_names(vocab, raw, unit, expected):
    assert vocab.resolve(raw, unit) == expected


def test_ambiguous_without_unit_returns_none(vocab):
    # "EFT/ EAT" maps to both eat_thickness and eat_volume; no unit -> ambiguous.
    assert vocab.resolve("EFT/ EAT", "") is None


def test_unknown_name_returns_none(vocab):
    assert vocab.resolve("Wongabongo score", "") is None


def test_vocabulary_json_round_trip(tmp_path, vocab):
    p = tmp_path / "v.json"
    vocab.save(p)
    reloaded = KnowledgeBase.from_json(p)
    assert reloaded.field_types() == vocab.field_types()


# --- normalize_field tool (cascade) ---

class _StubBackend(LLMBackend):
    """Returns a fixed JSON mapping for the LLM-fallback path."""

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


@pytest.mark.asyncio
async def test_tool_resolves_via_vocabulary_without_llm(vocab):
    backend = _StubBackend({})
    tool = NormalizeFieldTool(vocab, backend)
    out = await tool.run(NormalizeInput(raw_field_name="N"))
    assert out.field_type == "sample_size"
    assert out.source == "vocabulary"
    assert backend.calls == 0  # LLM never touched


@pytest.mark.asyncio
async def test_tool_caches_second_call(vocab):
    tool = NormalizeFieldTool(vocab, _StubBackend({}))
    first = await tool.run(NormalizeInput(raw_field_name="Age", research_context="ctx"))
    second = await tool.run(NormalizeInput(raw_field_name="Age", research_context="ctx"))
    assert first.source == "vocabulary"
    assert second.source == "cache"


@pytest.mark.asyncio
async def test_tool_llm_fallback_extends_vocabulary(vocab):
    backend = _StubBackend({
        "field_type": "hba1c",
        "concept": "glycated haemoglobin",
        "value_type": "numeric",
        "default_unit": "%",
        "is_new": True,
    })
    tool = NormalizeFieldTool(vocab, backend)
    out = await tool.run(NormalizeInput(raw_field_name="HbA1c", unit="%"))
    assert out.field_type == "hba1c"
    assert out.source == "llm"
    assert out.is_new is True
    assert backend.calls == 1
    # The new concept is now in the vocabulary for next time.
    assert "hba1c" in vocab.entries


@pytest.mark.asyncio
async def test_tool_without_backend_raises_on_unknown(vocab):
    tool = NormalizeFieldTool(vocab, backend=None)
    with pytest.raises(ValueError, match="no LLM backend"):
        await tool.run(NormalizeInput(raw_field_name="Totally novel field"))


# --- benchmark grade: the seed vocab must map the review's Table-1 headers ---

@pytest.mark.asyncio
async def test_normalize_grades_benchmark_table1_columns(vocab):
    """normalize_field must resolve every Table-1 column header in the benchmark
    to the field_type the ground truth assigned — deterministically (no LLM)."""
    import csv

    bench = Path(__file__).resolve().parents[2] / "eval" / "benchmark" / "review_ground_truth.csv"
    with bench.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["source_location"] == "Table 1"]

    tool = NormalizeFieldTool(vocab, backend=None)  # deterministic only
    wrong = []
    for r in rows:
        out = await tool.run(NormalizeInput(
            raw_field_name=r["raw_field_name"], unit=r["unit"],
            research_context="EAT in type 1 diabetes vs controls",
        ))
        if out.field_type != r["field_type"]:
            wrong.append((r["raw_field_name"], r["unit"], r["field_type"], out.field_type))
    assert not wrong, f"mis-normalised Table-1 columns: {wrong}"
