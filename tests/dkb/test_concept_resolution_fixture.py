"""The Phase-5 candidate-path acceptance fixture is executable in CI."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.run_concept_resolution import run_fixture

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.asyncio
async def test_every_new_concept_fixture_case_reaches_its_expected_outcome():
    results = await run_fixture(
        ROOT / "eval" / "fixtures" / "new_concepts.json",
        ROOT / "configs" / "knowledge.seed.json",
    )
    assert results
    assert all(r["ok"] for r in results), {
        r["id"]: r["failures"] for r in results if not r["ok"]
    }
    # This is the property the EAT benchmark cannot establish: the candidate
    # LLM branch really ran, and cache reuse did not erase its attempt trace.
    candidate = next(
        r for r in results if r["id"] == "new_numeric_concept_is_stable_and_recorded")
    assert candidate["llm_calls"] == 2
    assert candidate["attempts"] == 2
    assert candidate["cache_hits"] == 1
