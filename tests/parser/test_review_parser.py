"""Tests for the 2-stage ReviewParser (stub backend, monkeypatched PDF text)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.llm.base import LLMBackend
from react_review.normalize.vocabulary import Vocabulary
from react_review.parser.review_parser import ReviewParser, _study_slug
from react_review.tools.normalize import NormalizeFieldTool

SEED = Path(__file__).resolve().parents[2] / "configs" / "vocabulary.seed.json"


class QueueBackend(LLMBackend):
    def __init__(self, responses: list) -> None:
        super().__init__()
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]

    @property
    def model_id(self) -> str:
        return "queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        return self._responses.pop(0) if self._responses else "{}"


def _normalize_tool() -> NormalizeFieldTool:
    # backend=None → deterministic vocab resolution only; unknown names raise.
    return NormalizeFieldTool(Vocabulary.from_json(SEED), backend=None)


def test_study_slug():
    assert _study_slug("Ahmad et al. [2022]") == "ahmad_2022"
    assert _study_slug("de Gonzalo-Calvo et al. (2018)") == "de_2018"


@pytest.mark.asyncio
async def test_parse_produces_normalized_long_items(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text", lambda p: "review text"
    )
    backend = QueueBackend([
        {"table_name": "Table 1", "columns": ["Author", "N", "EFT/ EAT"],
         "group_handling": "T1DM row + Control row", "notes": ""},
        {"rows": [
            {"study": "Ahmad et al. [2022]", "group": "T1DM",
             "raw_field_name": "EFT/ EAT", "value": "6.60 ± 0.71", "unit": "mm"},
            {"study": "Ahmad et al. [2022]", "group": "Control",
             "raw_field_name": "EFT/ EAT", "value": "3.83 ± 0.35", "unit": "mm"},
            {"study": "Ahmad et al. [2022]", "group": "",
             "raw_field_name": "N", "value": "100", "unit": ""},
            {"study": "Ahmad et al. [2022]", "group": "T1DM",
             "raw_field_name": "Some Novel Column", "value": "5", "unit": ""},  # unknown → skipped
            {"study": "Ahmad et al. [2022]", "group": "T1DM",
             "raw_field_name": "Age", "value": "N/A", "unit": "years"},          # placeholder → skipped
        ]},
    ])
    parser = ReviewParser(backend, _normalize_tool())
    parsed = await parser.parse("dummy.pdf", research_context="EAT in T1DM")

    got = {(i.study_id, i.group, i.field_type, i.value, i.unit) for i in parsed.items}
    assert got == {
        ("ahmad_2022", "t1dm", "eat_thickness", "6.60 ± 0.71", "mm"),
        ("ahmad_2022", "control", "eat_thickness", "3.83 ± 0.35", "mm"),
        ("ahmad_2022", "all", "sample_size", "100", ""),
    }
    # unknown field + placeholder value were dropped
    assert len(parsed.items) == 3
    assert parsed.record.agent == "parser"
    assert [s.tool for s in parsed.record.steps] == ["llm:stage1_structure", "llm:stage2_unpivot"]


@pytest.mark.asyncio
async def test_parse_survives_stage_failure(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text", lambda p: "text"
    )
    # both stages return non-JSON → empty structure/rows → no items, no crash
    parser = ReviewParser(QueueBackend(["oops", "also oops"]), _normalize_tool())
    parsed = await parser.parse("dummy.pdf")
    assert parsed.items == []
    assert parsed.record.status == "finished"
