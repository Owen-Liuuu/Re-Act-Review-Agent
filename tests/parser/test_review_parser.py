"""Tests for the 2-stage ReviewParser (stub backend, monkeypatched PDF text)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.dkb import FieldResolver, KnowledgeBase
from react_review.llm.base import LLMBackend
from react_review.parser.review_parser import ReviewParser, _study_slug

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"


class QueueBackend(LLMBackend):
    def __init__(self, responses: list) -> None:
        super().__init__()
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]

    @property
    def model_id(self) -> str:
        return "queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        return self._responses.pop(0) if self._responses else "{}"


def _resolver() -> FieldResolver:
    # backend=None → deterministic KB resolution only; unknown names → unresolved.
    return FieldResolver(KnowledgeBase.from_json(SEED))


def test_study_slug():
    assert _study_slug("Ahmad et al. [2022]") == "ahmad_2022"
    assert _study_slug("de Gonzalo-Calvo et al. (2018)") == "de_2018"


def _capture(rows, columns=("Study", "N", "EFT/ EAT")) -> dict:
    """A Stage-1 capture payload: the table verbatim, in its own shape."""
    return {"research_context": "",
            "tables": [{"table_id": "table_1", "caption": "Table 1",
                        "header_rows": [list(columns)], "rows": rows,
                        "row_axis_columns": ["Study"]}]}


def _cell(row, col, study, header, value, unit="", cohort="") -> dict:
    return {"row": row, "col": col, "row_key": {"Study": study},
            "column_header": header, "cohort_label": cohort,
            "timepoint_label": "", "value": value, "unit": unit}


@pytest.mark.asyncio
async def test_parse_produces_normalized_long_items(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text", lambda p: "review text"
    )
    backend = QueueBackend([
        _capture([["Ahmad et al. [2022]", "100", "6.60 ± 0.71"]]),
        {"rows": [
            _cell(0, 2, "Ahmad et al. [2022]", "EFT/ EAT", "6.60 ± 0.71", "mm", "T1DM"),
            _cell(0, 2, "Ahmad et al. [2022]", "EFT/ EAT", "3.83 ± 0.35", "mm", "Control"),
            _cell(0, 1, "Ahmad et al. [2022]", "N", "100"),
            # unknown column → KEPT as unresolved, never dropped
            _cell(0, 3, "Ahmad et al. [2022]", "Some Novel Column", "5", "", "T1DM"),
            # written placeholder → KEPT with a reason (it may mean "not reached")
            _cell(0, 4, "Ahmad et al. [2022]", "Age", "N/A", "years", "T1DM"),
            # an EMPTY cell is layout, not a statement → skipped
            _cell(0, 5, "Ahmad et al. [2022]", "Country", "", "", "Control"),
        ]},
        {"studies": []},
    ])
    parser = ReviewParser(backend, _resolver())
    parsed = await parser.parse("dummy.pdf", research_context="EAT in T1DM")

    got = {(i.study_id, i.group, i.field_type, i.value, i.unit) for i in parsed.items}
    assert got == {
        ("ahmad_2022", "t1dm", "eat_thickness", "6.60 ± 0.71", "mm"),
        ("ahmad_2022", "control", "eat_thickness", "3.83 ± 0.35", "mm"),
        ("ahmad_2022", "-", "sample_size", "100", ""),          # study-level → group "-"
        ("ahmad_2022", "t1dm", "", "5", ""),                    # unknown → KEPT, field_type null
        ("ahmad_2022", "t1dm", "", None, "years"),              # placeholder → KEPT, value null
    }
    assert len(parsed.items) == 5
    placeholder = next(i for i in parsed.items if i.value is None)
    assert placeholder.reasons[0].code == "placeholder_cell"
    assert "'N/A'" in placeholder.reasons[0].message

    # cell-level provenance back to the captured table
    eat = next(i for i in parsed.items if i.field_type == "eat_thickness")
    assert eat.table_id == "table_1" and eat.cell_ref == (0, 2)
    assert eat.cohort_label == "T1DM"          # the review's OWN word is preserved

    assert parsed.record.agent == "parser"
    assert [s.tool for s in parsed.record.steps] == [
        "llm:table_capture", "llm:unpivot", "llm:stage_refs"]
    assert [t.table_id for t in parsed.tables.tables] == ["table_1"]


@pytest.mark.asyncio
async def test_study_level_field_collapses_to_one_row(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "t")
    # the review repeats N in BOTH cohort rows; it must collapse to one study-level
    # row (group "-"), while a genuine per-cohort field stays split.
    backend = QueueBackend([
        _capture([["Ahmad et al. [2022]", "100", "6.6"]], ("Study", "N", "EFT/ EAT")),
        {"rows": [
            _cell(0, 1, "Ahmad et al. [2022]", "N", "100", "", "T1DM"),
            _cell(0, 1, "Ahmad et al. [2022]", "N", "100", "", "Control"),
            _cell(0, 2, "Ahmad et al. [2022]", "EFT/ EAT", "6.6", "mm", "T1DM"),
            _cell(0, 2, "Ahmad et al. [2022]", "EFT/ EAT", "3.8", "mm", "Control"),
        ]},
        {"studies": []},
    ])
    parsed = await ReviewParser(backend, _resolver()).parse("d.pdf")
    ss = [i for i in parsed.items if i.field_type == "sample_size"]
    assert len(ss) == 1 and ss[0].group == "-"                      # collapsed
    eat_groups = {i.group for i in parsed.items if i.field_type == "eat_thickness"}
    assert eat_groups == {"t1dm", "control"}                        # cohort-level stays split


@pytest.mark.asyncio
async def test_parse_extracts_research_context_and_dois(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text",
        lambda p: "body text …\n\nReferences\n1. Ahmad A. 2022. J Cardiol. doi:10.1/x\n"
                  "2. Aslan B. 2015. Echocardiography.",
    )
    capture = _capture([["Ahmad et al. [2022]", "100"]], ("Study", "N"))
    capture["research_context"] = "epicardial adipose tissue in T1DM vs healthy controls"
    backend = QueueBackend([
        capture,
        {"rows": [_cell(0, 1, "Ahmad et al. [2022]", "N", "100", "", "T1DM")]},
        {"studies": [
            {"citation": "Ahmad A et al. 2022. J Cardiol.", "doi": "https://doi.org/10.1/X"},
            {"citation": "Aslan B et al. 2015. Echocardiography.", "doi": ""}]},
    ])
    parsed = await ReviewParser(backend, _resolver()).parse("d.pdf", research_context="fallback")

    # LLM-extracted research context wins over the passed-in fallback
    assert parsed.research_context == "epicardial adipose tissue in T1DM vs healthy controls"
    # DOIs normalized; study_id slugged from the citation; empty DOI kept as ""
    assert [(s.study_id, s.doi) for s in parsed.studies] == [
        ("ahmad_2022", "10.1/x"), ("aslan_2015", "")]
    assert parsed.record.steps[-1].tool == "llm:stage_refs"


@pytest.mark.asyncio
async def test_research_context_falls_back_to_arg_when_not_extracted(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "t")
    backend = QueueBackend([
        {"tables": []},                      # nothing captured, no research_context
        {"studies": []},
    ])
    parsed = await ReviewParser(backend, _resolver()).parse("d.pdf", research_context="my ctx")
    assert parsed.research_context == "my ctx"
    assert parsed.studies == []


@pytest.mark.asyncio
async def test_parse_survives_stage_failure(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text", lambda p: "text"
    )
    # both stages return non-JSON → empty structure/rows → no items, no crash
    parser = ReviewParser(QueueBackend(["oops", "also oops"]), _resolver())
    parsed = await parser.parse("dummy.pdf")
    assert parsed.items == []
    assert parsed.record.status == "finished"
