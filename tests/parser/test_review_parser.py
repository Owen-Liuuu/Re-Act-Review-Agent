"""Tests for the 2-stage ReviewParser (stub backend, monkeypatched PDF text)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.checklist import Checklist, ChecklistItem
from react_review.dkb import FieldResolver, KnowledgeBase, load_runtime_knowledge
from react_review.core.exceptions import RunStopped
from react_review.hitl import Decision, ScriptedCheckpoint, StepReporter, StepStage
from react_review.llm.base import LLMBackend
from react_review.parser.review_parser import ReviewParser, _study_slug

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"
ONTOLOGY = Path(__file__).resolve().parents[2] / "configs" / "ontology"


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
    # Surname particles are kept: a first-word rule made "van den Berg 2020" and
    # "van Rooij 2020" the same study. See normalize/study_key.py.
    assert _study_slug("de Gonzalo-Calvo et al. (2018)") == "degonzalocalvo_2018"


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
    gate = ScriptedCheckpoint()
    parser = ReviewParser(
        backend, _resolver(), reporter=StepReporter("parser-test", gate=gate))
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
    assert [item.review_data_id for item in parsed.items] == [
        "A_01", "A_02", "A_03", "A_04", "A_05"]
    placeholder = next(i for i in parsed.items if i.value is None)
    assert placeholder.reasons[0].code == "placeholder_cell"
    assert "'N/A'" in placeholder.reasons[0].message

    # cell-level provenance back to the captured table
    eat = next(i for i in parsed.items if i.field_type == "eat_thickness")
    assert eat.table_id == "table_1" and eat.cell_ref == (0, 2)
    assert eat.cohort_label == "T1DM"          # the review's OWN word is preserved

    assert parsed.record.agent == "parser"
    assert [s.tool for s in parsed.record.steps] == [
        "llm:table_capture", "llm:unpivot", "dkb:resolve_fields", "llm:stage_refs"]
    assert [t.table_id for t in parsed.tables.tables] == ["table_1"]

    # Decisions are inspectable before any long row consumes them.  Rows link
    # back to one run-level record instead of duplicating its model trace.
    stages = [event.stage for event in gate.seen]
    assert stages.index(StepStage.FIELD_RESOLUTION) < stages.index(StepStage.LONG_FORMAT_ROWS)
    resolution_keys = {r.resolution_key for r in parsed.field_resolutions}
    assert resolution_keys
    assert all(i.resolution_key in resolution_keys for i in parsed.items if i.value is not None)
    resolution_event = next(
        event for event in gate.seen if event.stage is StepStage.FIELD_RESOLUTION)
    assert resolution_event.payload["resolutions"]
    long_event = next(
        event for event in gate.seen if event.stage is StepStage.LONG_FORMAT_ROWS)
    assert long_event.payload["claim_index"]["A_01"]["cell_ref"] == [0, 1]
    assert "[A_01" in long_event.render_blocks[0]


@pytest.mark.asyncio
async def test_runtime_ontology_is_visible_in_field_gate_and_parsed_contract(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "t")
    backend = QueueBackend([
        _capture([["Ahmad et al. [2022]", "100"]], ("Study", "N")),
        {"rows": [_cell(0, 1, "Ahmad et al. [2022]", "N", "100")]},
        {"studies": []},
    ])
    gate = ScriptedCheckpoint()
    kb = load_runtime_knowledge(SEED, ONTOLOGY)
    parsed = await ReviewParser(
        backend, FieldResolver(kb),
        reporter=StepReporter("ontology-test", gate=gate),
    ).parse("d.pdf")

    assert parsed.knowledge_fingerprint == kb.fingerprint()
    assert parsed.knowledge_concept_count == 12
    assert [record.source for record in parsed.knowledge_imports] == ["ontology:labs"]
    event = next(e for e in gate.seen if e.stage is StepStage.FIELD_RESOLUTION)
    snapshot = event.payload["knowledge_base"]
    assert snapshot["fingerprint"] == parsed.knowledge_fingerprint
    assert snapshot["concept_count"] == 12
    assert snapshot["imports"][0]["source"] == "ontology:labs"
    assert "labs.json" in event.render_blocks[0]


@pytest.mark.asyncio
async def test_checklist_gates_routed_long_rows_and_is_preserved(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text",
        lambda p: "Methods: risk of bias was assessed.\nReferences\nSmith 2020")
    backend = QueueBackend([
        _capture([["Smith 2020", "100"]], ("Study", "N")),
        {"rows": [_cell(0, 1, "Smith 2020", "N", "100")]},
        {"studies": [{"citation": "Smith 2020", "doi": None}]},
    ])
    checklist = Checklist(name="test", source_file="checklist.yaml", sha256="hash", items=[
        ChecklistItem(
            id="risk", question="Risk assessed?", required=True,
            scope="review", where=["review_text"], value_kind="presence",
            aliases=["risk of bias"]),
        ChecklistItem(
            id="sample", question="Sample size per study?", required=True,
            scope="per_study", value_kind="numeric", field_types=["sample_size"]),
    ])
    gate = ScriptedCheckpoint()
    parsed = await ReviewParser(
        backend, _resolver(), checklist=checklist,
        reporter=StepReporter("checklist-test", gate=gate)).parse("d.pdf")

    stages = [event.stage for event in gate.seen]
    assert stages.index(StepStage.FIELD_RESOLUTION) < stages.index(
        StepStage.CHECKLIST_REVIEW)
    assert stages.index(StepStage.CHECKLIST_REVIEW) < stages.index(
        StepStage.LONG_FORMAT_ROWS)
    assert stages.index(StepStage.LONG_FORMAT_ROWS) < stages.index(StepStage.REFERENCE_COVERAGE)
    assert stages.index(StepStage.REFERENCE_COVERAGE) < stages.index(
        StepStage.CHECKLIST_STUDY_COVERAGE)
    assert parsed.checklist is not None and parsed.checklist.gaps == []
    assert [a.status for a in parsed.checklist.assessments] == ["covered", "covered"]
    assert parsed.checklist.completed_passes == ["review", "study_coverage"]
    event = next(e for e in gate.seen if e.stage is StepStage.CHECKLIST_REVIEW)
    assert event.payload["sha256"] == "hash"
    assert event.payload["routed_claims"][0]["field_type"] == "sample_size"
    assert event.payload["routed_claims"][0]["checklist_id"] == "sample"
    assert "1 item(s), 0 required gap(s)" in event.render_blocks[0]
    late = next(
        e for e in gate.seen if e.stage is StepStage.CHECKLIST_STUDY_COVERAGE)
    assert late.payload["approved_study_ids"] == ["smith_2020"]
    assert late.payload["pass_assessments"][0]["checklist_id"] == "sample"
    assert parsed.items[0].origin == "checklist"
    assert parsed.items[0].checklist_id == "sample"
    assert [step.tool for step in parsed.record.steps][-3:] == [
        "checklist:review_coverage", "llm:stage_refs",
        "checklist:study_coverage"]


@pytest.mark.asyncio
async def test_stopping_at_checklist_never_emits_routed_long_rows(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text",
        lambda p: "Methods: risk of bias was assessed.")
    backend = QueueBackend([
        _capture([["Smith 2020", "100"]], ("Study", "N")),
        {"rows": [_cell(0, 1, "Smith 2020", "N", "100")]},
    ])
    checklist = Checklist(name="test", items=[ChecklistItem(
        id="sample", question="Sample size?", required=True,
        scope="per_study", value_kind="numeric", field_types=["sample_size"])])
    gate = ScriptedCheckpoint([
        Decision.CONTINUE, Decision.CONTINUE, Decision.CONTINUE,
        Decision.CONTINUE, Decision.STOP])
    parser = ReviewParser(
        backend, _resolver(), checklist=checklist,
        reporter=StepReporter("checklist-stop", gate=gate))

    with pytest.raises(RunStopped):
        await parser.parse("d.pdf")
    assert gate.seen[-1].stage is StepStage.CHECKLIST_REVIEW
    assert StepStage.LONG_FORMAT_ROWS not in [event.stage for event in gate.seen]


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
async def test_stopping_at_field_resolution_never_emits_derived_long_rows(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "t")
    backend = QueueBackend([
        _capture([["Ahmad et al. [2022]", "100"]], ("Study", "N")),
        {"rows": [_cell(0, 1, "Ahmad et al. [2022]", "N", "100")]},
    ])
    gate = ScriptedCheckpoint([
        Decision.CONTINUE, Decision.CONTINUE, Decision.CONTINUE, Decision.STOP])
    parser = ReviewParser(
        backend, _resolver(), reporter=StepReporter("stop-test", gate=gate))

    with pytest.raises(RunStopped):
        await parser.parse("d.pdf")

    assert gate.seen[-1].stage is StepStage.FIELD_RESOLUTION
    assert StepStage.LONG_FORMAT_ROWS not in [event.stage for event in gate.seen]


@pytest.mark.asyncio
async def test_unknown_column_has_one_resolution_record_for_all_numeric_rows(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "t")
    backend = QueueBackend([
        _capture(
            [["Smith 2020", "7.1"], ["Jones 2021", "8.2"]],
            ("Study", "Novel marker")),
        {"rows": [
            _cell(0, 1, "Smith 2020", "Novel marker", "7.1", "%", "Treatment"),
            _cell(1, 1, "Jones 2021", "Novel marker", "8.2", "%", "Treatment"),
        ]},
        {"studies": []},
    ])
    parsed = await ReviewParser(backend, _resolver()).parse("d.pdf")

    assert len(parsed.items) == 2
    assert len(parsed.field_resolutions) == 1
    assert len(parsed.field_resolutions[0].affected_cells) == 2
    assert {item.resolution_key for item in parsed.items} == {
        parsed.field_resolutions[0].resolution_key}


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
