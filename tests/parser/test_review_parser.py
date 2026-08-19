"""Tests for the 2-stage ReviewParser (stub backend, monkeypatched PDF text)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.checklist import Checklist, ChecklistItem
from react_review.dkb import FieldResolver, KnowledgeBase, load_runtime_knowledge
from react_review.core.exceptions import RunStopped
from react_review.hitl import Decision, ScriptedCheckpoint, StepReporter, StepStage, SubjectKind
from react_review.llm.base import LLMBackend
from react_review.parser.review_parser import ReviewParser, _study_slug
from react_review.schemas.table import CapturedTable

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


def _parser(backend, resolver=None, **kwargs):
    """Existing unpivot tests pin frozen v1 so DEFAULT=v3 does not change their queue."""
    kwargs.setdefault("table_capture_prompt_profile", "table_capture_v1")
    return ReviewParser(backend, resolver or _resolver(), **kwargs)


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
        "react_review.parser.review_parser._pdf_text",
        lambda p: "Ahmad 2022 Egypt T1DM 6.60 review text",
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
    parser = _parser(
        backend, _resolver(), reporter=StepReporter("parser-test", gate=gate))
    parsed = await parser.parse("dummy.pdf", research_context="EAT in T1DM")

    got = {(i.study_id, i.group, i.field_type, i.value, i.unit) for i in parsed.items}
    assert got == {
        ("ahmad_2022", "t1dm", "eat_thickness", "6.60 ± 0.71", "mm"),
        ("ahmad_2022", "control", "eat_thickness", "3.83 ± 0.35", "mm"),
        ("ahmad_2022", "-", "sample_size", "100", ""),  # study-level → group "-"
        ("ahmad_2022", "t1dm", "", "5", ""),            # unknown → KEPT, field_type null
        ("ahmad_2022", "t1dm", "", None, "years"),      # placeholder → KEPT, value null
    }
    assert {i.study_label_raw for i in parsed.items} == {"Ahmad et al. [2022]"}
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
    review_pdf = str(Path("dummy.pdf").resolve())
    for stage in (
        StepStage.COHORT_REGISTRY,
        StepStage.LONG_FORMAT_ROWS,
        StepStage.REFERENCE_COVERAGE,
    ):
        event = next(e for e in gate.seen if e.stage is stage)
        assert event.subject == review_pdf
        assert event.subject_kind is SubjectKind.REVIEW_PDF
    assert all(event.subject for event in gate.seen)
    assert all(event.subject_kind is not None for event in gate.seen)


@pytest.mark.asyncio
async def test_runtime_ontology_is_visible_in_field_gate_and_parsed_contract(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "Ahmad T1DM 6.60 t")
    backend = QueueBackend([
        _capture([["Ahmad et al. [2022]", "100"]], ("Study", "N")),
        {"rows": [_cell(0, 1, "Ahmad et al. [2022]", "N", "100")]},
        {"studies": []},
    ])
    gate = ScriptedCheckpoint()
    kb = load_runtime_knowledge(SEED, ONTOLOGY)
    parsed = await _parser(
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
    parsed = await _parser(
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
    parser = _parser(
        backend, _resolver(), checklist=checklist,
        reporter=StepReporter("checklist-stop", gate=gate))

    with pytest.raises(RunStopped):
        await parser.parse("d.pdf")
    assert gate.seen[-1].stage is StepStage.CHECKLIST_REVIEW
    assert StepStage.LONG_FORMAT_ROWS not in [event.stage for event in gate.seen]


@pytest.mark.asyncio
async def test_study_level_field_collapses_to_one_row(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "Ahmad T1DM 6.60 t")
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
    parsed = await _parser(backend, _resolver()).parse("d.pdf")
    ss = [i for i in parsed.items if i.field_type == "sample_size"]
    assert len(ss) == 1 and ss[0].group == "-"                      # collapsed
    eat_groups = {i.group for i in parsed.items if i.field_type == "eat_thickness"}
    assert eat_groups == {"t1dm", "control"}                        # cohort-level stays split


@pytest.mark.asyncio
async def test_stopping_at_field_resolution_never_emits_derived_long_rows(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "Ahmad T1DM 6.60 t")
    backend = QueueBackend([
        _capture([["Ahmad et al. [2022]", "100"]], ("Study", "N")),
        {"rows": [_cell(0, 1, "Ahmad et al. [2022]", "N", "100")]},
    ])
    gate = ScriptedCheckpoint([
        Decision.CONTINUE, Decision.CONTINUE, Decision.CONTINUE, Decision.STOP])
    parser = _parser(
        backend, _resolver(), reporter=StepReporter("stop-test", gate=gate))

    with pytest.raises(RunStopped):
        await parser.parse("d.pdf")

    assert gate.seen[-1].stage is StepStage.FIELD_RESOLUTION
    assert StepStage.LONG_FORMAT_ROWS not in [event.stage for event in gate.seen]


@pytest.mark.asyncio
async def test_unknown_column_has_one_resolution_record_for_all_numeric_rows(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "Ahmad T1DM 6.60 t")
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
    parsed = await _parser(backend, _resolver()).parse("d.pdf")

    assert len(parsed.items) == 2
    assert len(parsed.field_resolutions) == 1
    assert len(parsed.field_resolutions[0].affected_cells) == 2
    assert {item.resolution_key for item in parsed.items} == {
        parsed.field_resolutions[0].resolution_key}


@pytest.mark.asyncio
async def test_parse_extracts_research_context_and_dois(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text",
        lambda p: "body text T1DM …\n\nReferences\n1. Ahmad A. 2022. J Cardiol. doi:10.1/x\n"
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
    parsed = await _parser(backend, _resolver()).parse("d.pdf", research_context="fallback")

    # LLM-extracted research context wins over the passed-in fallback
    assert parsed.research_context == "epicardial adipose tissue in T1DM vs healthy controls"
    # DOIs normalized; study_id slugged from the citation; empty DOI kept as ""
    assert [(s.study_id, s.doi) for s in parsed.studies] == [
        ("ahmad_2022", "10.1/x"), ("aslan_2015", "")]
    assert parsed.record.steps[-1].tool == "llm:stage_refs"


@pytest.mark.asyncio
async def test_parser_keeps_only_printed_doi_and_pmid(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text",
        lambda p: "References\n1. Capovilla G. Front Oncol. 2023;13:1104109. PMID: 36726501\n",
    )
    backend = QueueBackend([
        {"tables": []},
        {"studies": [{
            "citation": "Capovilla G. Front Oncol. 2023;13:1104109. PMID: 36726501",
            "doi": "10.3389/fonc.2023.1104109",
        }]},
    ])
    parsed = await _parser(backend, _resolver()).parse("d.pdf")
    assert parsed.studies[0].doi == ""
    assert parsed.studies[0].pmid == "36726501"


@pytest.mark.asyncio
async def test_table_join_key_keeps_printed_words_and_pairs_citations(monkeypatch):
    # Offline doc05 shape: Study cell without a year, Year in the next column,
    # two papers whose first word is Li. Claims slug to the same keys as the
    # citation list; the printed row wording is kept as study_label_raw.
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "Ahmad T1DM 6.60 t")
    backend = QueueBackend([
        {"research_context": "",
         "tables": [
             {"table_id": "table_1", "caption": "Table 1",
              "role": "characteristics",
              "header_rows": [["Study", "Year", "N"]],
              "rows": [["Li J et al.", "2015", "80"],
                       ["Li K et al.", "2025", "90"],
                       ["Capovilla G et al.", "2023", "70"]],
              "row_axis_columns": ["Study"]},
             {"table_id": "table_2", "caption": "Table 2",
              "role": "outcomes",
              "header_rows": [["Outcome", "Studies (n)", "OR (95% CI)"]],
              "rows": [["Overall Complications", "3", "0.40 (0.27-0.60)"],
                       ["Pulmonary Complications", "3", "0.42 (0.28-0.64)"]],
              "row_axis_columns": ["Outcome"]},
         ]},
        {"rows": [
            _cell(0, 2, "Li J et al.", "N", "80"),
            _cell(1, 2, "Li K et al.", "N", "90"),
            _cell(2, 2, "Capovilla G et al.", "N", "70"),
        ]},
        {"studies": [
            {"citation": "Li J, Shen Y, Tan L, et al. Surg Endosc. 2015;29(4):925-930.",
             "doi": ""},
            {"citation": "Capovilla G, Uzun E, Scarton A, et al. Front Oncol. 2023;13:1104109.",
             "doi": ""},
            {"citation": "Li K, Lu S, Li C, et al. Langenbecks Arch Surg. 2025;410(1):311. doi:10.1007/s00423-025-03877-4",
             "doi": "10.1007/s00423-025-03877-4"},
        ]},
    ])
    gate = ScriptedCheckpoint()
    parsed = await _parser(
        backend, _resolver(),
        reporter=StepReporter("doc05-join", gate=gate)).parse("d.pdf")

    assert {i.study_id for i in parsed.items} == {
        "li_2015", "li_2025", "capovilla_2023"}
    assert {i.study_label_raw for i in parsed.items} == {
        "Li J et al. 2015", "Li K et al. 2025", "Capovilla G et al. 2023"}
    assert [t.table_id for t in parsed.tables.tables] == ["table_1", "table_2"]
    long_event = next(e for e in gate.seen if e.stage is StepStage.LONG_FORMAT_ROWS)
    assert any("not included papers" in w for w in long_event.warnings)
    assert "table_2" in long_event.warnings[0]
    by_id = {s.study_id: s for s in parsed.studies}
    assert set(by_id) == {"li_2015", "li_2025", "capovilla_2023"}
    assert by_id["li_2025"].doi == "10.1007/s00423-025-03877-4"
    event = next(e for e in gate.seen if e.stage is StepStage.REFERENCE_COVERAGE)
    assert not event.warnings
    assert "3 match a study in the data table" in event.render_blocks[0]
    assert "0 table study/studies have none" in event.render_blocks[0]
    block = event.render_blocks[0]
    assert "retrieval plan:" in block
    assert "li_2015" in block and "no DOI · no PMID" in block
    assert "plan:title + author + year search" in block
    assert "li_2025" in block and "DOI 10.1007/s00423-025-03877-4" in block
    assert "plan:lookup by DOI" in block


@pytest.mark.asyncio
async def test_research_context_falls_back_to_arg_when_not_extracted(monkeypatch):
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: "Ahmad T1DM 6.60 t")
    backend = QueueBackend([
        {"tables": []},                      # nothing captured, no research_context
        {"studies": []},
    ])
    parsed = await _parser(backend, _resolver()).parse("d.pdf", research_context="my ctx")
    assert parsed.research_context == "my ctx"
    assert parsed.studies == []


@pytest.mark.asyncio
async def test_parse_survives_stage_failure(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text", lambda p: "text"
    )
    # both stages return non-JSON → empty structure/rows → no items, no crash
    parser = _parser(QueueBackend(["oops", "also oops"]), _resolver())
    parsed = await parser.parse("dummy.pdf")
    assert parsed.items == []
    assert parsed.record.status == "finished"


def test_empty_cohort_label_takes_arm_from_column_header():
    """Unpivot left cohort_label empty; the arm is only in 'N MIE'.

    Matching is CohortRegistry.resolve second pass (alias). Combined labels
    such as Total must not count as a header hit — that is the 2b trap when
    forest columns are Events, Total, Events, Total.
    """
    from react_review.normalize.cohorts import build_cohort_registry

    parser = _parser(QueueBackend([]))
    registry = build_cohort_registry(["MIE", "OE"])
    rows = [
        {"column_header": "N MIE", "value": "58", "cohort_label": "",
         "row_key": {"Study": "Li J 2015"}, "table_id": "table_1",
         "row": 0, "col": 1},
        {"column_header": "Country", "value": "China", "cohort_label": "",
         "row_key": {"Study": "Li J 2015"}, "table_id": "table_1",
         "row": 0, "col": 2},
        {"column_header": "Total", "value": "116", "cohort_label": "",
         "row_key": {"Study": "Li J 2015"}, "table_id": "table_1",
         "row": 0, "col": 3},
    ]
    items = parser._postprocess(rows, {}, registry)
    by_header = {i.raw_field_name: i for i in items}
    n_mie = by_header["N MIE"]
    assert n_mie.group == "mie"
    assert n_mie.cohort_status == "alias"
    assert n_mie.cohort_label == ""
    country = by_header["Country"]
    assert country.group == "all"
    assert country.cohort_status == "combined"
    total = by_header["Total"]
    assert total.group == "all"
    assert total.cohort_status == "combined"


def test_table_and_forest_row_labels_share_one_study_id():
    """Printed wording stays on study_label_raw; study_key is applied without taken=."""
    parser = _parser(QueueBackend([]))
    rows = [
        {"column_header": "N", "value": "80",
         "row_key": {"Study": "Li J et al."}, "row_year": "2015",
         "table_id": "table_1", "row": 0, "col": 2},
        {"column_header": "Events", "value": "22",
         "row_key": {"Study": "Li J 2015"}, "row_year": "",
         "table_id": "figure_2", "row": 0, "col": 1,
         "display_kind": "forest_plot", "outcome": "overall complications"},
    ]
    items = parser._postprocess(rows, {})
    assert len(items) == 2
    assert {i.study_id for i in items} == {"li_2015"}
    assert {i.study_label_raw for i in items} == {"Li J et al. 2015", "Li J 2015"}


@pytest.mark.asyncio
async def test_unpivot_does_not_send_summary_row_kinds_to_the_model():
    class PromptQueue(QueueBackend):
        def __init__(self, responses: list) -> None:
            super().__init__(responses)
            self.prompts: list[str] = []

        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.prompts.append(prompt)
            return await super().complete(prompt, seed=seed)

    table = CapturedTable(
        table_id="table_1",
        header_rows=[["Study", "N"]],
        rows=[["Li J 2015", "58"], ["Total", "100"]],
        row_axis_columns=["Study"],
        row_kinds=["study", "summary"],
    )
    backend = PromptQueue([{
        "rows": [{"row": 0, "column_header": "N", "value": "58"}],
    }])
    out = await _parser(backend)._unpivot(table)
    assert backend.prompts
    assert '"Total"' not in backend.prompts[0]
    assert "[0, [\"Li J 2015\", \"58\"]]" in backend.prompts[0]
    assert out[0]["row_key"]["study"] == "Li J 2015"


MIE_EVENTS = "Minimally Invasive Esophagectomy (MIE) Events"
OE_EVENTS = "Open Esophagectomy (OE) Events"


@pytest.mark.asyncio
async def test_forest_count_columns_skip_dkb_and_do_not_mint_arm_concepts():
    parser = _parser(QueueBackend([]))
    called: list[str] = []
    original = parser._resolver.resolve

    async def spy(raw_name, **kwargs):
        called.append(raw_name)
        return await original(raw_name, **kwargs)

    parser._resolver.resolve = spy  # type: ignore[method-assign]
    rows = [
        {"column_header": MIE_EVENTS, "value": "19",
         "display_kind": "forest_plot", "row_key": {"study": "Capovilla 2023"},
         "table_id": "forest_2", "row": 0},
        {"column_header": "Odds ratio", "value": "0.40",
         "display_kind": "forest_plot", "row_key": {"study": "Capovilla 2023"},
         "table_id": "forest_2", "row": 0},
    ]
    by_row, records = await parser._resolve_fields(rows, "esophagectomy")
    assert MIE_EVENTS not in called
    assert "Odds ratio" in called
    assert by_row[0].field_type == "events"
    assert by_row[0].scope == "cohort"
    assert by_row[0].source == "deterministic"
    assert by_row[0].status == "authoritative"
    forest_records = [r for r in records if r.raw_field_name == MIE_EVENTS]
    assert forest_records
    assert forest_records[0].field_type == "events"
    assert not any(a.is_new for rec in forest_records for a in rec.attempts)


def test_cohort_scoped_forest_events_are_kept_across_displays():
    from react_review.dkb import ResolvedField
    from react_review.normalize.cohorts import build_cohort_registry

    parser = _parser(QueueBackend([]))
    registry = build_cohort_registry(["MIE", "OE"])
    rows = [
        {"column_header": MIE_EVENTS, "value": "23", "cohort_label": "",
         "row_key": {"study": "Capovilla 2023"}, "table_id": "forest_1",
         "display_kind": "forest_plot", "row": 0, "col": 1},
        {"column_header": MIE_EVENTS, "value": "19", "cohort_label": "",
         "row_key": {"study": "Capovilla 2023"}, "table_id": "forest_2",
         "display_kind": "forest_plot", "row": 0, "col": 1},
        {"column_header": OE_EVENTS, "value": "64", "cohort_label": "",
         "row_key": {"study": "Capovilla 2023"}, "table_id": "forest_1",
         "display_kind": "forest_plot", "row": 0, "col": 3},
    ]
    resolved = ResolvedField(
        raw_field_name=MIE_EVENTS, field_type="events",
        status="authoritative", scope="cohort", source="deterministic",
    )
    items = parser._postprocess(rows, {0: resolved, 1: resolved, 2: resolved}, registry)
    events = [i for i in items if i.field_type == "events"]
    assert len(events) == 3
    assert {i.table_id for i in events} == {"forest_1", "forest_2"}
    assert {i.group for i in events} <= {"mie", "oe"}
    assert "-" not in {i.group for i in events}
    assert "all" not in {i.group for i in events}


def test_study_scope_conflict_keeps_both_disagreeing_values():
    from react_review.dkb import ResolvedField

    parser = _parser(QueueBackend([]))
    rows = [
        {"column_header": "Year", "value": "2015", "cohort_label": "",
         "row_key": {"Study": "Li J 2015"}, "table_id": "table_1",
         "row": 0, "col": 1},
        {"column_header": "Year", "value": "2016", "cohort_label": "",
         "row_key": {"Study": "Li J 2015"}, "table_id": "forest_1",
         "row": 0, "col": 1},
    ]
    resolved = ResolvedField(
        raw_field_name="Year", field_type="publication_year",
        status="authoritative", scope="study", source="deterministic",
    )
    items = parser._postprocess(rows, {0: resolved, 1: resolved})
    years = [i for i in items if i.field_type == "publication_year"]
    assert [i.value for i in years] == ["2015", "2016"]
    for item in years:
        assert any(r.code == "study_scope_conflict" for r in item.reasons)


def test_study_scope_identical_values_are_still_deduped():
    from react_review.dkb import ResolvedField

    parser = _parser(QueueBackend([]))
    rows = [
        {"column_header": "Year", "value": "2015", "cohort_label": "",
         "row_key": {"Study": "Li J 2015"}, "table_id": "table_1",
         "row": 0, "col": 1},
        {"column_header": "Year", "value": "2015", "cohort_label": "",
         "row_key": {"Study": "Li J 2015"}, "table_id": "forest_1",
         "row": 0, "col": 1},
    ]
    resolved = ResolvedField(
        raw_field_name="Year", field_type="publication_year",
        status="authoritative", scope="study", source="deterministic",
    )
    items = parser._postprocess(rows, {0: resolved, 1: resolved})
    years = [i for i in items if i.field_type == "publication_year"]
    assert [i.value for i in years] == ["2015"]
    assert not any(
        r.code == "study_scope_conflict" for i in years for r in i.reasons)


@pytest.mark.asyncio
async def test_truncated_pdf_and_empty_cohorts_pass_force_gate(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text", lambda p: "x" * 20)
    recorded: list[tuple[StepStage, bool]] = []
    parser = _parser(
        QueueBackend([{"tables": []}, {"studies": []}]), max_chars=5)
    orig = parser._reporter.step

    async def wrapped(stage, **kw):
        recorded.append((stage, bool(kw.get("force_gate", False))))
        return await orig(stage, **kw)

    parser._reporter.step = wrapped  # type: ignore[method-assign]
    await parser.parse("d.pdf")
    by_stage = dict(recorded)
    assert by_stage[StepStage.REVIEW_PDF_LOADED] is True
    assert by_stage[StepStage.COHORT_REGISTRY] is True


@pytest.mark.asyncio
async def test_complete_pdf_does_not_force_gate_loaded_or_known_cohorts(monkeypatch):
    monkeypatch.setattr(
        "react_review.parser.review_parser._pdf_text",
        lambda p: "Ahmad T1DM 6.60 t",
    )
    recorded: list[tuple[StepStage, bool]] = []
    parser = _parser(QueueBackend([
        _capture([["Ahmad et al. [2022]", "100"]], ("Study", "N")),
        {"rows": [_cell(0, 1, "Ahmad et al. [2022]", "N", "100", cohort="T1DM")]},
        {"studies": []},
    ]))
    orig = parser._reporter.step

    async def wrapped(stage, **kw):
        recorded.append((stage, bool(kw.get("force_gate", False))))
        return await orig(stage, **kw)

    parser._reporter.step = wrapped  # type: ignore[method-assign]
    await parser.parse("d.pdf")
    by_stage = dict(recorded)
    assert by_stage[StepStage.REVIEW_PDF_LOADED] is False
    assert by_stage[StepStage.COHORT_REGISTRY] is False


def test_coverage_plan_uses_doi_pmid_not_retrieval_results():
    from react_review.parser.review_parser import ParsedStudy, ReviewParser

    doi = ParsedStudy(study_id="li_2025", doi="10.xxxx/y")
    pmid = ParsedStudy(study_id="x", pmid="12345")
    none = ParsedStudy(study_id="li_2015")
    assert ReviewParser._planned_lookup(doi) == "lookup by DOI"
    assert ReviewParser._planned_lookup(pmid) == "lookup by PMID"
    assert ReviewParser._planned_lookup(none) == "title + author + year search"
    assert ReviewParser._ident_status(none) == "no DOI · no PMID"
    assert ReviewParser._ident_status(doi) == "DOI 10.xxxx/y · no PMID"

