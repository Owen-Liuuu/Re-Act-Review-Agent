"""Review Extraction pipeline: selected capture, mock OCR, non-source cells out."""
from __future__ import annotations

import json

import pytest

from react_review.core.exceptions import RunStopped
from react_review.hitl import Decision, ScriptedCheckpoint, StepReporter, StepStage
from react_review.llm.base import LLMBackend
from react_review.parser.review_extraction import ReviewExtraction
from react_review.parser.review_extraction.origin import drop_non_source, fields_for_cell
from react_review.schemas.table import CapturedTable
from react_review.tools.models import ForestOcrResult

DOC = """
Minimally invasive esophagectomy in elderly patients

Abstract
SECRET_ABSTRACT_TOKEN Elderly patients with resectable ESCC. MIE versus OE
for postoperative complications.

Introduction
Background of esophageal surgery.

Results
Table 1 Characteristics of the included studies. Study Year N.
Li J et al. 2015 80
Figure 3.3.1 Forest plot of overall complications
Table 2 GRADE summary of findings. Outcome OR.

Discussion
Pooled GRADE is not an included paper.

References
1. Li J 2015
"""

TABLE1 = {
    "table_id": "table_1",
    "caption": "Table 1 Characteristics of the included studies",
    "role": "characteristics",
    "header_rows": [["Study", "Year", "N"]],
    "rows": [["Li J et al.", "2015", "80"]],
    "row_axis_columns": ["Study"],
}

FOREST = CapturedTable(
    table_id="fig_3_3_1",
    caption="Figure 3.3.1 Forest plot of overall complications",
    header_rows=[["Study or Subgroup", "Events", "Total", "Odds ratio", "Weight"]],
    rows=[["Li J 2015", "23", "58", "0.40", "12%"]],
    row_axis_columns=["Study or Subgroup"],
    display_kind="forest_plot",
    capture_method="figure_ocr",
)


class RecordingQueue(LLMBackend):
    def __init__(self, responses: list) -> None:
        super().__init__()
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else "{}"


class FakeForest:
    async def run(self, payload):
        table = FOREST.model_copy(deep=True)
        table.table_id = payload.figure_id or table.table_id
        table.caption = payload.caption or table.caption
        return ForestOcrResult(table=table)


def _backend() -> RecordingQueue:
    return RecordingQueue([
        {"lens_one_line": "elderly ESCC, MIE vs OE, complications",
         "domain": "esophageal cancer surgery",
         "population": "elderly ESCC",
         "comparison": "MIE vs OE",
         "outcomes": ["overall complications"]},
        {"displays": [
            {"display_id": "table_1", "kind": "pdf_table",
             "caption": "Table 1 Characteristics of the included studies",
             "evidence_chain": True, "reason": "per-study N"},
            {"display_id": "table_2", "kind": "pdf_table",
             "caption": "Table 2 GRADE summary of findings",
             "evidence_chain": False, "reason": "pooled OR"},
            {"display_id": "fig_3_3_1", "kind": "forest_plot",
             "caption": "Figure 3.3.1 Forest plot of overall complications",
             "evidence_chain": True, "reason": "per-study events"},
        ]},
        {"research_context": "", "tables": [TABLE1]},
        {"labels": [
            {"table_id": "table_1", "column_path": "N", "value_source": "source_paper"},
            {"table_id": "table_1", "column_path": "Year",
             "value_source": "bibliographic"},
        ]},
        {"labels": [
            {"table_id": "fig_3_3_1", "column_path": "Events",
             "value_source": "source_paper", "outcome": "overall complications"},
            {"table_id": "fig_3_3_1", "column_path": "Total",
             "value_source": "source_paper"},
            {"table_id": "fig_3_3_1", "column_path": "Odds ratio",
             "value_source": "review_computed"},
            {"table_id": "fig_3_3_1", "column_path": "Weight",
             "value_source": "review_computed"},
        ]},
    ])


@pytest.mark.asyncio
async def test_pipeline_selects_table1_and_forest_skips_table2():
    backend = _backend()
    gate = ScriptedCheckpoint()
    result = await ReviewExtraction(
        backend, reporter=StepReporter("re", gate=gate),
        forest_ocr=FakeForest(),
    ).run(DOC, pdf_path="", text_window=DOC, chars_total=25606, chars_used=25606)

    ids = [t.table_id for t in result.tables.tables]
    assert "table_1" in ids and "fig_3_3_1" in ids
    assert "table_2" not in ids
    assert result.lens.lens_one_line
    stages = [e.stage for e in gate.seen]
    assert stages[:3] == [
        StepStage.REVIEW_LENS, StepStage.EVIDENCE_LOCALIZE, StepStage.TABLE_CAPTURE]
    assert StepStage.FOREST_OCR in stages
    assert StepStage.CLAIM_ORIGIN in stages

    later = backend.prompts[1:]
    assert later, "localize/capture/origin must have been prompted"
    assert all("SECRET_ABSTRACT_TOKEN" not in p for p in later)
    assert "SELECTED DISPLAYS" in backend.prompts[2]
    assert "table_1" in backend.prompts[2]
    lens = next(e for e in gate.seen if e.stage is StepStage.REVIEW_LENS)
    focus = lens.render_blocks[0]
    assert focus.startswith("  review focus:")
    assert "elderly ESCC, MIE vs OE, complications" in focus
    assert "  extraction status:" in focus
    assert "25,606 characters extracted (using the first 25,606)" in focus
    localize = next(e for e in gate.seen if e.stage is StepStage.EVIDENCE_LOCALIZE)
    listed = localize.render_blocks[0]
    assert "— per-study N" in listed
    assert "[off] table_2" in listed
    assert localize.payload["displays"][0]["label"].startswith("[on]")
    assert lens.offers == ["retry"]
    assert "retry_alt" not in lens.offers
    capture = next(e for e in gate.seen if e.stage is StepStage.TABLE_CAPTURE)
    assert capture.offers == ["retry"]
    assert "retry_alt" not in capture.offers
    assert "model: queue" in lens.render_blocks[0]


@pytest.mark.asyncio
async def test_pipeline_origin_excludes_non_source_cells_from_audit_set():
    result = await ReviewExtraction(
        _backend(), forest_ocr=FakeForest(),
    ).run(DOC, text_window=DOC)
    forest = next(t for t in result.tables.tables if t.table_id == "fig_3_3_1")
    kept, dropped = [], []
    for header in forest.column_paths():
        fields = fields_for_cell(
            result.origin_labels, forest, {"column_header": header, "row": 0})
        (dropped if drop_non_source(fields["value_source"]) else kept).append(header)
    assert "Events" in kept and "Total" in kept
    assert "Odds ratio" in dropped and "Weight" in dropped
    assert any("review_computed" in note for note in result.dropped_non_source)


@pytest.mark.asyncio
async def test_parser_v3_items_exclude_review_computed_cells(monkeypatch):
    from pathlib import Path

    from react_review.dkb import FieldResolver, KnowledgeBase
    from react_review.parser.review_parser import ReviewParser
    from react_review.tools.forest_ocr import ForestOcrTool

    seed = Path(__file__).resolve().parents[3] / "configs" / "knowledge.seed.json"
    monkeypatch.setattr("react_review.parser.review_parser._pdf_text", lambda p: DOC)

    async def fake_ocr(self, payload):
        return await FakeForest().run(payload)

    monkeypatch.setattr(ForestOcrTool, "run", fake_ocr)
    backend = _backend()
    backend._responses.extend([
        json.dumps({"rows": [
            {"row": 0, "col": 2, "column_header": "N", "value": "80",
             "cohort_label": "", "unit": ""},
        ]}),
        json.dumps({"studies": []}),
    ])
    parsed = await ReviewParser(
        backend, FieldResolver(KnowledgeBase.from_json(seed)),
    ).parse("d.pdf")

    headers = {i.column_header for i in parsed.items}
    assert "N" in headers
    assert "Events" in headers and "Total" in headers
    assert "Odds ratio" not in headers and "Weight" not in headers
    assert "Year" not in headers
    assert all(
        (i.value_source or "source_paper") == "source_paper" for i in parsed.items)


@pytest.mark.asyncio
async def test_displays_captured_pause_shows_tables_and_figure_warnings():
    table_payload = dict(TABLE1)
    table_payload["difficulties"] = ["last column was cut off"]
    forest = FOREST.model_copy(deep=True)
    forest.difficulties = ["blurry axis labels"]
    forest.checksum_failures = ["Events"]

    class ForestWithWarnings(FakeForest):
        async def run(self, payload):
            table = forest.model_copy(deep=True)
            table.table_id = payload.figure_id or table.table_id
            table.caption = payload.caption or table.caption
            return ForestOcrResult(table=table)

    backend = RecordingQueue([
        {"lens_one_line": "elderly ESCC, MIE vs OE, complications",
         "domain": "esophageal cancer surgery",
         "population": "elderly ESCC",
         "comparison": "MIE vs OE",
         "outcomes": ["overall complications"]},
        {"displays": [
            {"display_id": "table_1", "kind": "pdf_table",
             "caption": "Table 1 Characteristics of the included studies",
             "evidence_chain": True, "reason": "per-study N"},
            {"display_id": "fig_3_3_1", "kind": "forest_plot",
             "caption": "Figure 3.3.1 Forest plot of overall complications",
             "evidence_chain": True, "reason": "per-study events"},
        ]},
        {"research_context": "", "tables": [table_payload]},
        {"labels": [
            {"table_id": "table_1", "column_path": "N", "value_source": "source_paper"},
        ]},
        {"labels": [
            {"table_id": "fig_3_3_1", "column_path": "Events",
             "value_source": "source_paper"},
        ]},
    ])
    gate = ScriptedCheckpoint()
    await ReviewExtraction(
        backend, reporter=StepReporter("re", gate=gate),
        forest_ocr=ForestWithWarnings(),
    ).run(DOC, text_window=DOC)

    capture = next(e for e in gate.seen if e.stage is StepStage.TABLE_CAPTURE)
    assert capture.title == "Displays captured"
    summary = next(b for b in capture.render_blocks if "figures:" in b)
    assert "tables:" in summary and "figures:" in summary
    assert any("last column was cut off" in w for w in capture.warnings)
    assert any("blurry axis labels" in w for w in capture.warnings)
    assert any("Events" in w for w in capture.warnings)
    assert StepStage.FOREST_OCR in [e.stage for e in gate.seen]


@pytest.mark.asyncio
async def test_no_forest_shows_none_and_skips_forest_ocr_step():
    backend = RecordingQueue([
        {"lens_one_line": "elderly ESCC, MIE vs OE, complications",
         "domain": "esophageal cancer surgery",
         "population": "elderly ESCC",
         "comparison": "MIE vs OE",
         "outcomes": ["overall complications"]},
        {"displays": [
            {"display_id": "table_1", "kind": "pdf_table",
             "caption": "Table 1 Characteristics of the included studies",
             "evidence_chain": True, "reason": "per-study N"},
        ]},
        {"research_context": "", "tables": [TABLE1]},
        {"labels": [
            {"table_id": "table_1", "column_path": "N", "value_source": "source_paper"},
        ]},
    ])
    gate = ScriptedCheckpoint()
    await ReviewExtraction(
        backend, reporter=StepReporter("re", gate=gate),
        forest_ocr=FakeForest(),
    ).run(DOC, text_window=DOC)

    capture = next(e for e in gate.seen if e.stage is StepStage.TABLE_CAPTURE)
    summary = next(b for b in capture.render_blocks if "figures:" in b)
    assert "(none)" in summary.split("figures:", 1)[1]
    assert StepStage.FOREST_OCR not in [e.stage for e in gate.seen]


class _NamedQueue(RecordingQueue):
    def __init__(self, name: str, responses: list) -> None:
        super().__init__(responses)
        self._name = name
        self.seeds: list[int] = []

    @property
    def model_id(self) -> str:
        return self._name

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.seeds.append(seed)
        return await super().complete(prompt, seed=seed)


_LENS = {
    "lens_one_line": "elderly ESCC, MIE vs OE, complications",
    "domain": "esophageal cancer surgery",
    "population": "elderly ESCC",
    "comparison": "MIE vs OE",
    "outcomes": ["overall complications"],
}


@pytest.mark.asyncio
async def test_lens_retry_increments_seed_and_reruns():
    backend = _NamedQueue("queue", [_LENS, {**_LENS, "lens_one_line": "retried lens"}])
    gate = ScriptedCheckpoint([Decision.RETRY, Decision.STOP])
    with pytest.raises(RunStopped):
        await ReviewExtraction(
            backend, reporter=StepReporter("re", gate=gate),
            forest_ocr=FakeForest(),
        ).run(DOC, text_window=DOC)
    assert backend.seeds[:2] == [42, 43]
    assert gate.seen[0].payload["seed"] == 42
    assert gate.seen[1].payload["seed"] == 43
    assert "retried lens" in gate.seen[1].render_blocks[0]
    assert gate.seen[0].offers == ["retry"]


@pytest.mark.asyncio
async def test_lens_retry_alt_uses_model_2_and_shows_model_id():
    primary = _NamedQueue("glm-a", [_LENS])
    alt = _NamedQueue("glm-b", [{**_LENS, "lens_one_line": "alt lens"}])
    gate = ScriptedCheckpoint([Decision.RETRY_ALT, Decision.STOP])
    with pytest.raises(RunStopped):
        await ReviewExtraction(
            primary, alt_backend=alt, reporter=StepReporter("re", gate=gate),
            forest_ocr=FakeForest(),
        ).run(DOC, text_window=DOC)
    assert primary.seeds == [42] and alt.seeds == [42]
    assert gate.seen[0].payload["model_id"] == "glm-a"
    assert gate.seen[1].payload["model_id"] == "glm-b"
    assert "retry_alt" in gate.seen[0].offers
    assert "model: glm-b" in gate.seen[1].render_blocks[0]


@pytest.mark.asyncio
async def test_lens_retry_alt_without_fallback_does_not_silently_accept():
    backend = _NamedQueue("queue", [_LENS, _LENS])
    gate = ScriptedCheckpoint([Decision.RETRY_ALT, Decision.CONTINUE])
    with pytest.raises(RuntimeError, match="no alt_backend"):
        await ReviewExtraction(
            backend, reporter=StepReporter("re", gate=gate),
            forest_ocr=FakeForest(),
        ).run(DOC, text_window=DOC)
    assert gate.seen[0].offers == ["retry"]
    assert "retry_alt" not in gate.seen[0].offers
    assert len(gate.seen) == 1

