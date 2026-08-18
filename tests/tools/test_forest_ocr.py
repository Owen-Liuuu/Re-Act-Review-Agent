"""ocr_forest_plot: text-layer grid only; raster prose does not invent events."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.llm.base import LLMBackend
from react_review.tools.forest_ocr import (
    ForestOcrPromptContract,
    ForestOcrTool,
    _has_per_study_grid,
)
from react_review.tools.models import ForestOcrInput


class QueueBackend(LLMBackend):
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


INJECTED = {
    "table_id": "fig_3_3_1",
    "caption": "Figure 3.3.1 Forest plot of overall complications",
    "header_rows": [["Study or Subgroup", "Events", "Total"]],
    "rows": [["Li J 2015", "23", "58"], ["Capovilla 2023", "12", "40"]],
    "row_axis_columns": ["Study or Subgroup"],
}

# doc05 page 10: forest-plot prose, no per-study Events/Total grid in the text layer.
DOC05_P10_PROSE = """\
95% CI 0.19-0.73), and Li K 2025 (OR 0.51, 95% CI 0.25-1.05). The forest plot is presented
in Figure 3. The certainty of evidence was rated as moderate (GRADE ⊕⊕⊕⊝).

3.3.3 30-Day Mortality
Two studies reported 30-day mortality data in the matched population. MIE was associated
with a non-significantly lower 30-day mortality compared to OE (OR 0.38, 95% CI 0.07-2.04,
P=0.26). Li K 2025 reported zero mortality events in both groups (0/92 MIE vs 0/55 OE) and
was therefore not estimable. Heterogeneity assessment was not applicable. The forest plot is
presented in Figure 4. The certainty of evidence was rated as very low (GRADE ⊕◯◯◯),
primarily due to serious risk of bias from retrospective study designs and very serious
imprecision from the low number of events (7 total events).

3.3.4 Anastomotic Leak
All three studies reported anastomotic leak rates. Anastomotic leak was non-significantly
higher in the MIE group compared to OE (OR 1.77, 95% CI 0.91-3.45, P=0.09). No significant
heterogeneity was detected (I²=0%, Chi²=0.18, P=0.91). Individual study ORs were: Li J 2015
(OR 1.81, 95% CI 0.61-5.35), Capovilla 2023 (OR 1.45, 95% CI 0.43-4.88), and Li K 2025
(OR 2.10, 95% CI 0.65-6.79). The forest plot is presented in Figure 5. The certainty of evidence
was rated as low (GRADE ⊕⊕◯◯).
"""

DOC05 = Path(__file__).resolve().parents[2] / "eval" / "benchmark_3" / "raw" / "doc05.pdf"


def test_forest_ocr_contract_does_not_drift():
    assert ForestOcrPromptContract.load().drifts() == []


def test_forest_vision_contract_does_not_drift():
    from react_review.tools.forest_ocr import ForestVisionPromptContract
    assert ForestVisionPromptContract.load().drifts() == []


def test_has_per_study_grid_requires_name_year_and_two_ints():
    assert _has_per_study_grid("Li J 2015 23 58 32 54")
    assert _has_per_study_grid("Capovilla 2023 12 40 15 30")
    assert not _has_per_study_grid(DOC05_P10_PROSE)
    assert not _has_per_study_grid("Li J 2015 (OR 0.40, 95% CI 0.19-0.85)")


@pytest.mark.asyncio
async def test_injected_table_is_returned_verbatim():
    out = await ForestOcrTool().run(ForestOcrInput(
        figure_id="fig_3_3_1", injected_table=INJECTED))
    assert out.table.rows == [["Li J 2015", "23", "58"], ["Capovilla 2023", "12", "40"]]
    assert out.table.display_kind == "forest_plot"
    assert "23" in [cell for row in out.table.rows for cell in row]


@pytest.mark.asyncio
async def test_text_only_path_returns_empty_grid_and_does_not_invent_events():
    out = await ForestOcrTool().run(ForestOcrInput(
        figure_id="fig_3_3_1",
        caption="Figure 3.3.1 Forest plot of overall complications",
    ))
    assert out.table.rows == []
    assert any("not invented" in d for d in out.difficulties)
    assert "23" not in str(out.table.model_dump())


def _write_pdf(path: Path, *pages: str) -> None:
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


@pytest.mark.asyncio
async def test_llm_path_parses_a_text_dump_when_a_backend_is_present(tmp_path):
    pdf = tmp_path / "review.pdf"
    _write_pdf(pdf, "Figure 3.3.1 Forest plot\nLi J 2015 23 58 32 54")

    backend = QueueBackend([{
        "tables": [{
            "table_id": "fig_3_3_1",
            "caption": "Figure 3.3.1",
            "header_rows": [["Study or Subgroup", "Events", "Total"]],
            "rows": [["Li J 2015", "23", "58"]],
            "row_axis_columns": ["Study or Subgroup"],
            "difficulties": [],
        }],
    }])
    out = await ForestOcrTool(backend).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="fig_3_3_1",
        caption="Figure 3.3.1 Forest plot", page_hint="1",
        outcomes=["overall complications"],
    ))
    assert backend.prompts, "a per-study grid must reach the model"
    assert out.table.rows == [["Li J 2015", "23", "58"]]
    assert "Never invent" in backend.prompts[0]


@pytest.mark.asyncio
async def test_missed_locate_does_not_dump_page1_grid(tmp_path):
    """A fake Events grid on page 1 must not be read when caption/page_hint miss."""
    pdf = tmp_path / "two.pdf"
    _write_pdf(
        pdf,
        "Li J 2015 23 58 32 54",
        "Discussion with no forest caption",
    )
    backend = QueueBackend([{"tables": [{"rows": [["Li J 2015", "23", "58"]]}]}])
    out = await ForestOcrTool(backend).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="figure_2",
        caption="Forest plot of overall postoperative complications",
        page_hint="printed page or empty",
    ))
    assert backend.prompts == []
    assert out.table.rows == []
    assert any("not invented" in d for d in out.difficulties)


@pytest.mark.asyncio
async def test_doc05_page10_prose_does_not_call_the_model(tmp_path):
    pdf = tmp_path / "p10.pdf"
    _write_pdf(pdf, DOC05_P10_PROSE)
    backend = QueueBackend([{"tables": [{"rows": [["Li J 2015", "23", "58"]]}]}])
    out = await ForestOcrTool(backend).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="figure_2",
        caption="Forest plot of overall postoperative complications",
        page_hint="1",
    ))
    assert backend.prompts == []
    assert out.table.rows == []
    assert any("raster pixels" in d for d in out.difficulties)


@pytest.mark.asyncio
async def test_doc05_pdf_page10_does_not_call_the_model():
    if not DOC05.is_file():
        pytest.skip("eval/benchmark_3/raw/doc05.pdf is not present")
    backend = QueueBackend([{"tables": [{"rows": [["Li J 2015", "23", "58"]]}]}])
    out = await ForestOcrTool(backend).run(ForestOcrInput(
        pdf_path=str(DOC05), figure_id="figure_2",
        caption="Forest plot of overall postoperative complications (MIE vs. OE).",
        page_hint="10",
    ))
    assert backend.prompts == []
    assert out.table.rows == []
    assert any("raster pixels" in d for d in out.difficulties)


class VisionQueue(LLMBackend):
    def __init__(self, payload=None, error: Exception | None = None) -> None:
        super().__init__()
        self.payload = payload if payload is not None else {}
        self.error = error
        self.prompts: list[str] = []
        self.image_sizes: list[int] = []

    @property
    def model_id(self) -> str:
        return "vision-queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        raise AssertionError("text complete must not run on the vision path")

    async def complete_vision(
        self, prompt: str, images: list[bytes], *, seed: int = 42,
    ) -> str:
        self.prompts.append(prompt)
        self.image_sizes.append(len(images[0]) if images else 0)
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload)


@pytest.mark.asyncio
async def test_vision_path_applies_checksum_and_keeps_other_columns(monkeypatch, tmp_path):
    from react_review.tools.forest_ocr import _parse_int_cell, _split_forest_rows

    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures" / "forest_checksum"
        / "vision_probe_v2_flashx__forest_1.json")
    parsed = json.loads(fixture.read_text(encoding="utf-8"))["parsed"]
    vision = VisionQueue(parsed)
    monkeypatch.setattr(
        "react_review.tools.forest_ocr._figure_image", lambda *a, **k: b"fake-png")
    pdf = tmp_path / "p10.pdf"
    _write_pdf(pdf, DOC05_P10_PROSE)
    out = await ForestOcrTool(QueueBackend([]), vision_backend=vision).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="figure_2",
        caption="Forest plot of overall postoperative complications",
        page_hint="1",
    ))
    assert vision.prompts, "raster figures must call complete_vision"
    assert "Favours" in vision.prompts[0]
    assert out.table.checksum_failures == ["Open Esophagectomy (OE) Total"]
    study, _summary = _split_forest_rows(out.table)
    kept = sum(
        1 for row in study for cell in row[1:]
        if _parse_int_cell(cell) is not None)
    assert kept == 9
    assert any("not released" in note for note in out.table.difficulties)


@pytest.mark.asyncio
async def test_vision_429_is_an_honest_empty_table(monkeypatch, tmp_path):
    from react_review.core.exceptions import LLMError

    vision = VisionQueue(error=LLMError("HTTP 429: code 1305"))
    monkeypatch.setattr(
        "react_review.tools.forest_ocr._figure_image", lambda *a, **k: b"fake-png")
    pdf = tmp_path / "p10.pdf"
    _write_pdf(pdf, DOC05_P10_PROSE)
    out = await ForestOcrTool(QueueBackend([]), vision_backend=vision).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="figure_2",
        caption="Forest plot of overall postoperative complications",
        page_hint="1",
    ))
    assert out.table.rows == []
    assert any("429" in d for d in out.difficulties)
    assert any("not invented" in d for d in out.difficulties)
    assert "23" not in str(out.table.model_dump())


@pytest.mark.asyncio
async def test_text_path_applies_checksum_before_release(tmp_path):
    pdf = tmp_path / "review.pdf"
    _write_pdf(pdf, "Figure 3.3.1 Forest plot\nLi J 2015 23 58 32 54")
    backend = QueueBackend([{
        "tables": [{
            "table_id": "fig_3_3_1",
            "header_rows": [["Study or Subgroup", "Events", "Total"]],
            "rows": [["Li J 2015", "23", "58"], ["Total", "1", "1"]],
            "row_axis_columns": ["Study or Subgroup"],
        }],
    }])
    out = await ForestOcrTool(backend).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="fig_3_3_1",
        caption="Figure 3.3.1 Forest plot", page_hint="1",
    ))
    assert out.table.capture_path == "text"
    assert out.table.checksum_failures == ["Events", "Total"]
    assert out.table.rows[0][1:] == ["", ""]
    assert any("not released" in note for note in out.table.difficulties)


@pytest.mark.asyncio
async def test_vision_runs_before_text_when_both_are_available(monkeypatch, tmp_path):
    pdf = tmp_path / "review.pdf"
    _write_pdf(pdf, "Figure 3.3.1 Forest plot\nLi J 2015 23 58 32 54")
    backend = QueueBackend([{
        "tables": [{"rows": [["Li J 2015", "99", "99"]]}],
    }])
    vision = VisionQueue({
        "column_headers": ["Events", "Total"],
        "rows": [
            {"label": "Li J 2015", "kind": "study",
             "values": [{"column": "Events", "value": "23"},
                        {"column": "Total", "value": "58"}]},
            {"label": "Total", "kind": "summary",
             "values": [{"column": "Events", "value": "23"},
                        {"column": "Total", "value": "58"}]},
        ],
        "difficulties": [],
    })
    monkeypatch.setattr(
        "react_review.tools.forest_ocr._figure_image", lambda *a, **k: b"fake-png")
    out = await ForestOcrTool(backend, vision_backend=vision).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="fig_3_3_1",
        caption="Figure 3.3.1 Forest plot", page_hint="1",
    ))
    assert vision.prompts
    assert backend.prompts == []
    assert out.table.capture_path == "vision"
    assert out.table.rows[0][1:] == ["23", "58"]
    assert any("text-layer grid was present but not used" in d for d in out.difficulties)


@pytest.mark.asyncio
async def test_429_does_not_fall_back_to_text(monkeypatch, tmp_path):
    from react_review.core.exceptions import LLMError

    pdf = tmp_path / "review.pdf"
    _write_pdf(pdf, "Figure 3.3.1 Forest plot\nLi J 2015 23 58 32 54")
    backend = QueueBackend([{
        "tables": [{"rows": [["Li J 2015", "23", "58"]]}],
    }])
    vision = VisionQueue(error=LLMError("HTTP 429: code 1305"))
    monkeypatch.setattr(
        "react_review.tools.forest_ocr._figure_image", lambda *a, **k: b"fake-png")
    out = await ForestOcrTool(backend, vision_backend=vision).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="fig_3_3_1",
        caption="Figure 3.3.1 Forest plot", page_hint="1",
    ))
    assert backend.prompts == []
    assert out.table.rows == []
    assert out.table.capture_path == "vision"
    assert out.table.image_bytes == len(b"fake-png")
    assert any("429" in d for d in out.difficulties)


@pytest.mark.asyncio
async def test_missing_image_falls_back_to_text(tmp_path):
    pdf = tmp_path / "review.pdf"
    _write_pdf(pdf, "Figure 3.3.1 Forest plot\nLi J 2015 23 58 32 54")
    backend = QueueBackend([{
        "tables": [{
            "header_rows": [["Study or Subgroup", "Events", "Total"]],
            "rows": [["Li J 2015", "23", "58"]],
            "row_axis_columns": ["Study or Subgroup"],
        }],
    }])
    vision = VisionQueue({"rows": [{"label": "invented", "kind": "study", "values": []}]})
    out = await ForestOcrTool(backend, vision_backend=vision).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="fig_3_3_1",
        caption="Figure 3.3.1 Forest plot", page_hint="1",
    ))
    assert vision.prompts == []
    assert backend.prompts
    assert out.table.capture_path == "text"
    assert out.table.rows == [["Li J 2015", "23", "58"]]


def test_doc05_pairs_four_forests_in_reading_order_and_skips_prisma():
    if not DOC05.is_file():
        pytest.skip("eval/benchmark_3/raw/doc05.pdf is not present")
    from react_review.tools.forest_ocr import (
        _locate_figure_images, _paired_forest_images, _resolve_figure,
    )

    pairs = _paired_forest_images(str(DOC05))
    assert [p.image.xref for p in pairs] == [48, 51, 52, 55]
    assert [p.outcome_key for p in pairs] == [
        "overall_complications", "pulmonary", "30_day_mortality", "anastomotic_leak"]
    assert 39 in {im.xref for im in _locate_figure_images(str(DOC05))}

    pul = _resolve_figure(
        str(DOC05), caption="Forest plot of pulmonary complications",
        figure_ordinal=2)
    mort = _resolve_figure(
        str(DOC05), caption="Forest plot of 30-day mortality",
        figure_ordinal=1)
    assert pul.status == "ok" and pul.image is not None and pul.image.xref == 51
    assert mort.status == "ok" and mort.image is not None and mort.image.xref == 52
    assert pul.review_required is False

    literal = _resolve_figure(
        str(DOC05), caption="Forest plot of pulmonary complications",
        page_hint="printed page or empty", figure_ordinal=0)
    assert literal.image is not None and literal.image.xref == 51


def test_doc05_figure_image_works_without_a_numeric_page_hint():
    if not DOC05.is_file():
        pytest.skip("eval/benchmark_3/raw/doc05.pdf is not present")
    from react_review.tools.forest_ocr import _figure_image

    blob = _figure_image(
        str(DOC05), page_hint="printed page or empty",
        caption="Forest plot of pulmonary complications", figure_ordinal=99)
    assert len(blob) > 1000


@pytest.mark.asyncio
async def test_caption_mismatch_does_not_release_or_fall_back_to_text(tmp_path):
    pdf = tmp_path / "review.pdf"
    _write_pdf(pdf, "Figure 3.3.1 Forest plot\nLi J 2015 23 58 32 54")
    backend = QueueBackend([{
        "tables": [{"rows": [["Li J 2015", "23", "58"]]}],
    }])
    vision = VisionQueue({"rows": []})
    out = await ForestOcrTool(backend, vision_backend=vision).run(ForestOcrInput(
        pdf_path=str(pdf), figure_id="fig_3_3_1",
        caption="Forest plot of pulmonary complications",
        page_hint="1",
    ))
    assert backend.prompts == []
    assert vision.prompts == []
    assert out.table.rows == []
    assert out.table.capture_path == "mismatch"
    assert out.table.review_required is True
    assert any("not invented" in d for d in out.difficulties)

