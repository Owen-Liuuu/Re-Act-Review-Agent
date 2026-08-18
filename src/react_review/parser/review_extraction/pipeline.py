"""Orchestrate lens → localize → selected capture → forest OCR → origin labels."""
from __future__ import annotations

from pathlib import Path

import structlog

from react_review.hitl.events import StepStage, SubjectKind
from react_review.hitl.reporter import StepReporter
from react_review.llm.base import LLMBackend
from react_review.parser.review_extraction.lens import read_lens
from react_review.parser.review_extraction.localize import localize, selected
from react_review.parser.review_extraction.origin import dropped_notes, label_origins
from react_review.parser.review_extraction.schemas import (
    DisplayHit,
    ExtractionResult,
    ReviewLens,
)
from react_review.parser.review_extraction.windows import capture_window, missed_forest_hint
from react_review.parser.table_capture import TableCapturer
from react_review.parser.table_render import render_table_set, to_csv
from react_review.schemas.table import CapturedTable, CapturedTableSet
from react_review.tools.models import ForestOcrInput

logger = structlog.get_logger(__name__)


class ReviewExtraction:
    """The testable Review Extraction unit. Parser tail stays in ReviewParser."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        reporter: StepReporter | None = None,
        alt_backend: LLMBackend | None = None,
        prompt_profile: str = "table_capture_v3",
        keep_tables: set[str] | None = None,
        drop_tables: set[str] | None = None,
        forest_ocr=None,
        vision_backend: LLMBackend | None = None,
    ) -> None:
        self._backend = backend
        self._reporter = reporter or StepReporter()
        self._keep = keep_tables
        self._drop = drop_tables
        self._forest_ocr = forest_ocr
        self._vision = vision_backend
        self._capturer = TableCapturer(
            backend, alt_backend=alt_backend, prompt_profile=prompt_profile)

    async def run(
        self,
        full_text: str,
        *,
        pdf_path: Path | str = "",
        text_window: str = "",
    ) -> ExtractionResult:
        path = str(Path(pdf_path).resolve()) if pdf_path else ""
        subject = path
        kind = SubjectKind.REVIEW_PDF if path else SubjectKind.NONE

        lens = await read_lens(self._backend, full_text)
        await self._reporter.step_or_stop(
            StepStage.REVIEW_LENS,
            title="Review lens compressed",
            subject=subject, subject_kind=kind,
            payload={"lens": lens.model_dump(mode="json")},
            render_blocks=[_render_lens(lens)],
            warnings=list(lens.difficulties),
        )

        hits = await localize(self._backend, lens, full_text)
        hits = await self._gate_hits(hits, subject, kind, full_text)

        tables_sel = [
            {"display_id": h.display_id, "caption": h.caption,
             "page_hint": h.page_hint}
            for h in selected(hits, kind="pdf_table")
        ]
        table_set, captured_ctx = await self._capturer.capture(
            capture_window(full_text) or text_window or full_text,
            reporter=self._reporter, pdf_path=path,
            keep=self._keep, drop=self._drop, selected=tables_sel,
        )
        for table in table_set.tables:
            if not table.display_kind:
                table.display_kind = "pdf_table"
            if not table.capture_method:
                table.capture_method = "table_text"

        forest_tables = await self._ocr_forests(
            selected(hits, kind="forest_plot"), path, lens)
        await self._gate_forests(forest_tables, subject, kind)

        combined = CapturedTableSet(
            tables=[*table_set.tables, *forest_tables],
            source_pdf=table_set.source_pdf or path,
            dropped=list(table_set.dropped),
            dropped_reason=table_set.dropped_reason,
        )
        labels = await label_origins(self._backend, lens, combined.tables)
        combined.origin_labels = labels
        dropped = dropped_notes(labels)
        await self._reporter.step_or_stop(
            StepStage.CLAIM_ORIGIN,
            title="Source vs review-computed labels",
            subject=subject, subject_kind=kind,
            payload={
                "labels": [item.model_dump(mode="json") for item in labels],
                "dropped_non_source": dropped,
            },
            render_blocks=[_render_origins(labels, dropped)],
            warnings=dropped,
        )

        context = lens.lens_one_line or captured_ctx
        return ExtractionResult(
            lens=lens, hits=hits, tables=combined, origin_labels=labels,
            research_context=context, dropped_non_source=dropped,
        )

    async def _gate_hits(
        self, hits: list[DisplayHit], subject: str, kind: SubjectKind, text: str,
    ) -> list[DisplayHit]:
        hint = missed_forest_hint(text, hits)
        warnings = [hint] if hint else []
        if not hits:
            warnings.append("no displays were listed from the results window")
        await self._reporter.step_or_stop(
            StepStage.EVIDENCE_LOCALIZE,
            title="Evidence-chain displays",
            subject=subject, subject_kind=kind,
            payload={"displays": [
                {"id": h.display_id, "label": (
                    f"{h.display_id}  {h.kind}  "
                    f"{'on' if h.evidence_chain else 'off'}  "
                    f"{h.caption or '(no caption)'}"),
                 **h.model_dump(mode="json")}
                for h in hits]},
            render_blocks=[_render_hits(hits)],
            warnings=warnings,
            selectable="displays",
        )
        event = self._reporter.last_event
        if event is None or not event.dropped:
            return hits
        kept = {item.get("id") for item in event.selectable_items()}
        return [h for h in hits if h.display_id in kept]

    async def _ocr_forests(
        self, hits: list[DisplayHit], pdf_path: str, lens: ReviewLens,
    ) -> list[CapturedTable]:
        if not hits:
            return []
        tool = self._forest_ocr
        if tool is None:
            from react_review.tools.forest_ocr import ForestOcrTool
            tool = ForestOcrTool(self._backend, vision_backend=self._vision)
        tables: list[CapturedTable] = []
        for ordinal, hit in enumerate(hits):
            try:
                result = await tool.run(ForestOcrInput(
                    pdf_path=pdf_path,
                    figure_id=hit.display_id,
                    caption=hit.caption,
                    page_hint=hit.page_hint,
                    outcomes=list(lens.outcomes),
                    figure_ordinal=ordinal,
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning("forest_ocr_failed", display_id=hit.display_id,
                               error=str(exc)[:160])
                tables.append(_empty_forest(hit, [str(exc)[:160]]))
                continue
            table = result.table
            if not table.table_id:
                table.table_id = hit.display_id
            table.display_kind = "forest_plot"
            table.capture_method = "figure_ocr"
            if not table.caption:
                table.caption = hit.caption
            if not table.outcome:
                table.outcome = hit.caption
            tables.append(table)
        return tables

    async def _gate_forests(
        self, tables: list[CapturedTable], subject: str, kind: SubjectKind,
    ) -> None:
        warnings = [w for t in tables for w in t.difficulties]
        if not tables:
            return
        await self._reporter.step_or_stop(
            StepStage.FOREST_OCR,
            title="Forest plots read",
            subject=subject, subject_kind=kind,
            payload={"tables": [
                {"id": t.table_id, "table_id": t.table_id,
                 "label": f"{t.table_id}  {t.caption or '(no caption)'}"
                          f"  [{len(t.rows)} rows]",
                 **t.model_dump(mode="json")}
                for t in tables]},
            render_blocks=[render_table_set(tables)] if tables else ["  (no forest plots)"],
            warnings=warnings,
            selectable="tables",
            sidecars={f"{t.table_id}.csv": to_csv(t) for t in tables},
        )
        event = self._reporter.last_event
        if event is not None and event.dropped:
            kept = {item.get("id") or item.get("table_id") for item in event.selectable_items()}
            tables[:] = [t for t in tables if t.table_id in kept]


def _empty_forest(hit: DisplayHit, difficulties: list[str]) -> CapturedTable:
    return CapturedTable(
        table_id=hit.display_id,
        caption=hit.caption,
        page_hint=hit.page_hint,
        role="outcomes",
        display_kind="forest_plot",
        capture_method="figure_ocr",
        outcome=hit.caption,
        row_axis_columns=["Study or Subgroup"],
        difficulties=difficulties,
    )


def _render_lens(lens: ReviewLens) -> str:
    lines = ["  compressed review lens (later steps do not see the abstract)"]
    if lens.lens_one_line:
        lines.append(f"    {lens.lens_one_line}")
    for name in ("domain", "population", "comparison"):
        value = getattr(lens, name)
        if value:
            lines.append(f"    {name}: {value}")
    if lens.outcomes:
        lines.append("    outcomes: " + "; ".join(lens.outcomes))
    return "\n".join(lines)


def _render_hits(hits: list[DisplayHit]) -> str:
    if not hits:
        return "  (no displays listed)"
    lines = [f"  {len(hits)} display(s)"]
    for hit in hits:
        flag = "ON " if hit.evidence_chain else "off"
        lines.append(
            f"    [{flag}] {hit.display_id:<16} {hit.kind:<12} "
            f"{hit.caption[:60] or '(no caption)'}")
    return "\n".join(lines)


def _render_origins(labels, dropped: list[str]) -> str:
    if not labels:
        return "  (no origin labels)"
    lines = [f"  {len(labels)} origin label(s); {len(dropped)} non-source column(s) dropped"]
    for label in labels[:40]:
        lines.append(
            f"    {label.table_id:<16} {label.column_path or '(all)':<28} "
            f"{label.value_source}")
    if len(labels) > 40:
        lines.append(f"    … {len(labels) - 40} more")
    return "\n".join(lines)
