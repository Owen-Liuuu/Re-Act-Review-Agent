"""Orchestrate lens → localize → selected capture → forest OCR → origin labels."""
from __future__ import annotations

import time
from pathlib import Path

import structlog

from react_review.hitl.events import StepStage, SubjectKind
from react_review.hitl.gate import Decision, require_alt_backend, retry_offers
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
from react_review.parser.table_render import (
    display_caption,
    render_display_summary,
    render_shape_report,
    render_table_set,
    to_csv,
)
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
        step_backends=None,
    ) -> None:
        self._backend = backend
        self._reporter = reporter or StepReporter()
        self._keep = keep_tables
        self._drop = drop_tables
        self._forest_ocr = forest_ocr
        self._vision = vision_backend
        self._steps = step_backends
        self._capturer = TableCapturer(
            self._slot("table_capture"), alt_backend=alt_backend,
            prompt_profile=prompt_profile)
        self._alt = alt_backend

    def _slot(self, name: str) -> LLMBackend:
        if name == "forest_ocr_vision":
            if self._steps is None:
                return self._vision
            got = getattr(self._steps, name, None)
            return got if got is not None else self._vision
        if self._steps is None:
            return self._backend
        got = getattr(self._steps, name, None)
        return got if got is not None else self._backend

    async def run(
        self,
        full_text: str,
        *,
        pdf_path: Path | str = "",
        text_window: str = "",
        chars_total: int = 0,
        chars_used: int = 0,
        truncated: bool = False,
    ) -> ExtractionResult:
        path = str(Path(pdf_path).resolve()) if pdf_path else ""
        subject = path
        kind = SubjectKind.REVIEW_PDF if path else SubjectKind.NONE

        started = time.monotonic()
        seed = 42
        lens_backend = self._slot("review_lens")
        while True:
            lens = await read_lens(lens_backend, full_text, seed=seed)
            self._reporter.progress("review_lens", started=started)
            decision = await self._reporter.step_or_stop(
                StepStage.REVIEW_LENS,
                title="Review lens compressed",
                subject=subject, subject_kind=kind,
                payload={"lens": lens.model_dump(mode="json"),
                         "chars_total": chars_total, "chars_used": chars_used,
                         "truncated": truncated,
                         "model_id": lens_backend.model_id,
                         "seed": seed},
                render_blocks=[_render_lens(
                    lens, chars_total=chars_total, chars_used=chars_used,
                    truncated=truncated, model_id=lens_backend.model_id)],
                warnings=list(lens.difficulties),
                offers=retry_offers(self._alt),
                started=started,
            )
            if decision is Decision.RETRY:
                seed += 1
                started = time.monotonic()
                continue
            if decision is Decision.RETRY_ALT:
                lens_backend = require_alt_backend(self._alt, stage="review_lens")
                started = time.monotonic()
                continue
            break

        started = time.monotonic()
        hits = await localize(self._slot("evidence_localize"), lens, full_text)
        self._reporter.progress("evidence_localize", started=started)
        hits = await self._gate_hits(hits, subject, kind, full_text, started=started)

        window = capture_window(full_text) or text_window or full_text
        tables_sel = [
            {"display_id": h.display_id, "caption": h.caption,
             "page_hint": h.page_hint}
            for h in selected(hits, kind="pdf_table")
        ]
        seed = 42
        capture_backend = None
        forest_tables: list[CapturedTable] | None = None
        while True:
            table_set, captured_ctx = await self._capturer.capture(
                window, reporter=self._reporter, pdf_path=path,
                keep=self._keep, drop=self._drop, selected=tables_sel,
                defer_gate=True, seed=seed, backend=capture_backend,
            )
            for table in table_set.tables:
                if not table.display_kind:
                    table.display_kind = "pdf_table"
                if not table.capture_method:
                    table.capture_method = "table_text"
            if forest_tables is None:
                forest_tables = await self._ocr_forests(
                    selected(hits, kind="forest_plot"), path, lens)
            decision = await self._gate_displays(
                table_set, forest_tables, subject, kind,
                model_id=(capture_backend or self._backend).model_id)
            if decision is Decision.RETRY:
                seed += 1
                continue
            if decision is Decision.RETRY_ALT:
                capture_backend = require_alt_backend(
                    self._capturer._alt, stage="review_table_capture")
                continue
            event = self._reporter.last_event
            if event is not None and event.dropped:
                dropped = set(event.dropped)
                table_set = table_set.keep_only(
                    {t.table_id for t in table_set.tables if t.table_id not in dropped},
                    reason="dropped at checkpoint")
                forest_tables[:] = [t for t in forest_tables if t.table_id not in dropped]
            break

        await self._gate_forests(forest_tables or [], subject, kind)

        combined = CapturedTableSet(
            tables=[*table_set.tables, *(forest_tables or [])],
            source_pdf=table_set.source_pdf or path,
            dropped=list(table_set.dropped),
            dropped_reason=table_set.dropped_reason,
        )
        started = time.monotonic()
        labels = await label_origins(self._slot("claim_origin"), lens, combined.tables)
        self._reporter.progress("claim_origin", 1, 1, started=started)
        combined.origin_labels = labels
        dropped = dropped_notes(labels)
        await self._reporter.step_or_stop(
            StepStage.CLAIM_ORIGIN,
            title="Claim origin (source vs review-computed)",
            subject=subject, subject_kind=kind,
            payload={
                "labels": [item.model_dump(mode="json") for item in labels],
                "dropped_non_source": dropped,
            },
            render_blocks=[_render_origins(labels, dropped)],
            warnings=dropped,
            started=started,
        )

        context = lens.lens_one_line or captured_ctx
        return ExtractionResult(
            lens=lens, hits=hits, tables=combined, origin_labels=labels,
            research_context=context, dropped_non_source=dropped,
        )

    async def _gate_hits(
        self, hits: list[DisplayHit], subject: str, kind: SubjectKind, text: str,
        *, started: float | None = None,
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
                    f"{'[on]' if h.evidence_chain else '[off]'} {h.display_id}  "
                    f"{h.kind}  {h.caption or '(no caption)'}"
                    + (f"  — {_compress_reason(h.reason)}" if h.reason else "")),
                 **h.model_dump(mode="json")}
                for h in hits]},
            render_blocks=[_render_hits(hits)],
            warnings=warnings,
            selectable="displays",
            started=started,
        )
        event = self._reporter.last_event
        if event is None:
            return hits
        by_id = {h.display_id: h for h in hits}
        dropped = set(event.dropped)
        for item in event.selectable_items():
            hid = str(item.get("id") or item.get("display_id") or "")
            hit = by_id.get(hid)
            if hit is None:
                continue
            if "evidence_chain" in item:
                hit.evidence_chain = bool(item["evidence_chain"])
        for hid in dropped:
            hit = by_id.get(hid)
            if hit is not None:
                hit.evidence_chain = False
        return hits

    async def _ocr_forests(
        self, hits: list[DisplayHit], pdf_path: str, lens: ReviewLens,
    ) -> list[CapturedTable]:
        if not hits:
            return []
        tool = self._forest_ocr
        if tool is None:
            from react_review.tools.forest_ocr import ForestOcrTool
            tool = ForestOcrTool(
                self._slot("forest_ocr_text"),
                vision_backend=self._slot("forest_ocr_vision"))
        tables: list[CapturedTable] = []
        for ordinal, hit in enumerate(hits):
            started = time.monotonic()
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
                self._reporter.progress(
                    "figure", ordinal + 1, len(hits),
                    caption=display_caption(hit.caption), started=started)
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
            self._reporter.progress(
                "figure", ordinal + 1, len(hits),
                caption=display_caption(table.caption or hit.caption),
                started=started)
        return tables

    async def _gate_displays(
        self,
        table_set: CapturedTableSet,
        figures: list[CapturedTable],
        subject: str,
        kind: SubjectKind,
        *,
        model_id: str = "",
    ) -> Decision:
        table_warnings = [w for t in table_set.tables for w in render_shape_report(t)]
        if not table_set.tables:
            table_warnings.insert(
                0, "no table was captured — the review's data table was "
                   "not found, or the text layer does not contain it")
        figure_warnings = [w for t in figures for w in t.difficulties]
        figure_warnings += [
            f"{t.table_id}: checksum failed — {fail}"
            for t in figures for fail in (t.checksum_failures or [])
            if fail not in t.difficulties
        ]
        displays = [
            {"id": t.table_id, "table_id": t.table_id, "kind": "table",
             "label": f"{t.table_id}  {t.caption or '(no caption)'}"
                      f"  [{len(t.rows)} rows]",
             **t.model_dump(mode="json")}
            for t in table_set.tables
        ] + [
            {"id": t.table_id, "table_id": t.table_id, "kind": "figure",
             "label": f"{t.table_id}  {t.caption or '(no caption)'}"
                      f"  [{len(t.rows)} rows]",
             **t.model_dump(mode="json")}
            for t in figures
        ]
        blocks = [render_display_summary(table_set.tables, figures)]
        if model_id:
            blocks.insert(0, f"  model: {model_id}")
        if table_set.tables:
            blocks.append(render_table_set(table_set.tables))
        if figures:
            blocks.append(render_table_set(figures))
        sidecars = {f"{t.table_id}.csv": to_csv(t)
                    for t in [*table_set.tables, *figures]}
        return await self._reporter.step_or_stop(
            StepStage.TABLE_CAPTURE,
            title="Displays captured",
            subject=subject, subject_kind=kind,
            payload={"tables": displays,
                     "figures": [
                         {"id": t.table_id, "table_id": t.table_id,
                          "label": f"{t.table_id}  {t.caption or '(no caption)'}"
                                   f"  [{len(t.rows)} rows]",
                          **t.model_dump(mode="json")}
                         for t in figures],
                     "model_id": model_id},
            render_blocks=blocks,
            warnings=table_warnings + figure_warnings,
            offers=retry_offers(self._alt),
            selectable="tables",
            sidecars=sidecars,
        )

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
        # Drops already applied on the combined Displays captured pause.


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


def _render_lens(
    lens: ReviewLens, *, chars_total: int = 0, chars_used: int = 0,
    truncated: bool = False, model_id: str = "",
) -> str:
    lines = ["  review focus:"]
    if lens.lens_one_line:
        lines.append(f"    {lens.lens_one_line}")
    for name in ("domain", "population", "comparison"):
        value = getattr(lens, name)
        if value:
            lines.append(f"    {name}: {value}")
    if lens.outcomes:
        lines.append("    outcomes: " + "; ".join(lens.outcomes))
    if chars_total:
        used = chars_used or chars_total
        lines.append("")
        lines.append("  extraction status:")
        line = f"    {chars_total:,} characters extracted (using the first {used:,})"
        if truncated:
            line += " — truncated"
        lines.append(line)
    if model_id:
        lines.append(f"    model: {model_id}")
    return "\n".join(lines)


def _compress_reason(reason: str, limit: int = 90) -> str:
    text = " ".join((reason or "").split())
    if not text:
        return ""
    for sep in (". ", "; "):
        if sep in text:
            text = text.split(sep, 1)[0]
            break
    if len(text) > limit:
        return text[:limit - 1] + "…"
    return text


def _render_hits(hits: list[DisplayHit]) -> str:
    if not hits:
        return "  (no displays listed)"
    lines = [f"  {len(hits)} display(s)"]
    for hit in hits:
        flag = "on" if hit.evidence_chain else "off"
        reason = _compress_reason(hit.reason)
        tail = f"  — {reason}" if reason else ""
        lines.append(
            f"    [{flag}] {hit.display_id:<16} {hit.kind:<12} "
            f"{hit.caption[:60] or '(no caption)'}{tail}")
    return "\n".join(lines)


def _render_origins(labels, dropped: list[str]) -> str:
    if not labels:
        return (
            "  Mark whether each review-table number is copied from a source paper "
            "or computed by the review; the latter stay out of the audit set\n"
            "  (no origin labels)"
        )
    lines = [
        "  Mark whether each review-table number is copied from a source paper "
        "or computed by the review; the latter stay out of the audit set",
        f"  {len(labels)} origin label(s); {len(dropped)} non-source column(s) dropped",
    ]
    for label in labels[:40]:
        lines.append(
            f"    {label.table_id:<16} {label.column_path or '(all)':<28} "
            f"{label.value_source}")
    if len(labels) > 40:
        lines.append(f"    … {len(labels) - 40} more")
    return "\n".join(lines)
