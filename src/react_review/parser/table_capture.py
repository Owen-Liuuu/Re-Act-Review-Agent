"""Stage 1 — capture the review's tables verbatim, show them, and gate on them.

This replaces a prompt that explicitly forbade extracting values ("do NOT extract
table values yet"). The acceptance review reversed that: get the whole table out
FIRST, in its own shape, and put it in front of a human before anything is
interpreted — "only if this table is correctly extracted does anything downstream
have meaning", and if it is wrong, stop and fix it rather than proceeding.

So the prompt asks for a faithful transcription and for the model to say what it
could not read, and the checkpoint offers RETRY (re-transcribe), a per-table drop,
and STOP.
"""
from __future__ import annotations

import structlog

from react_review.hitl.events import StepStage, SubjectKind
from react_review.hitl.gate import Decision
from react_review.hitl.reporter import StepReporter
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.parser.table_render import render_shape_report, render_table_set, to_csv
from react_review.schemas.table import CapturedTable, CapturedTableSet

logger = structlog.get_logger(__name__)


_CAPTURE = """You are a systematic-review methodologist transcribing the tables of a review
so a colleague can check them against the original PDF.

Transcribe every DATA table: the characteristics-of-included-studies table, any
outcome/effect table, any risk-of-bias table. Skip pure layout or navigation tables.

TRANSCRIBE — do not interpret:
- Copy every cell EXACTLY as printed, including "NR", "NA", "—", "not reported",
  "not reached", and blanks. An empty cell stays an empty string.
- Do NOT rename headers, do NOT standardise units, do NOT convert numbers, do NOT
  reorder or drop columns — including columns whose meaning is unclear to you.
- Keep multi-level headers as SEPARATE header rows. If a header spans several
  columns, put it once and leave the columns it spans empty on that row.
- Every data row must have the same number of cells as the widest header row.
- If part of a table is unreadable, still transcribe what you can and say what
  went wrong in "difficulties". Never invent a value to fill a gap.

Also state the review's research question in one line.

{{"research_context": "one line: population + exposure/intervention + outcome",
  "tables": [
    {{"table_id": "table_1",
      "caption": "the table's printed caption",
      "role": "characteristics | outcomes | quality | other",
      "header_rows": [["Study","Country","EAT",""],["","","T1DM","Control"]],
      "rows": [["Ahmad 2022","Egypt","6.60 ± 0.71","3.83 ± 0.35"]],
      "footnotes": ["values are mean ± SD unless stated"],
      "row_axis_columns": ["Study"],
      "shape_notes": "one row per study; the cohort split is a column pair",
      "cohort_labels_seen": ["T1DM","Control"],
      "extraction_confidence": 0.0,
      "difficulties": ["the last column was cut off in the text layer"]}}
  ]}}

## REVIEW TEXT
{text}

Return JSON only."""


class TableCapturer:
    """Transcribe the review's tables, then hold the run at a human checkpoint."""

    def __init__(self, backend: LLMBackend, *, alt_backend: LLMBackend | None = None) -> None:
        self._backend = backend
        self._alt = alt_backend

    async def capture(
        self, text: str, *, reporter: StepReporter, pdf_path: str = "",
        keep: set[str] | None = None, drop: set[str] | None = None,
    ) -> tuple[CapturedTableSet, str]:
        """Return the approved tables and the extracted research context."""
        backend = self._backend
        seed = 42
        while True:
            raw = await self._transcribe(backend, text, seed)
            tables = _parse_tables(raw)
            context = str(raw.get("research_context") or "").strip()
            table_set = CapturedTableSet(tables=tables, source_pdf=pdf_path)

            # Non-interactive filtering (--tables / --drop-tables) happens before
            # the checkpoint so what is shown is what will actually be processed.
            if keep:
                table_set = table_set.keep_only(keep, reason="--tables")
            if drop:
                table_set = table_set.keep_only(
                    {t.table_id for t in table_set.tables if t.table_id not in drop},
                    reason="--drop-tables")

            decision = await self._gate(table_set, reporter, pdf_path)
            if decision is Decision.RETRY:
                seed += 1               # same prompt, different sampling
                continue
            if decision is Decision.RETRY_ALT and self._alt is not None:
                backend = self._alt
                continue

            # The checkpoint may have dropped tables; take what survived.
            event = reporter.last_event
            if event is not None and event.dropped:
                kept = {t["table_id"] for t in event.selectable_items()}
                table_set = table_set.keep_only(kept, reason="dropped at checkpoint")
            return table_set, context

    async def _transcribe(self, backend: LLMBackend, text: str, seed: int) -> dict:
        try:
            raw = await backend.complete(_CAPTURE.format(text=text), seed=seed)
            return parse_llm_response(raw, backend.model_id)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("table_capture_failed", error=str(exc)[:160])
            return {}

    @staticmethod
    async def _gate(table_set: CapturedTableSet, reporter: StepReporter,
                    pdf_path: str) -> Decision:
        warnings = [w for t in table_set.tables for w in render_shape_report(t)]
        if not table_set.tables:
            warnings.insert(0, "no table was captured — the review's data table was "
                               "not found, or the text layer does not contain it")
        return await reporter.step_or_stop(
            StepStage.TABLE_CAPTURE,
            title="Main data table captured",
            subject=pdf_path,
            subject_kind=SubjectKind.REVIEW_PDF,
            payload={"tables": [
                {"id": t.table_id, "table_id": t.table_id,
                 "label": f"{t.table_id}  {t.caption or '(no caption)'}"
                          f"  [{len(t.rows)} rows]",
                 **t.model_dump(mode="json")}
                for t in table_set.tables]},
            render_blocks=[render_table_set(table_set.tables)],
            warnings=warnings,
            offers=["retry", "retry_alt"],
            selectable="tables",
            sidecars={f"{t.table_id}.csv": to_csv(t) for t in table_set.tables},
        )


def _parse_tables(raw: dict) -> list[CapturedTable]:
    """Build CapturedTable objects, skipping anything malformed rather than failing."""
    out: list[CapturedTable] = []
    for i, body in enumerate(raw.get("tables") or [], start=1):
        if not isinstance(body, dict):
            continue
        try:
            body.setdefault("table_id", f"table_{i}")
            body["header_rows"] = [[str(c) if c is not None else "" for c in row]
                                   for row in (body.get("header_rows") or [])]
            body["rows"] = [[str(c) if c is not None else "" for c in row]
                            for row in (body.get("rows") or [])]
            out.append(CapturedTable(**{k: v for k, v in body.items()
                                        if k in CapturedTable.model_fields}))
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("captured_table_malformed", index=i, error=str(exc)[:120])
    return out
