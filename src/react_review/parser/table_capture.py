"""Stage 1 — capture the review's tables verbatim, show them, and gate on them.

This replaces a prompt that explicitly forbade extracting values ("do NOT extract
table values yet"). The acceptance review reversed that: get the whole table out
FIRST, in its own shape, and put it in front of a human before anything is
interpreted — "only if this table is correctly extracted does anything downstream
have meaning", and if it is wrong, stop and fix it rather than proceeding.

So the prompt asks for a faithful transcription and for the model to say what it
could not read, and the checkpoint offers RETRY (re-transcribe), a per-table drop,
and STOP.

v1/v2 transcribe every data table in the text. v3 transcribes only the displays
Review Extraction already selected, and must not invent forest-plot cells.
"""
from __future__ import annotations

import json
import time

import structlog

from react_review.hitl.events import StepStage, SubjectKind
from react_review.hitl.gate import Decision, require_alt_backend, retry_offers
from react_review.hitl.reporter import StepReporter
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.parser.table_capture_contract import (
    DEFAULT_TABLE_CAPTURE_PROFILE,
    PROMPT_TEMPLATES,
    render_table_capture_prompt,
)
from react_review.parser.table_render import (
    render_display_summary, render_shape_report, render_table_set, to_csv,
)
from react_review.schemas.table import CapturedTable, CapturedTableSet

logger = structlog.get_logger(__name__)


# Backwards-compatible import for tests/tools that observed the old production
# constant.  Frozen A/B stays on v1; new runs default to v3.
_CAPTURE = PROMPT_TEMPLATES[DEFAULT_TABLE_CAPTURE_PROFILE]


class TableCapturer:
    """Transcribe the review's tables, then hold the run at a human checkpoint."""

    def __init__(self, backend: LLMBackend, *, alt_backend: LLMBackend | None = None,
                 prompt_profile: str = DEFAULT_TABLE_CAPTURE_PROFILE) -> None:
        self._backend = backend
        self._alt = alt_backend
        # Validate at construction, before the first model call.
        render_table_capture_prompt(prompt_profile, text="", selected="[]")
        self.prompt_profile = prompt_profile

    async def capture(
        self, text: str, *, reporter: StepReporter, pdf_path: str = "",
        keep: set[str] | None = None, drop: set[str] | None = None,
        selected: list[dict] | None = None,
        defer_gate: bool = False,
        seed: int = 42,
        backend: LLMBackend | None = None,
    ) -> tuple[CapturedTableSet, str]:
        """Return the approved tables and the extracted research context."""
        transcribe = backend or self._backend
        while True:
            started = time.monotonic()
            raw = await self._transcribe(transcribe, text, seed, selected=selected)
            tables = _parse_tables(raw)
            context = str(raw.get("research_context") or "").strip()
            table_set = CapturedTableSet(tables=tables, source_pdf=pdf_path)
            caption = tables[0].caption if tables else ""
            reporter.progress(
                "table", 1, max(1, len(tables)),
                caption=caption, started=started)

            # Non-interactive filtering (--tables / --drop-tables) happens before
            # the checkpoint so what is shown is what will actually be processed.
            if keep:
                table_set = table_set.keep_only(keep, reason="--tables")
            if drop:
                table_set = table_set.keep_only(
                    {t.table_id for t in table_set.tables if t.table_id not in drop},
                    reason="--drop-tables")

            if defer_gate:
                return table_set, context

            decision = await self._gate(
                table_set, reporter, pdf_path,
                model_id=transcribe.model_id, alt=self._alt)
            if decision is Decision.RETRY:
                seed += 1               # same prompt, different sampling
                continue
            if decision is Decision.RETRY_ALT:
                transcribe = require_alt_backend(
                    self._alt, stage="review_table_capture")
                continue

            # The checkpoint may have dropped tables; take what survived.
            event = reporter.last_event
            if event is not None and event.dropped:
                dropped = set(event.dropped)
                kept = {t.table_id for t in table_set.tables if t.table_id not in dropped}
                table_set = table_set.keep_only(kept, reason="dropped at checkpoint")
            return table_set, context

    async def _transcribe(self, backend: LLMBackend, text: str, seed: int,
                          selected: list[dict] | None = None) -> dict:
        try:
            prompt = render_table_capture_prompt(
                self.prompt_profile, text=text,
                selected=json.dumps(selected or [], ensure_ascii=False),
            )
            raw = await backend.complete(prompt, seed=seed)
            return parse_llm_response(raw, backend.model_id, source_text=text)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("table_capture_failed", error=str(exc)[:160])
            return {}

    @staticmethod
    async def _gate(table_set: CapturedTableSet, reporter: StepReporter,
                    pdf_path: str, *, model_id: str = "",
                    alt=None) -> Decision:
        warnings = [w for t in table_set.tables for w in render_shape_report(t)]
        if not table_set.tables:
            warnings.insert(0, "no table was captured — the review's data table was "
                               "not found, or the text layer does not contain it")
        return await reporter.step_or_stop(
            StepStage.TABLE_CAPTURE,
            title="Displays captured",
            subject=pdf_path,
            subject_kind=SubjectKind.REVIEW_PDF,
            payload={"tables": [
                {"id": t.table_id, "table_id": t.table_id,
                 "label": f"{t.table_id}  {t.caption or '(no caption)'}"
                          f"  [{len(t.rows)} rows]",
                 **t.model_dump(mode="json")}
                for t in table_set.tables],
                "figures": [],
                "model_id": model_id},
            render_blocks=[
                *([f"  model: {model_id}"] if model_id else []),
                render_display_summary(table_set.tables, []),
                render_table_set(table_set.tables),
            ],
            warnings=warnings,
            offers=retry_offers(alt),
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
