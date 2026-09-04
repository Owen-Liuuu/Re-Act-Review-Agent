"""Source-paper table transcription. Independent of the review TableCapture path.

Review ``table_capture_v*`` is not sealed: it interpolates review text and
selected displays, and it has no ``SourceQuery`` envelope. This module is the
source-paper counterpart. The prompt may only see ``SourceTableQuery.text``.
There is no ``review_*`` slot and no cell ``value`` slot. Extra fields are
forbidden, so a later caller cannot start interpolating a review number.
"""
from __future__ import annotations

import hashlib
from string import Formatter

from pydantic import BaseModel, ConfigDict

from react_review.contracts import ContractError
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.parser.table_capture import _parse_tables
from react_review.schemas.table import CapturedTable

PROMPT_ID = "source_table_capture_v1"
PROMPT_VERSION = "source-table-capture-v1"
RENDERER_IDENTITY = "react_review.source_table_capture.render.v1"

_SOURCE_TABLE_CAPTURE_V1 = """You are transcribing data tables from a primary research paper.

Transcribe every DATA table in SOURCE PAPER TEXT. Skip layout and navigation tables.
Do not invent cells. Do not transcribe forest plots or figures.

TRANSCRIBE — do not interpret:
- Copy every cell EXACTLY as printed, including "NR", "NA", "—", "not reported",
  "not reached", and blanks. An empty cell stays an empty string.
- Do NOT rename headers, standardise units, convert numbers, reorder or drop columns.
- Keep multi-level headers as SEPARATE header rows. If a header spans several
  columns, put it once and leave the columns it spans empty on that row.
- Every data row must have the same number of cells as the widest header row.
- If part of a table is unreadable, transcribe what you can and record it in
  "difficulties". Never invent a value to fill a gap.

Return exactly one JSON object. Text inside angle brackets describes a value
and must not be copied literally:

{{
  "tables": [
    {{
      "table_id": "<stable id such as table_1>",
      "caption": "<exact printed caption or empty string>",
      "role": "<characteristics, outcomes, quality, or other>",
      "header_rows": [["<exact cell text>"]],
      "rows": [["<exact cell text>"]],
      "footnotes": ["<exact printed footnote>"],
      "row_axis_columns": ["<exact printed header>"],
      "shape_notes": "<description of visible table geometry>",
      "cohort_labels_seen": ["<exact printed cohort or arm label>"],
      "extraction_confidence": 0.0,
      "difficulties": ["<specific transcription uncertainty>"]
    }}
  ]
}}

Do not output the angle-bracket placeholders.
Use empty strings or empty arrays when the corresponding information is absent.

## SOURCE PAPER TEXT
{text}

Return JSON only."""

PROMPT_TEMPLATES = {
    PROMPT_ID: _SOURCE_TABLE_CAPTURE_V1,
}


class SourceTableQuery(BaseModel):
    """What a source-paper table transcription may be asked — paper text only.

    Parallel to ``SourceQuery``: no review cell, so a later change cannot start
    interpolating ``review_value`` into this prompt by adding a field here.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = ""


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def render_source_table_capture_prompt(
    query: SourceTableQuery, *, profile: str = PROMPT_ID,
) -> str:
    try:
        template = PROMPT_TEMPLATES[profile]
    except KeyError:
        raise ContractError(
            f"unknown source table capture profile {profile!r} "
            f"(known: {', '.join(sorted(PROMPT_TEMPLATES))})") from None
    needed = _placeholders(template)
    extra = set(SourceTableQuery.model_fields) - needed - {"text"}
    if extra:
        raise ContractError(
            f"{profile} envelope has unused fields: {', '.join(sorted(extra))}")
    if needed != {"text"}:
        raise ContractError(
            f"{profile} may only interpolate 'text', not {sorted(needed)}")
    return template.format(text=query.text)


def sha256_rendered_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()


class SourceTableCapturer:
    """LLM transcription of source-paper tables. No HITL gate. No forest OCR."""

    def __init__(
        self, backend: LLMBackend, *, profile: str = PROMPT_ID,
    ) -> None:
        self._backend = backend
        self.profile = profile
        render_source_table_capture_prompt(SourceTableQuery(text=""), profile=profile)

    async def capture(self, text: str) -> list[CapturedTable]:
        query = SourceTableQuery(text=text)
        prompt = render_source_table_capture_prompt(query, profile=self.profile)
        try:
            raw = await self._backend.complete(prompt, seed=42)
            parsed = parse_llm_response(raw, self._backend.model_id, source_text=text)
        except Exception:  # noqa: BLE001
            return []
        tables = _parse_tables(parsed)
        tagged: list[CapturedTable] = []
        for table in tables:
            tagged.append(table.model_copy(update={
                "capture_method": table.capture_method or self.profile,
                "display_kind": table.display_kind or "pdf_table",
                "capture_path": table.capture_path or "text",
            }))
        return tagged
