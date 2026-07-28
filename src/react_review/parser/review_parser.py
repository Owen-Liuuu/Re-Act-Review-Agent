"""Two-stage review parser: PDF → structure → long rows → ReviewDataItem[]."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.normalize.groups import normalize_group
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.evidence import ReviewDataItem
from react_review.tools.models import NormalizeInput
from react_review.tools.normalize import NormalizeFieldTool

logger = structlog.get_logger(__name__)


_STAGE1 = """You are reading a systematic review. Find its main DATA-EXTRACTION table
(often "Table 1" / "Characteristics of included studies").

Describe its structure as JSON — do NOT extract values yet:
{{"table_name": "...", "columns": ["Author", "N", ...],
  "group_handling": "how per-cohort values are shown (e.g. 'each study has a T1DM row and a Control row', or 'single row, no split')",
  "notes": "any footnotes that define groups/units/timepoints"}}

## REVIEW TEXT
{text}

Return JSON only."""


_STAGE2 = """You are extracting the data-extraction table of a systematic review into LONG format.

Table structure (from a prior pass):
{structure}

Emit ONE row per data cell. For a study with T1DM and Control cohorts, emit
separate rows per cohort. Copy values EXACTLY as printed (keep "mean ± SD").

{{"rows": [
  {{"study": "First-author Year", "group": "T1DM|Control|all",
    "raw_field_name": "the column header verbatim", "value": "raw cell value", "unit": "unit if any"}}
]}}

Rules:
- ``group`` = the cohort the value belongs to; use "all" when the study reports a single combined value.
- Do NOT emit rows for IDENTIFIER/label columns (Author, Study, Reference, Group, Cohort, Subgroup). Those identify the row — "Group" only sets ``group`` — they are not measurements.
- Do NOT invent studies, columns, or values. Skip blank cells.

## REVIEW TEXT
{text}

Return JSON only."""


class ParsedReview(BaseModel):
    items: list[ReviewDataItem] = Field(default_factory=list)
    record: AgentRun
    research_context: str = ""


def _pdf_text(pdf_path: Path | str) -> str:
    import fitz  # PyMuPDF

    from react_review.normalize.text import clean_pdf_text

    doc = fitz.open(str(pdf_path))
    try:
        return clean_pdf_text("\n\n".join(doc[i].get_text() for i in range(len(doc))))
    finally:
        doc.close()


def _study_slug(raw: str) -> str:
    """"Ahmad et al. [2022]" -> "ahmad_2022" (first alpha word + first 4-digit year)."""
    s = (raw or "").strip()
    word = re.search(r"[A-Za-z][A-Za-z\-]+", s)
    year = re.search(r"(19|20)\d{2}", s)
    parts = [p for p in (word.group(0).lower() if word else "", year.group(0) if year else "") if p]
    return "_".join(parts) or re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "study"


_PLACEHOLDER = {"", "null", "none", "n/a", "na", "-", "—", "nr", "not reported"}

# Identifier / dimension columns describe a row; they are NOT measurements and
# must not leak in as spurious field_types (Author→author, Group→study_group).
_IDENTIFIER_COLUMNS = {
    "author", "authors", "first author", "study", "studies", "citation",
    "reference", "ref", "ref no", "reference number", "reference no",
    "group", "groups", "cohort", "subgroup", "arm", "study group",
}


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


class ReviewParser:
    """Parse a review PDF into a long table of ReviewDataItem."""

    def __init__(
        self,
        backend: LLMBackend,
        normalize_field: NormalizeFieldTool,
        *,
        max_chars: int = 50000,
    ) -> None:
        self._backend = backend
        self._normalize = normalize_field
        self._max_chars = max_chars

    async def parse(
        self, pdf_path: Path | str, *, research_context: str = ""
    ) -> ParsedReview:
        text = _pdf_text(pdf_path)[: self._max_chars]
        logger.info("review_parse_start", chars=len(text))

        structure = await self._call(_STAGE1.format(text=text))
        raw_rows = (await self._call(_STAGE2.format(
            structure=json.dumps(structure, ensure_ascii=False), text=text,
        ))).get("rows", [])

        items = await self._postprocess(raw_rows, research_context)
        record = AgentRun(
            agent="parser",
            task={"pdf": str(pdf_path)},
            steps=[
                StepRecord(index=0, thought="identify table structure",
                           tool="llm:stage1_structure", observation=structure),
                StepRecord(index=1, thought="unpivot to long rows",
                           tool="llm:stage2_unpivot",
                           observation={"n_rows": len(raw_rows)}),
            ],
            status="finished",
            final={"n_items": len(items)},
        )
        logger.info("review_parse_done", n_rows=len(raw_rows), n_items=len(items))
        return ParsedReview(items=items, record=record, research_context=research_context)

    async def _call(self, prompt: str) -> dict[str, Any]:
        try:
            raw = await self._backend.complete(prompt)
            return parse_llm_response(raw, self._backend.model_id)
        except Exception as exc:
            logger.warning("review_parse_stage_failed", error=str(exc)[:160])
            return {}

    async def _postprocess(
        self, raw_rows: list[dict[str, Any]], research_context: str
    ) -> list[ReviewDataItem]:
        items: list[ReviewDataItem] = []
        for r in raw_rows:
            if not isinstance(r, dict):
                continue
            raw_name = str(r.get("raw_field_name") or "").strip()
            value = r.get("value")
            if not raw_name or value is None:
                continue
            if _norm_col(raw_name) in _IDENTIFIER_COLUMNS:
                continue  # identifier/dimension column, not a measurement
            if isinstance(value, str) and value.strip().lower() in _PLACEHOLDER:
                continue
            unit = str(r.get("unit") or "").strip()
            try:
                ft = (await self._normalize.run(NormalizeInput(
                    raw_field_name=raw_name, unit=unit,
                    research_context=research_context,
                ))).field_type
            except Exception:
                continue  # unresolvable field name (no vocab hit, no backend) → skip
            items.append(ReviewDataItem(
                study_id=_study_slug(str(r.get("study") or "")),
                group=normalize_group(str(r.get("group") or "")),
                field_type=ft,
                raw_field_name=raw_name,
                value=value,
                unit=unit,
            ))
        return items
