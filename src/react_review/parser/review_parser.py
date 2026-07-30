"""Two-stage review parser: PDF → structure → long rows → ReviewDataItem[]."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.normalize.doi import normalize_doi
from react_review.normalize.groups import normalize_group
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.evidence import ReviewDataItem
from react_review.dkb import FieldResolver, ResolvedField

logger = structlog.get_logger(__name__)


_STAGE1 = """You are reading a systematic review. Find its main DATA-EXTRACTION table
(often "Table 1" / "Characteristics of included studies").

Also state the review's research question in ONE line — population + exposure/intervention
+ outcome — from the title / abstract / objective.

Describe as JSON — do NOT extract table values yet:
{{"table_name": "...", "columns": ["Author", "N", ...],
  "group_handling": "how per-cohort values are shown (e.g. 'each study has a T1DM row and a Control row', or 'single row, no split')",
  "notes": "any footnotes that define groups/units/timepoints",
  "research_context": "one-line research question, e.g. 'epicardial adipose tissue in type 1 diabetes vs healthy controls'"}}

## REVIEW TEXT
{text}

Return JSON only."""


_STAGE_REFS = """You are reading the REFERENCE LIST / included-studies list of a systematic review.
For each INCLUDED primary study, return its citation and DOI when the reference prints one.

{{"studies": [
  {{"citation": "First-author Year, journal …", "doi": "10.xxxx/… or empty string"}}
]}}

Rules:
- One entry per study reference (author + year + journal). Skip editorials / guidelines / non-studies.
- Copy the DOI EXACTLY as printed; use "" when the reference has no DOI. Do NOT invent or guess a DOI.

## REFERENCE TEXT
{refs}

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


class ParsedStudy(BaseModel):
    """An included-study reference extracted from the review's reference list."""

    study_id: str                       # slug, e.g. "ahmad_2022"
    citation: str = ""                  # verbatim reference text
    doi: str = ""                       # normalized DOI, "" when the reference prints none


class ParsedReview(BaseModel):
    items: list[ReviewDataItem] = Field(default_factory=list)
    record: AgentRun
    research_context: str = ""          # LLM-extracted from the review (falls back to the arg)
    studies: list[ParsedStudy] = Field(default_factory=list)   # included-study DOIs


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


def _refs_window(text: str, tail: int = 20000) -> str:
    """Slice the reference-list region — DOIs live at the END, past the 50k body window.

    Anchors on the LAST 'References'/'Bibliography' heading; falls back to the last
    ``tail`` characters so the DOI pass never runs on the (truncated) front matter.
    """
    last = -1
    for m in re.finditer(r"\b(references|bibliography|reference list)\b", text, re.I):
        last = m.start()
    if last >= 0 and len(text) - last >= 200:
        return text[last:]
    return text[-tail:]


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
        resolver: FieldResolver,
        *,
        max_chars: int = 50000,
    ) -> None:
        self._backend = backend
        self._resolver = resolver          # the parser holds NO domain knowledge itself
        self._max_chars = max_chars

    async def parse(
        self, pdf_path: Path | str, *, research_context: str = ""
    ) -> ParsedReview:
        full_text = _pdf_text(pdf_path)
        text = full_text[: self._max_chars]
        logger.info("review_parse_start", chars=len(text))

        structure = await self._call(_STAGE1.format(text=text))
        # research_context: prefer the LLM-extracted question, fall back to the arg.
        ctx = str(structure.get("research_context") or "").strip() or research_context

        raw_rows = (await self._call(_STAGE2.format(
            structure=json.dumps(structure, ensure_ascii=False), text=text,
        ))).get("rows", [])
        items = await self._postprocess(raw_rows, ctx)

        # included-study DOIs, extracted from the reference window (doc tail).
        studies = await self._extract_studies(_refs_window(full_text))

        record = AgentRun(
            agent="parser",
            task={"pdf": str(pdf_path)},
            steps=[
                StepRecord(index=0, thought="identify table structure + research context",
                           tool="llm:stage1_structure", observation=structure),
                StepRecord(index=1, thought="unpivot to long rows",
                           tool="llm:stage2_unpivot",
                           observation={"n_rows": len(raw_rows)}),
                StepRecord(index=2, thought="extract included-study DOIs",
                           tool="llm:stage_refs", observation={"n_studies": len(studies)}),
            ],
            status="finished",
            final={"n_items": len(items), "n_studies": len(studies)},
        )
        logger.info("review_parse_done", n_rows=len(raw_rows),
                    n_items=len(items), n_studies=len(studies))
        return ParsedReview(items=items, record=record, research_context=ctx, studies=studies)

    async def _call(self, prompt: str) -> dict[str, Any]:
        try:
            raw = await self._backend.complete(prompt)
            return parse_llm_response(raw, self._backend.model_id)
        except Exception as exc:
            logger.warning("review_parse_stage_failed", error=str(exc)[:160])
            return {}

    async def _extract_studies(self, refs_text: str) -> list[ParsedStudy]:
        """LLM-extract included-study {citation, doi} from the reference window.

        Raw extraction only: study_id is the deterministic slug of the citation and
        the DOI is normalized; matching these to the data rows is a downstream step.
        """
        if not refs_text.strip():
            return []
        data = await self._call(_STAGE_REFS.format(refs=refs_text))
        studies: list[ParsedStudy] = []
        for r in data.get("studies", []):
            if not isinstance(r, dict):
                continue
            citation = str(r.get("citation") or "").strip()
            if not citation:
                continue
            studies.append(ParsedStudy(
                study_id=_study_slug(citation), citation=citation,
                doi=normalize_doi(r.get("doi")),
            ))
        return studies

    async def _postprocess(
        self, raw_rows: list[dict[str, Any]], research_context: str
    ) -> list[ReviewDataItem]:
        items: list[ReviewDataItem] = []
        seen_study_level: set[tuple[str, str]] = set()
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
            # Parser applies NO domain knowledge — it asks the DKB Resolver.
            # The value is passed so the resolver can range-check a candidate.
            try:
                rf = await self._resolver.resolve(
                    raw_name, unit=unit, research_context=research_context, value=value)
            except Exception as exc:               # never drop on a resolver error
                logger.warning("resolve_failed", raw=raw_name, error=str(exc)[:120])
                rf = ResolvedField(raw_field_name=raw_name)   # → unresolved

            study_id = _study_slug(str(r.get("study") or ""))
            group = normalize_group(str(r.get("group") or ""))

            # UNRESOLVED: keep the raw item (field_type null) — the audit marks it
            # not_comparable / needs_review; the concept goes to proposals to learn.
            if rf.status == "unresolved":
                items.append(ReviewDataItem(
                    study_id=study_id, group=group, field_type="",
                    raw_field_name=raw_name, value=value, unit=unit,
                    resolution_status="unresolved",
                ))
                continue

            ft = rf.field_type
            # Parser APPLIES scope, using the knowledge the Resolver provided.
            if rf.scope == "study":
                group = "-"
                if (study_id, ft) in seen_study_level:
                    continue
                seen_study_level.add((study_id, ft))
            items.append(ReviewDataItem(
                study_id=study_id, group=group, field_type=ft,
                raw_field_name=raw_name, value=value, unit=unit,
                resolution_status=("resolved" if rf.status == "authoritative" else rf.status),
            ))
        return items
