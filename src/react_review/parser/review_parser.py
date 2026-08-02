"""Two-stage review parser: PDF → structure → long rows → ReviewDataItem[]."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from react_review.hitl.events import StepStage, SubjectKind
from react_review.hitl.reporter import StepReporter
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.normalize.doi import normalize_doi
from react_review.normalize.groups import normalize_group
from react_review.parser.table_capture import TableCapturer
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.reason import ReasonRecord
from react_review.schemas.table import CapturedTable, CapturedTableSet
from react_review.dkb import FieldResolver, ResolvedField

logger = structlog.get_logger(__name__)


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


# Unpivot works from the ALREADY CAPTURED table, not from the raw PDF text, and
# in row chunks — so a 60-study review cannot silently lose rows to an output
# token ceiling. Note there is no cohort vocabulary here: whatever the table
# calls its arms is what gets recorded.
_UNPIVOT = """You are converting one table of a systematic review into long format.

Table {table_id} — {caption}
Column labels, by index:
{columns}
Row-identifying columns: {row_axis}

Rows to convert, as [row_index, cells]:
{rows}

Emit ONE entry for EVERY non-empty cell, in EVERY column except {row_axis}.
Country, N, sample size, quality ratings and so on are DATA — do not skip them.

{{"rows": [
  {{"row": 0, "col": 3,
    "column_header": "the column label from the list above, copied verbatim",
    "cohort_label": "which arm/cohort this cell belongs to, IN THE TABLE'S OWN WORDS; empty if the table does not split by cohort",
    "timepoint_label": "the timepoint in the table's own words; empty if none",
    "value": "the cell EXACTLY as printed",
    "unit": "the unit, taken from the column header, the cell text, or a footnote; empty if none is stated"}}
]}}

Rules:
- Copy values verbatim: keep "mean ± SD", "median (IQR)", ranges, "NR", "—", "not reported".
- Take cohort_label and timepoint_label from the header path or the row itself, in the
  table's own wording. Do NOT map them onto any standard vocabulary of your own.
- If a value carries its unit inline (e.g. "80.2 ± 49.0 cm3"), still report that unit.
- Do NOT invent rows, columns or values.

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
    # The verbatim tables everything above was derived from — kept so a reader can
    # check any extracted value back against the table a human approved.
    tables: CapturedTableSet = Field(default_factory=CapturedTableSet)


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


def _row_key_label(row_key: Any) -> str:
    """The study name from a row's identifying columns ("Ahmad 2022")."""
    if isinstance(row_key, dict):
        for value in row_key.values():
            text = str(value or "").strip()
            if text:
                return text
        return ""
    return str(row_key or "").strip()


# A unit printed inside the cell itself ("80.2 ± 49.0 cm3"). Recovering it is
# what lets an ambiguous header like "EFT/ EAT" — which holds thicknesses in one
# study and volumes in another — resolve to the right concept per row.
_INLINE_UNIT = re.compile(
    r"(?<![A-Za-z])(cm\s*3|cm³|mm\s*3|mm³|ml|cc|kg/m\s*2|kg/m²|mm\s*hg|mmol/l|mg/dl|"
    r"cm|mm|kg|g|%|years?|yrs?|months?|days?)\s*$", re.I)


def _inline_unit(value: object) -> str:
    """The unit trailing a cell's text, if it prints one; "" otherwise."""
    m = _INLINE_UNIT.search(str(value or "").strip().rstrip(",;"))
    return m.group(1).replace(" ", "") if m else ""


def _cell_ref(row: dict[str, Any]) -> tuple[int, int] | None:
    """(row, column) coordinates in the captured table, when the model gave them."""
    try:
        return int(row["row"]), int(row["col"])
    except (KeyError, TypeError, ValueError):
        return None


class ReviewParser:
    """Parse a review PDF into a long table of ReviewDataItem."""

    def __init__(
        self,
        backend: LLMBackend,
        resolver: FieldResolver,
        *,
        max_chars: int = 50000,
        reporter: StepReporter | None = None,
        chunk_rows: int = 8,
        keep_tables: set[str] | None = None,
        drop_tables: set[str] | None = None,
        alt_backend: LLMBackend | None = None,
    ) -> None:
        self._backend = backend
        self._resolver = resolver          # the parser holds NO domain knowledge itself
        self._max_chars = max_chars
        # Default reporter never blocks and never writes (library/CI behaviour).
        self._reporter = reporter or StepReporter()
        self._chunk_rows = max(1, chunk_rows)
        self._keep_tables = keep_tables
        self._drop_tables = drop_tables
        self._capturer = TableCapturer(backend, alt_backend=alt_backend)

    async def parse(
        self, pdf_path: Path | str, *, research_context: str = ""
    ) -> ParsedReview:
        full_text = _pdf_text(pdf_path)
        text = full_text[: self._max_chars]
        logger.info("review_parse_start", chars=len(text))

        await self._reporter.step_or_stop(
            StepStage.REVIEW_PDF_LOADED,
            title="Review PDF loaded",
            subject=str(Path(pdf_path).resolve()),
            subject_kind=SubjectKind.REVIEW_PDF,
            payload={"chars_total": len(full_text), "chars_used": len(text),
                     "truncated": len(full_text) > self._max_chars},
            render_blocks=[f"  {len(full_text):,} characters extracted"
                           f" (using the first {len(text):,})"],
            warnings=([f"text truncated to {self._max_chars:,} chars — the data table "
                       "may sit past the cut"] if len(full_text) > self._max_chars else []),
        )

        # 1. Capture the review's tables verbatim and hold the run until a human
        #    has looked at them: nothing downstream means anything if this is wrong.
        table_set, captured_ctx = await self._capturer.capture(
            text, reporter=self._reporter, pdf_path=str(Path(pdf_path).resolve()),
            keep=self._keep_tables, drop=self._drop_tables,
        )
        ctx = captured_ctx or research_context

        # 2. Unpivot each approved table, in row chunks.
        raw_rows: list[dict[str, Any]] = []
        for table in table_set.tables:
            raw_rows.extend(await self._unpivot(table))
        items = await self._postprocess(raw_rows, ctx)

        await self._reporter.step_or_stop(
            StepStage.LONG_FORMAT_ROWS,
            title="Table converted to long format",
            payload={"rows": raw_rows,
                     "items": [i.model_dump(mode="json") for i in items]},
            render_blocks=[self._render_items(items)],
            warnings=self._unpivot_warnings(table_set, raw_rows, items),
        )

        # 3. included-study references, extracted from the reference window (doc tail).
        studies = await self._extract_studies(_refs_window(full_text))

        record = AgentRun(
            agent="parser",
            task={"pdf": str(pdf_path)},
            steps=[
                StepRecord(index=0, thought="capture the review's tables verbatim",
                           tool="llm:table_capture",
                           observation={"tables": [t.table_id for t in table_set.tables],
                                        "dropped": table_set.dropped}),
                StepRecord(index=1, thought="unpivot the captured tables to long rows",
                           tool="llm:unpivot", observation={"n_rows": len(raw_rows)}),
                StepRecord(index=2, thought="extract included-study references",
                           tool="llm:stage_refs", observation={"n_studies": len(studies)}),
            ],
            status="finished",
            final={"n_items": len(items), "n_studies": len(studies)},
        )
        logger.info("review_parse_done", n_rows=len(raw_rows),
                    n_items=len(items), n_studies=len(studies),
                    n_tables=len(table_set.tables))
        return ParsedReview(items=items, record=record, research_context=ctx,
                            studies=studies, tables=table_set)

    async def _unpivot(self, table: CapturedTable) -> list[dict[str, Any]]:
        """Convert one captured table to long rows, a chunk of rows at a time.

        Chunking is what lets a 40-80 study review work at all: one shot at the
        whole table overruns the model's output budget and silently truncates.
        """
        paths = table.column_paths()
        columns = "\n".join(f"  {j}: {p}" for j, p in enumerate(paths))
        row_axis = ", ".join(table.row_axis_columns) or "(none)"
        # Which study a row belongs to is read off the table, not asked of the
        # model: merged study cells make it a deterministic fill-down, and a
        # wrong answer here would silently attribute values to another paper.
        labels = table.row_labels()
        out: list[dict[str, Any]] = []
        for start in range(0, len(table.rows), self._chunk_rows):
            chunk = table.rows[start: start + self._chunk_rows]
            payload = json.dumps(
                [[start + i, row] for i, row in enumerate(chunk)], ensure_ascii=False)
            data = await self._call(_UNPIVOT.format(
                table_id=table.table_id, caption=table.caption or "(no caption)",
                columns=columns, row_axis=row_axis, rows=payload,
            ))
            for r in data.get("rows", []):
                if not isinstance(r, dict):
                    continue
                r["table_id"] = table.table_id
                try:
                    idx = int(r["row"])
                    r["row_key"] = {"study": labels[idx]}
                    # The rest of the row, verbatim, as context for the Resolver.
                    # The parser does not interpret it — a column like "EFT/ EAT"
                    # is ambiguous on its own, and the knowledge base's own rules
                    # decide what a neighbouring "Echocardiography" or "CT" means.
                    r["row_context"] = " ".join(
                        c.strip() for c in table.rows[idx] if c and c.strip())
                except (KeyError, TypeError, ValueError, IndexError):
                    r.setdefault("row_key", {})
                out.append(r)
        return out

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
            raw_name = str(r.get("column_header") or "").strip()
            value = r.get("value")
            if not raw_name or value is None:
                continue
            if _norm_col(raw_name) in _IDENTIFIER_COLUMNS:
                continue  # identifier/dimension column, not a measurement

            unit = str(r.get("unit") or "").strip() or _inline_unit(value)
            cohort_label = str(r.get("cohort_label") or "").strip()
            timepoint_label = str(r.get("timepoint_label") or "").strip()
            table_id = str(r.get("table_id") or "")
            cell_ref = _cell_ref(r)
            study_id = _study_slug(_row_key_label(r.get("row_key")))
            group = normalize_group(cohort_label)
            common = dict(
                study_id=study_id, raw_field_name=raw_name, unit=unit,
                table_id=table_id, cell_ref=cell_ref, column_header=raw_name,
                cohort_label=cohort_label, timepoint_label=timepoint_label,
            )

            # A written placeholder is REPORTED, not dropped: "NR" in an oncology
            # table may mean "median not reached" — a result, not an absence — and
            # deleting the row hides it from the audit entirely. An EMPTY cell is
            # different: in a table whose rows continue a merged study block it is
            # layout, not a statement, so it is skipped as before.
            text = value.strip() if isinstance(value, str) else value
            if isinstance(text, str) and not text:
                continue
            if isinstance(text, str) and text.lower() in _PLACEHOLDER:
                items.append(ReviewDataItem(
                    **common, group=group, field_type="", value=None,
                    resolution_status="unresolved",
                    reasons=[ReasonRecord(
                        code="placeholder_cell", stage="parser",
                        message=f"the table cell reads {value.strip()!r}; "
                                "no value was reported here")],
                ))
                continue

            # Parser applies NO domain knowledge — it asks the DKB Resolver.
            # The value is passed so the resolver can range-check a candidate.
            try:
                rf = await self._resolver.resolve(
                    raw_name, unit=unit, modality=str(r.get("row_context") or ""),
                    research_context=research_context, value=value)
            except Exception as exc:               # never drop on a resolver error
                logger.warning("resolve_failed", raw=raw_name, error=str(exc)[:120])
                rf = ResolvedField(raw_field_name=raw_name)   # → unresolved

            # UNRESOLVED: keep the raw item (field_type null) — the audit marks it
            # not_comparable / needs_review; the concept goes to proposals to learn.
            if rf.status == "unresolved":
                items.append(ReviewDataItem(
                    **common, group=group, field_type="", value=value,
                    resolution_status="unresolved",
                    reasons=[ReasonRecord(
                        code="concept_unresolved", stage="parser",
                        message=f"column {raw_name!r} did not map to a known concept")],
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
                **common, group=group, field_type=ft, value=value,
                resolution_status=("resolved" if rf.status == "authoritative" else rf.status),
            ))
        return items

    # --- rendering + warnings for the long-format checkpoint ---

    @staticmethod
    def _render_items(items: list[ReviewDataItem], limit: int = 25) -> str:
        if not items:
            return "  (no rows produced)"
        lines = [f"  {len(items)} row(s)"]
        for i in items[:limit]:
            concept = i.field_type or f"UNRESOLVED «{i.raw_field_name}»"
            cohort = i.cohort_label or i.group
            lines.append(f"    {i.study_id:<24} {cohort:<12} {concept:<18} {i.value!r}")
        if len(items) > limit:
            lines.append(f"    … {len(items) - limit} more")
        return "\n".join(lines)

    @staticmethod
    def _unpivot_warnings(table_set, raw_rows: list[dict], items: list) -> list[str]:
        out: list[str] = []
        if table_set.tables and not raw_rows:
            out.append("the captured table produced no long rows")
        n_unresolved = sum(1 for i in items if i.resolution_status == "unresolved")
        if n_unresolved:
            out.append(f"{n_unresolved} row(s) have no known concept and cannot be "
                       "compared automatically")
        labels = sorted({i.cohort_label for i in items if i.cohort_label})
        if labels:
            out.append("cohort labels found in the table: " + ", ".join(labels))
        return out
