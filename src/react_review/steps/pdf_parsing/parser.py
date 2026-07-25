"""PDF parser: produces a fully-understood ``StudentReviewInput`` from a PDF.

The parser runs four focused LLM sub-tasks in parallel, each responsible for
one slice of the student's submission, then merges the results through
Pydantic validation:

    A. search_strategy   — verbatim Methods text + per-database query
                           extraction / translation + reported counts.
    B. selected_papers   — the included-studies list (with DOI validation).
    C. submitted_tables  — the per-paper data-extraction values.
    D. evidence_schema   — the rich schema (type, tolerance, metadata flag,
                           synonym hints) describing each table column.

Splitting into focused prompts keeps each LLM call shorter, lets each
sub-task be retried independently, and avoids one task's failure cascading
into the whole ingestion. ``asyncio.gather`` runs the four tasks
concurrently, so wall-clock cost is roughly one LLM round-trip rather than
four.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import structlog

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.pipeline.schemas import (
    DatabaseQuery,
    EvidenceFieldSchema,
    SearchStrategy,
    StudentReviewInput,
)
from react_review.steps.data_extraction.schemas import ExtractedField, ExtractedTable
from react_review.steps.paper_verification.schemas import ReferenceEntry

from .defaults import resolve_field_thresholds

logger = structlog.get_logger(__name__)


# ======================================================================
# Sub-prompts — each focused on a single slice of the student submission.
# ======================================================================


_PROMPT_SEARCH_STRATEGY = """You are reading a systematic review paper and extracting its search strategy.

Identify everything below from the paper text. If a field is not reported, use null
or an empty array; do NOT invent values.

Output a single JSON object with exactly these fields:

```json
{{
  "research_context":      "one-sentence description of what the review studies (e.g. 'EAT thickness in T1DM patients vs healthy controls')",
  "raw_text":              "the verbatim paragraph(s) from the Methods section describing the search strategy. Copy the original wording, including any database list and search terms.",
  "search_date":           "the date the search was conducted, ISO format if possible (e.g. '2023-06-15'), or empty string",
  "filters":               {{"language": "English", "study_types": "RCT, observational"}},
  "reported_total_count":  110,
  "reported_per_db_count": {{"PubMed": 12, "Embase": 45, "Cochrane": 2, "Web of Science": 12, "CINAHL": 39}},
  "extracted_per_database": [
    {{
      "database": "PubMed",
      "query":    "the database-specific query the student used. If the student gives the verbatim query string, copy it. Otherwise translate the described keywords into PubMed syntax (use [tiab], [MeSH], Boolean OR/AND).",
      "source":   "verbatim" or "llm_translated",
      "notes":    "optional caveat (e.g. 'translated from generic keyword list')"
    }}
  ]
}}
```

Rules:
- ``raw_text`` must be a verbatim copy from the paper, not paraphrased.
- For each database the review claims to have searched, include one entry in ``extracted_per_database``.
- Mark ``source`` as "verbatim" only if the paper provides the database-specific query string. Otherwise mark "llm_translated" and translate the generic keywords into that database's expected syntax.
- ``filters`` may be empty if not reported.

## PAPER TEXT

{paper_text}

Return JSON only — no commentary, no markdown fences."""


_PROMPT_SELECTED_PAPERS = """You are reading a systematic review and listing the FINAL included studies.

These are the papers the authors actually included after screening — usually
shown in a "Characteristics of included studies" table or described in the
Results section. Do not include papers that were screened-out.

For each included study, extract the following:

```json
{{
  "selected_papers": [
    {{
      "title":   "Paper title",
      "authors": ["Lastname A", "Lastname B"],
      "journal": "Journal name (or empty string)",
      "year":    2023,
      "doi":     "10.xxxx/xxxxx (or empty string)"
    }}
  ]
}}
```

Rules:
- Return only papers that the review authors INCLUDED (not the broader screened set).
- For DOI: extract the value the paper actually shows in its References / Bibliography.
  Do NOT guess or fabricate a DOI; use empty string if you cannot find one.
- 3-4 leading authors per paper are sufficient.

## PAPER TEXT

{paper_text}

Return JSON only — no commentary, no markdown fences."""


_PROMPT_SUBMITTED_TABLES = """You are reading a systematic review and extracting the DATA EXTRACTION TABLE the authors built — typically called "Table 1" or "Characteristics of included studies".

This table contains, for each included study, the values the review authors extracted (sample size, mean age, intervention details, outcomes, etc.).

Output one entry per included paper, copying values exactly as they appear in the table. Use null for cells that are blank in the original.

```json
{{
  "submitted_tables": [
    {{
      "paper_id":     "must match the DOI of the corresponding included paper, or empty string",
      "extractor_id": "student",
      "fields": [
        {{"field_name": "exact column name as written in the table", "value": "raw value from the cell"}},
        {{"field_name": "another column",                              "value": 50}}
      ]
    }}
  ]
}}
```

Rules:
- ``field_name`` MUST be the verbatim column header (e.g. "N", "EFT/EAT", "Method") — do NOT rename or normalise.
- ``value`` is a number when the cell is numeric, a string otherwise.
- For multi-row cells (e.g. "12.5 ± 3.2 (n=20)"), keep the full string as-is.
- For empty cells, use null.

## PAPER TEXT

{paper_text}

Return JSON only — no commentary, no markdown fences."""


_PROMPT_EVIDENCE_SCHEMA = """You are reading the data extraction table of a systematic review and producing a SCHEMA describing each column.

For each column header in the student's Table 1, identify:

  * student_field_name : the column name verbatim (used to align student vs AI extraction in Step 4).
  * canonical_concept  : a short snake_case abstract concept name describing what the column represents (e.g. ``sample_size``, ``age_mean``, ``measurement_tool``, ``intervention_outcome``). This is the concept the AI extractor will look for in source papers — so use the standard biomedical name even if the student abbreviated.
  * type               : one of "numeric", "text", "categorical", "author", "year", "doi".
  * is_metadata        : true ONLY for author / title / year / doi-like identifier columns (these are validated separately and excluded from the data-evidence comparison).
  * synonym_check      : true for fields where wording varies a lot across papers — typically ``measurement_tool``, ``study_design``, ``intervention_type``. False for plain numeric or country fields.
  * description        : optional short note for the report (e.g. "Number of participants in the study").

Output:

```json
{{
  "evidence_schema": [
    {{
      "student_field_name": "N",
      "canonical_concept":  "sample_size",
      "type":               "numeric",
      "is_metadata":        false,
      "synonym_check":      false,
      "description":        "Number of participants in the study"
    }}
  ]
}}
```

Rules:
- One entry per column header in the student's data extraction table.
- DO NOT invent columns the student did not include.
- Mark ``is_metadata`` = true for author / title / year / doi columns; the rest are evidence fields.
- Use the same ``student_field_name`` strings that appear in submitted_tables — they are the join keys downstream.

## PAPER TEXT

{paper_text}

Return JSON only — no commentary, no markdown fences."""


# ======================================================================
# Parser
# ======================================================================


class PDFParser:
    """Parses a systematic review PDF into a fully-understood StudentReviewInput.

    Splits ingestion into four parallel LLM sub-tasks (search strategy /
    selected papers / submitted tables / evidence schema). Each sub-task
    is a focused prompt that runs concurrently. Results are merged with
    Pydantic validation; defaults from ``defaults.py`` fill in any
    thresholds the LLM omits (per the project decision to prefer a
    deterministic default table over LLM-suggested tolerances).

    Args:
        llm_backend: LLM backend used by every sub-task.
        max_chars: Truncation length for the paper text passed to the LLM.
            Tuned so that the four sub-prompts each fit under typical
            context windows (default 50,000 characters).
    """

    def __init__(self, llm_backend: LLMBackend, max_chars: int = 50000) -> None:
        self._backend = llm_backend
        self._max_chars = max_chars

    async def parse(self, pdf_path: Path) -> StudentReviewInput:
        """Parse a PDF and return a populated StudentReviewInput.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            ImportError: If PyMuPDF is not installed.
            LLMError: If any sub-task's LLM response cannot be parsed.
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("pdf_parse_start", path=str(pdf_path))

        # ---- A. PDF → plain text -----------------------------------------
        paper_text = self._extract_text(pdf_path)
        truncated_text = paper_text[: self._max_chars]
        logger.info(
            "pdf_text_extracted",
            chars=len(paper_text),
            truncated=len(paper_text) > self._max_chars,
        )

        # ---- B. Real-DOI whitelist (regex) ------------------------------
        # Used to drop hallucinated DOIs from the LLM's selected_papers.
        real_dois = self._extract_dois(paper_text)
        logger.info("pdf_dois_extracted", n_dois=len(real_dois))

        # ---- C. Four parallel LLM sub-tasks -----------------------------
        results = await asyncio.gather(
            self._run_subtask("search_strategy", _PROMPT_SEARCH_STRATEGY, truncated_text),
            self._run_subtask("selected_papers", _PROMPT_SELECTED_PAPERS, truncated_text),
            self._run_subtask("submitted_tables", _PROMPT_SUBMITTED_TABLES, truncated_text),
            self._run_subtask("evidence_schema", _PROMPT_EVIDENCE_SCHEMA, truncated_text),
            return_exceptions=False,
        )
        strategy_json, papers_json, tables_json, schema_json = results

        # ---- D. Merge into StudentReviewInput ---------------------------
        student_input = self._merge(
            strategy_json=strategy_json,
            papers_json=papers_json,
            tables_json=tables_json,
            schema_json=schema_json,
            pdf_path=pdf_path,
            real_dois=real_dois,
            review_full_text=paper_text,
        )

        logger.info(
            "pdf_parse_done",
            n_selected=len(student_input.selected_papers),
            n_tables=len(student_input.submitted_tables),
            n_schema=len(student_input.evidence_schema),
            n_databases=len(student_input.search_strategy.extracted_per_database),
        )
        return student_input

    # ------------------------------------------------------------------
    # Sub-task runner
    # ------------------------------------------------------------------

    async def _run_subtask(
        self, name: str, prompt_template: str, paper_text: str
    ) -> dict[str, Any]:
        """Run one sub-task: format prompt, call LLM, parse JSON response."""
        prompt = prompt_template.format(paper_text=paper_text)
        try:
            raw = await self._backend.complete(prompt)
        except Exception as exc:
            logger.error("pdf_subtask_llm_error", subtask=name, error=str(exc))
            return {}
        try:
            return parse_llm_response(raw, self._backend.model_id)
        except Exception as exc:
            logger.error("pdf_subtask_parse_error", subtask=name, error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # Merge: turn the four sub-task JSONs into a single StudentReviewInput
    # ------------------------------------------------------------------

    @classmethod
    def _merge(
        cls,
        strategy_json: dict[str, Any],
        papers_json: dict[str, Any],
        tables_json: dict[str, Any],
        schema_json: dict[str, Any],
        pdf_path: Path,
        real_dois: list[str],
        review_full_text: str,
    ) -> StudentReviewInput:
        """Combine the four sub-task outputs and build a StudentReviewInput."""

        # --- Search strategy ---
        search_strategy = cls._build_search_strategy(strategy_json)
        research_context = (strategy_json.get("research_context") or "").strip()

        # --- Selected papers (with DOI validation) ---
        selected_papers, doi_rewrite = cls._build_selected_papers(
            papers_json.get("selected_papers") or [],
            real_dois=real_dois,
        )

        # --- Submitted tables (apply DOI rewrite if LLM hallucinated paper_id) ---
        submitted_tables = cls._build_submitted_tables(
            tables_json.get("submitted_tables") or [],
            doi_rewrite=doi_rewrite,
        )

        # --- Evidence schema (with default tolerance fill-in) ---
        evidence_schema = cls._build_evidence_schema(
            schema_json.get("evidence_schema") or []
        )

        return StudentReviewInput(
            student_id=pdf_path.stem,
            review_title=papers_json.get("review_title") or pdf_path.stem,
            research_context=research_context,
            search_strategy=search_strategy,
            selected_papers=selected_papers,
            evidence_schema=evidence_schema,
            submitted_tables=submitted_tables,
            review_full_text=review_full_text,
        )

    # ------------------------------------------------------------------
    # Sub-merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_search_strategy(payload: dict[str, Any]) -> SearchStrategy:
        """Build a SearchStrategy object from the search-strategy sub-task JSON."""
        per_db_raw = payload.get("extracted_per_database") or []
        per_db: list[DatabaseQuery] = []
        for entry in per_db_raw:
            if not isinstance(entry, dict):
                continue
            db = (entry.get("database") or "").strip()
            query = (entry.get("query") or "").strip()
            if not db:
                continue
            source = entry.get("source") or "llm_translated"
            if source not in ("verbatim", "llm_translated"):
                source = "llm_translated"
            per_db.append(
                DatabaseQuery(
                    database=db,
                    query=query,
                    source=source,  # type: ignore[arg-type]
                    notes=entry.get("notes") or "",
                )
            )

        # Coerce filter values to strings (the LLM sometimes returns lists).
        filters_raw = payload.get("filters") or {}
        filters: dict[str, str] = {}
        if isinstance(filters_raw, dict):
            for k, v in filters_raw.items():
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    v = ", ".join(str(x) for x in v)
                filters[str(k)] = str(v)

        # Coerce per-DB counts to ints.
        per_db_count_raw = payload.get("reported_per_db_count") or {}
        per_db_count: dict[str, int] = {}
        if isinstance(per_db_count_raw, dict):
            for k, v in per_db_count_raw.items():
                try:
                    per_db_count[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue

        total_raw = payload.get("reported_total_count")
        try:
            reported_total = int(total_raw) if total_raw is not None else None
        except (TypeError, ValueError):
            reported_total = None

        return SearchStrategy(
            raw_text=(payload.get("raw_text") or "").strip(),
            extracted_per_database=per_db,
            search_date=(payload.get("search_date") or "").strip(),
            filters=filters,
            reported_total_count=reported_total,
            reported_per_db_count=per_db_count,
        )

    @classmethod
    def _build_selected_papers(
        cls,
        papers_raw: list[dict[str, Any]],
        real_dois: list[str],
    ) -> tuple[list[ReferenceEntry], dict[str, str]]:
        """Validate DOIs against the real-DOI whitelist; record any rewrites."""
        doi_rewrite: dict[str, str] = {}
        result: list[ReferenceEntry] = []
        for p in papers_raw:
            if not isinstance(p, dict):
                continue
            llm_doi = (p.get("doi") or "").strip()
            title = (p.get("title") or "").strip()
            validated_doi = cls._resolve_doi(llm_doi, title, real_dois)
            if llm_doi and llm_doi != validated_doi:
                logger.warning(
                    "pdf_doi_dropped",
                    title=title[:60],
                    llm_doi=llm_doi,
                    reason="not in PDF whitelist — likely LLM hallucination",
                )
                doi_rewrite[llm_doi] = validated_doi
            year_raw = p.get("year")
            try:
                year = int(year_raw) if year_raw is not None else None
            except (TypeError, ValueError):
                year = None
            result.append(
                ReferenceEntry(
                    title=title,
                    authors=p.get("authors") or [],
                    journal=p.get("journal") or "",
                    year=year,
                    doi=validated_doi,
                )
            )
        return result, doi_rewrite

    @staticmethod
    def _build_submitted_tables(
        tables_raw: list[dict[str, Any]],
        doi_rewrite: dict[str, str],
    ) -> list[ExtractedTable]:
        """Coerce LLM JSON into ExtractedTable objects, applying DOI rewrites."""
        tables: list[ExtractedTable] = []
        for t in tables_raw:
            if not isinstance(t, dict):
                continue
            fields_raw = t.get("fields") or []
            fields: list[ExtractedField] = []
            for f in fields_raw:
                if not isinstance(f, dict):
                    continue
                fields.append(
                    ExtractedField(
                        field_name=(f.get("field_name") or "unknown").strip(),
                        value=f.get("value"),
                        evidence=(f.get("evidence") or "").strip(),
                        confidence=float(f.get("confidence") or 1.0),
                    )
                )
            paper_id = (t.get("paper_id") or "").strip()
            if paper_id in doi_rewrite:
                paper_id = doi_rewrite[paper_id]
            tables.append(
                ExtractedTable(
                    paper_id=paper_id,
                    fields=fields,
                    extractor_id=(t.get("extractor_id") or "student").strip() or "student",
                )
            )
        return tables

    @staticmethod
    def _build_evidence_schema(
        schema_raw: list[dict[str, Any]],
    ) -> list[EvidenceFieldSchema]:
        """Convert raw schema entries to EvidenceFieldSchema with default tolerances.

        Per project decision #7C, the default tolerance table is the primary
        source of truth; LLM-supplied thresholds are only used when no default
        exists for that ``canonical_concept``.
        """
        schema: list[EvidenceFieldSchema] = []
        for s in schema_raw:
            if not isinstance(s, dict):
                continue
            student_field_name = (s.get("student_field_name") or "").strip()
            canonical_concept = (s.get("canonical_concept") or "").strip()
            type_raw = (s.get("type") or "text").strip().lower()
            if type_raw not in (
                "numeric", "text", "categorical", "author", "year", "doi"
            ):
                type_raw = "text"
            if not student_field_name:
                continue  # cannot align in Step 4 without a join key
            if not canonical_concept:
                # Fall back to a slugified field name so downstream extractor
                # has at least something to work with.
                canonical_concept = re.sub(r"[^a-z0-9]+", "_", student_field_name.lower()).strip("_")
                canonical_concept = canonical_concept or "unknown_field"

            llm_match = s.get("threshold_match")
            llm_partial = s.get("threshold_partial")
            llm_match_f = float(llm_match) if isinstance(llm_match, (int, float)) else None
            llm_partial_f = float(llm_partial) if isinstance(llm_partial, (int, float)) else None
            threshold_match, threshold_partial = resolve_field_thresholds(
                canonical_concept=canonical_concept,
                type_=type_raw,
                llm_match=llm_match_f,
                llm_partial=llm_partial_f,
            )

            schema.append(
                EvidenceFieldSchema(
                    student_field_name=student_field_name,
                    canonical_concept=canonical_concept,
                    type=type_raw,  # type: ignore[arg-type]
                    threshold_match=threshold_match,
                    threshold_partial=threshold_partial,
                    is_metadata=bool(s.get("is_metadata") or False),
                    synonym_check=bool(s.get("synonym_check") or False),
                    description=(s.get("description") or "").strip(),
                )
            )
        return schema

    # ------------------------------------------------------------------
    # PDF text extraction + DOI whitelist
    # ------------------------------------------------------------------

    # Matches DOIs like "10.1007/s00246-021-02811-x".
    _DOI_REGEX = re.compile(r"\b10\.\d{4,9}/[^\s\"<>\]]+", re.IGNORECASE)

    @classmethod
    def _extract_dois(cls, text: str) -> list[str]:
        """Extract unique real DOIs from the paper text using a regex.

        PyMuPDF often inserts line breaks, soft hyphens, and zero-width
        chars in the middle of DOIs (e.g. "10.1016/j.numecd.\\n2013.11.001"),
        so we normalise the text first.
        """
        cleaned = text.replace("­", "").replace("​", "").replace("‌", "")
        # Rejoin lines that were split mid-token.
        normalized = re.sub(r"(?<=\S)\n(?=\S)", "", cleaned)
        seen: set[str] = set()
        result: list[str] = []
        for m in cls._DOI_REGEX.findall(normalized):
            doi = m.rstrip(".,);:]")
            key = doi.lower()
            if key not in seen:
                seen.add(key)
                result.append(doi)
        return result

    @staticmethod
    def _resolve_doi(
        llm_doi: str,
        llm_title: str,
        real_dois: list[str],
    ) -> str:
        """Validate the LLM-supplied DOI against the regex-extracted whitelist.

        LLMs frequently fabricate DOIs that look plausible. We only accept
        a DOI if it appears verbatim in the original PDF text. Otherwise we
        drop it (CrossRef title-search in Step 2 can still find the paper).
        """
        if not real_dois:
            return llm_doi  # nothing to validate against
        whitelist_lower = {d.lower(): d for d in real_dois}
        if llm_doi and llm_doi.lower() in whitelist_lower:
            return whitelist_lower[llm_doi.lower()]
        return ""

    @staticmethod
    def _extract_text(pdf_path: Path) -> str:
        """Extract plain text from all pages of a PDF using PyMuPDF."""
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. "
                "Install with: pip install pymupdf"
            ) from exc

        doc = fitz.open(str(pdf_path))
        pages_text: list[str] = []
        try:
            for i in range(len(doc)):
                pages_text.append(doc[i].get_text())
        finally:
            doc.close()
        return "\n\n".join(pages_text)
