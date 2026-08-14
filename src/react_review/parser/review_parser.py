"""Two-stage review parser: PDF → structure → long rows → ReviewDataItem[]."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from react_review.checklist import (
    Checklist,
    ChecklistApplication,
    annotate_checklist_claims,
    apply_checklist,
    checklist_claim_evidence,
    merge_checklist_applications,
    render_checklist,
)
from react_review.claim_ids import assign_claim_ids, claim_index
from react_review.hitl.events import StepStage, SubjectKind
from react_review.hitl.reporter import StepReporter
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.normalize.cohorts import (
    CohortRegistry,
    build_cohort_registry,
    load_aliases,
)
from react_review.normalize.doi import normalize_doi
from react_review.normalize.study_key import study_key
from react_review.parser.table_capture import TableCapturer
from react_review.parser.table_capture_contract import DEFAULT_TABLE_CAPTURE_PROFILE
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.knowledge import KnowledgeImportRecord
from react_review.schemas.reason import ReasonRecord
from react_review.schemas.resolution import FieldResolutionRecord, ResolutionCellRef
from react_review.schemas.table import CapturedTable, CapturedTableSet
from react_review.dkb import FieldResolver, ResolvedField, resolution_key

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
    # The cohorts this review was found to report, in its own words.
    cohorts: CohortRegistry = Field(default_factory=CohortRegistry)
    # Run-level decisions that explain how raw column names became field_types.
    field_resolutions: list[FieldResolutionRecord] = Field(default_factory=list)
    knowledge_imports: list[KnowledgeImportRecord] = Field(default_factory=list)
    knowledge_fingerprint: str = ""
    knowledge_concept_count: int = 0
    checklist: ChecklistApplication | None = None


def _pdf_text(pdf_path: Path | str) -> str:
    import fitz  # PyMuPDF

    from react_review.normalize.text import clean_pdf_text

    doc = fitz.open(str(pdf_path))
    try:
        return clean_pdf_text("\n\n".join(doc[i].get_text() for i in range(len(doc))))
    finally:
        doc.close()


# The parser and the matcher must derive the SAME id from the same citation —
# it is the key the whole audit joins on. One implementation, in normalize/.
_study_slug = study_key


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


_ALIASES = Path(__file__).resolve().parents[3] / "configs" / "cohort_aliases.json"

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
        cohort_aliases: dict[str, list[str]] | None = None,
        checklist: Checklist | None = None,
        table_capture_prompt_profile: str = DEFAULT_TABLE_CAPTURE_PROFILE,
    ) -> None:
        self._backend = backend
        self._resolver = resolver          # the parser holds NO domain knowledge itself
        self._max_chars = max_chars
        # Default reporter never blocks and never writes (library/CI behaviour).
        self._reporter = reporter or StepReporter()
        self._chunk_rows = max(1, chunk_rows)
        self._keep_tables = keep_tables
        self._drop_tables = drop_tables
        self._checklist = checklist
        self._capturer = TableCapturer(
            backend, alt_backend=alt_backend,
            prompt_profile=table_capture_prompt_profile)
        # Alias file only RE-KEYS a discovered cohort (benchmark compatibility);
        # it never introduces one, so a new domain stays domain-neutral.
        self._cohort_aliases = (load_aliases(_ALIASES) if cohort_aliases is None
                                else cohort_aliases)

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

        # 3. Discover this review's cohorts from the labels it actually used —
        #    BEFORE any resolution, so nothing has been folded into a default.
        registry = build_cohort_registry(
            [str(r.get("cohort_label") or "") for r in raw_rows]
            + [c for t in table_set.tables for c in t.cohort_labels_seen],
            aliases=self._cohort_aliases,
        )
        await self._reporter.step_or_stop(
            StepStage.COHORT_REGISTRY,
            title="Cohorts found in this review",
            payload={"cohorts": [c.model_dump(mode="json") for c in registry.labels]},
            render_blocks=[self._render_cohorts(registry)],
            warnings=([] if registry.labels else
                      ["no cohort labels were found — every value will be treated "
                       "as a single combined cohort"]),
        )

        # 4. Resolve the UNIQUE field questions before applying any answer to a
        # row.  The human sees and may stop at these decisions before the long
        # table derived from them is presented as an output.
        row_resolutions, field_resolutions = await self._resolve_fields(raw_rows, ctx)
        knowledge_imports = [
            record.model_copy(deep=True) for record in self._resolver.kb.imports]
        kb_fingerprint = self._resolver.kb.version or self._resolver.kb.fingerprint()
        await self._reporter.step_or_stop(
            StepStage.FIELD_RESOLUTION,
            title="Field concepts resolved",
            subject=str(Path(pdf_path).resolve()),
            subject_kind=SubjectKind.REVIEW_PDF,
            payload={
                "research_context": ctx,
                "knowledge_base": {
                    "fingerprint": kb_fingerprint,
                    "concept_count": len(self._resolver.kb.entries),
                    "imports": [r.model_dump(mode="json") for r in knowledge_imports],
                },
                "resolutions": [r.model_dump(mode="json") for r in field_resolutions],
            },
            render_blocks=[
                self._render_knowledge(
                    knowledge_imports, kb_fingerprint, len(self._resolver.kb.entries)),
                self._render_resolutions(field_resolutions),
            ],
            warnings=[
                *self._knowledge_warnings(knowledge_imports),
                *self._resolution_warnings(field_resolutions),
            ],
        )

        items = self._postprocess(raw_rows, row_resolutions, registry)

        checklist_review_application = None
        checklist_application = None
        if self._checklist is not None:
            # First pass: route existing concrete claims and settle only review-
            # level coverage.  Study completeness cannot be known before the
            # human approves REFERENCE_COVERAGE below.
            checklist_review_application = apply_checklist(
                self._checklist, items, table_set, review_text=full_text,
                scopes={"review"}, evaluation_pass="review")
            routed_claims = checklist_claim_evidence(self._checklist, items)
            await self._reporter.step_or_stop(
                StepStage.CHECKLIST_REVIEW,
                title="Review-level checklist coverage and claim routing",
                subject=str(Path(pdf_path).resolve()),
                subject_kind=SubjectKind.REVIEW_PDF,
                payload={
                    **checklist_review_application.model_dump(mode="json"),
                    "routed_claims": [e.model_dump(mode="json") for e in routed_claims],
                },
                render_blocks=[render_checklist(checklist_review_application)],
                warnings=[gap.reason for gap in checklist_review_application.gaps],
            )
            # Apply the approved routing identity only after the checkpoint.
            # No rows are added: a checklist cannot manufacture a missing value.
            items = annotate_checklist_claims(self._checklist, items)

        # Human-readable claim identities belong to the normalised claims, not
        # to the verbatim table.  Assign them after study-level de-duplication
        # and checklist annotation, in approved table/cell order.
        items = assign_claim_ids(items, table_set)

        await self._reporter.step_or_stop(
            StepStage.LONG_FORMAT_ROWS,
            title="Table converted to long format",
            payload={"rows": raw_rows,
                     "items": [i.model_dump(mode="json") for i in items],
                     "claim_index": claim_index(items)},
            render_blocks=[self._render_items(items)],
            warnings=self._unpivot_warnings(table_set, raw_rows, items),
        )

        # 6. included-study references, extracted from the reference window (doc tail).
        studies = await self._extract_studies(_refs_window(full_text))
        studies = await self._review_coverage(studies, items)

        if self._checklist is not None and checklist_review_application is not None:
            # Second pass: the approved references are now the authoritative
            # expected-study set.  This pass only finalises coverage/gaps; it
            # never creates or re-routes a ReviewDataItem.
            study_application = apply_checklist(
                self._checklist, items, table_set, review_text=full_text,
                study_ids=[study.study_id for study in studies],
                scopes={"per_study", "per_cohort"},
                evaluation_pass="study_coverage")
            checklist_application = merge_checklist_applications(
                checklist_review_application, study_application)
            await self._reporter.step_or_stop(
                StepStage.CHECKLIST_STUDY_COVERAGE,
                title="Checklist coverage for approved studies",
                subject=str(Path(pdf_path).resolve()),
                subject_kind=SubjectKind.REVIEW_PDF,
                payload={
                    **checklist_application.model_dump(mode="json"),
                    "approved_study_ids": [study.study_id for study in studies],
                    "pass_assessments": [
                        assessment.model_dump(mode="json")
                        for assessment in study_application.assessments],
                },
                render_blocks=[render_checklist(checklist_application)],
                warnings=[gap.reason for gap in study_application.gaps],
            )

        run_steps = [
            StepRecord(index=0, thought="capture the review's tables verbatim",
                       tool="llm:table_capture",
                       observation={"tables": [t.table_id for t in table_set.tables],
                                    "dropped": table_set.dropped}),
            StepRecord(index=1, thought="unpivot the captured tables to long rows",
                       tool="llm:unpivot", observation={"n_rows": len(raw_rows)}),
            StepRecord(index=2, thought="resolve unique field concepts before applying them",
                       tool="dkb:resolve_fields",
                       observation={
                           "n_resolutions": len(field_resolutions),
                           "n_model_attempts": sum(
                               len(r.attempts) for r in field_resolutions),
                           "kb_fingerprint": kb_fingerprint,
                           "n_ontology_imports": len(knowledge_imports),
                       }),
        ]
        if checklist_review_application is not None:
            run_steps.append(StepRecord(
                index=len(run_steps), thought="check review-level coverage and route claims",
                tool="checklist:review_coverage",
                observation={
                    "name": checklist_review_application.name,
                    "n_items": len(checklist_review_application.assessments),
                    "n_required_gaps": len(checklist_review_application.gaps),
                    "sha256": checklist_review_application.sha256,
                }))
        run_steps.append(StepRecord(
            index=len(run_steps), thought="extract and approve included-study references",
            tool="llm:stage_refs", observation={"n_studies": len(studies)}))
        if checklist_application is not None:
            run_steps.append(StepRecord(
                index=len(run_steps), thought="finalise checklist coverage for approved studies",
                tool="checklist:study_coverage",
                observation={
                    "n_items": len(checklist_application.assessments),
                    "n_required_gaps": len(checklist_application.gaps),
                    "approved_study_ids": [study.study_id for study in studies],
                    "sha256": checklist_application.sha256,
                }))

        record = AgentRun(
            agent="parser",
            task={"pdf": str(pdf_path)},
            steps=run_steps,
            status="finished",
            final={"n_items": len(items), "n_studies": len(studies)},
        )
        logger.info("review_parse_done", n_rows=len(raw_rows),
                    n_items=len(items), n_studies=len(studies),
                    n_tables=len(table_set.tables))
        return ParsedReview(items=items, record=record, research_context=ctx,
                            studies=studies, tables=table_set, cohorts=registry,
                            field_resolutions=field_resolutions,
                            knowledge_imports=knowledge_imports,
                            knowledge_fingerprint=kb_fingerprint,
                            knowledge_concept_count=len(self._resolver.kb.entries),
                            checklist=checklist_application)

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

    async def _review_coverage(
        self, studies: list[ParsedStudy], items: list[ReviewDataItem],
    ) -> list[ParsedStudy]:
        """Show which studies have a citation, and let a human drop the rest.

        A reference list holds every work the review cites, not just the studies
        it included — on the benchmark that is 34 references for 9 included
        studies. Fetching all of them would audit papers the review never claimed
        anything about, so the coverage is shown and the extras can be removed.
        """
        claimed = {i.study_id for i in items}
        missing = sorted(claimed - {s.study_id for s in studies})

        await self._reporter.step_or_stop(
            StepStage.REFERENCE_COVERAGE,
            title="Source papers for the studies in the table",
            payload={"references": [
                {"id": s.study_id, "label": f"{s.study_id}  {s.citation[:70]}"
                                            f"{'  [doi]' if s.doi else '  [no doi]'}",
                 **s.model_dump(mode="json")} for s in studies]},
            render_blocks=[self._render_coverage(studies, claimed, missing)],
            warnings=([f"{len(missing)} study/studies in the table have no citation: "
                       + ", ".join(missing[:8])] if missing else []),
            selectable="references",
        )
        event = self._reporter.last_event
        if event is None or not event.dropped:
            return studies
        kept = {r["id"] for r in event.selectable_items()}
        logger.info("references_dropped",
                    ids=[s.study_id for s in studies if s.study_id not in kept])
        return [s for s in studies if s.study_id in kept]

    @staticmethod
    def _render_coverage(studies: list[ParsedStudy], claimed: set[str],
                         missing: list[str]) -> str:
        used = [s for s in studies if s.study_id in claimed]
        extra = [s for s in studies if s.study_id not in claimed]
        lines = [f"  {len(studies)} reference(s) extracted; "
                 f"{len(used)} match a study in the data table, "
                 f"{len(extra)} do not; {len(missing)} table study/studies have none."]
        if extra:
            lines.append("  references with no claim in the table "
                         "(cited works, not included studies):")
            lines += [f"    {s.study_id:<24} {s.citation[:64]}" for s in extra[:12]]
            if len(extra) > 12:
                lines.append(f"    … {len(extra) - 12} more")
        if missing:
            lines.append("  studies in the table with NO citation found:")
            lines += [f"    {sid}" for sid in missing[:12]]
        return "\n".join(lines)

    @staticmethod
    def _resolution_inputs(row: dict[str, Any]) -> tuple[str, object, str, str] | None:
        """The signals that make one row a real field-resolution question.

        Placeholders remain review evidence, but they do not ask the Resolver to
        infer a concept from an absent value. Identifier/dimension columns are
        likewise kept out of field normalisation.
        """
        if not isinstance(row, dict):
            return None
        raw_name = str(row.get("column_header") or "").strip()
        value = row.get("value")
        if not raw_name or value is None or _norm_col(raw_name) in _IDENTIFIER_COLUMNS:
            return None
        text = value.strip() if isinstance(value, str) else value
        if isinstance(text, str) and (not text or text.lower() in _PLACEHOLDER):
            return None
        unit = str(row.get("unit") or "").strip() or _inline_unit(value)
        modality = str(row.get("row_context") or "")
        return raw_name, value, unit, modality

    @staticmethod
    def _merge_resolution(
        records: dict[str, FieldResolutionRecord],
        resolved: ResolvedField,
        row: dict[str, Any],
    ) -> None:
        """Fold a per-row outcome into one auditable resolution record."""
        reason = "; ".join(r.message or r.code for r in resolved.reasons)
        cell = ResolutionCellRef(
            table_id=str(row.get("table_id") or ""), cell_ref=_cell_ref(row),
            study_id=_study_slug(_row_key_label(row.get("row_key"))),
            column_header=str(row.get("column_header") or ""), unit=resolved.unit,
            status=resolved.status, field_type=resolved.field_type or "", reason=reason,
        )
        incoming = FieldResolutionRecord.model_validate(
            resolved.model_dump(mode="python", exclude={"affected_cells"}))
        incoming.affected_cells = [cell]
        incoming.statuses_seen = list(dict.fromkeys(
            [*incoming.statuses_seen, resolved.status]))
        incoming.field_types_seen = list(dict.fromkeys(
            [*incoming.field_types_seen,
             *([resolved.field_type] if resolved.field_type else [])]))

        current = records.get(resolved.resolution_key)
        if current is None:
            records[resolved.resolution_key] = incoming
            return

        current.affected_cells.append(cell)
        current.cache_hits += incoming.cache_hits
        current.candidate_names = list(dict.fromkeys(
            [*current.candidate_names, *incoming.candidate_names]))
        current.consensus_count = max(current.consensus_count, incoming.consensus_count)
        if incoming.stability != "not_checked":
            if current.stability == "not_checked":
                current.stability = incoming.stability
            elif current.stability != incoming.stability:
                current.stability = "unstable"
        current.statuses_seen = list(dict.fromkeys(
            [*current.statuses_seen, *incoming.statuses_seen]))
        current.field_types_seen = list(dict.fromkeys(
            [*current.field_types_seen, *incoming.field_types_seen]))
        if len(current.statuses_seen) > 1 or len(current.field_types_seen) > 1:
            current.status = "mixed"
            current.source = "mixed"
        if len(current.field_types_seen) > 1:
            current.field_type = None
        for name, passed in incoming.checks.items():
            current.checks[name] = current.checks.get(name, True) and passed
        known_reasons = {(r.code, r.message, r.source) for r in current.reasons}
        current.reasons.extend(
            r for r in incoming.reasons
            if (r.code, r.message, r.source) not in known_reasons)
        known_attempts = {(a.seed, a.model_id, a.response_sha256, a.error)
                          for a in current.attempts}
        current.attempts.extend(
            a for a in incoming.attempts
            if (a.seed, a.model_id, a.response_sha256, a.error) not in known_attempts)
        current.proposal = current.proposal or incoming.proposal

    async def _resolve_fields(
        self, raw_rows: list[dict[str, Any]], research_context: str,
    ) -> tuple[dict[int, ResolvedField], list[FieldResolutionRecord]]:
        """Resolve all field questions first; do not create review items yet."""
        by_row: dict[int, ResolvedField] = {}
        records: dict[str, FieldResolutionRecord] = {}
        for index, row in enumerate(raw_rows):
            inputs = self._resolution_inputs(row)
            if inputs is None:
                continue
            raw_name, value, unit, modality = inputs
            try:
                resolved = await self._resolver.resolve(
                    raw_name, unit=unit, modality=modality,
                    research_context=research_context, value=value)
            except Exception as exc:                         # noqa: BLE001
                # Resolver errors are evidence too. Keep the row and make the
                # exception visible at FIELD_RESOLUTION rather than dropping it.
                logger.warning("resolve_failed", raw=raw_name, error=str(exc)[:120])
                resolved = ResolvedField(
                    resolution_key=resolution_key(
                        raw_name, unit, research_context, modality),
                    raw_field_name=raw_name, unit=unit, modality=modality,
                    reasons=[ReasonRecord(
                        code="concept_resolution_exception", stage="field_resolution",
                        source="exception", message=str(exc)[:200])],
                    statuses_seen=["unresolved"],
                )
            by_row[index] = resolved
            self._merge_resolution(records, resolved, row)
        return by_row, list(records.values())

    def _postprocess(
        self, raw_rows: list[dict[str, Any]],
        row_resolutions: dict[int, ResolvedField],
        registry: CohortRegistry | None = None,
    ) -> list[ReviewDataItem]:
        """Apply already-inspected resolution decisions to the raw long rows."""
        registry = registry or CohortRegistry()
        items: list[ReviewDataItem] = []
        seen_study_level: set[tuple[str, str]] = set()
        for index, r in enumerate(raw_rows):
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
            cohort = registry.resolve(cohort_label)
            group, cohort_status = cohort.key, cohort.status
            common = dict(
                study_id=study_id, raw_field_name=raw_name, unit=unit,
                table_id=table_id, cell_ref=cell_ref, column_header=raw_name,
                cohort_label=cohort_label, timepoint_label=timepoint_label,
            )
            cohort_reasons = ([] if cohort.known else [ReasonRecord(
                code="cohort_unknown", stage="parser", message=cohort.reason)])

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
                    **common, group=group, cohort_status=cohort_status,
                    field_type="", value=None, resolution_status="unresolved",
                    reasons=[ReasonRecord(
                        code="placeholder_cell", stage="parser",
                        message=f"the table cell reads {value.strip()!r}; "
                                "no value was reported here"), *cohort_reasons],
                ))
                continue

            # The decision was completed and shown at FIELD_RESOLUTION before
            # this row is built. Missing here means the resolver could not even
            # form a question; preserve that as an explicit unresolved outcome.
            rf = row_resolutions.get(index) or ResolvedField(raw_field_name=raw_name)

            # UNRESOLVED: keep the raw item (field_type null) — the audit marks it
            # not_comparable / needs_review; the concept goes to proposals to learn.
            if rf.status == "unresolved":
                resolution_reasons = list(rf.reasons) or [ReasonRecord(
                    code="concept_unresolved", stage="parser",
                    message=f"column {raw_name!r} did not map to a known concept")]
                items.append(ReviewDataItem(
                    **common, group=group, cohort_status=cohort_status,
                    field_type="", value=value, resolution_status="unresolved",
                    resolution_key=rf.resolution_key,
                    reasons=[*resolution_reasons, *cohort_reasons],
                ))
                continue

            ft = rf.field_type
            # Parser APPLIES scope, using the knowledge the Resolver provided.
            # A study-level field has no cohort dimension at all — that is not
            # the same as a cohort we failed to place, so it gets its own status.
            if rf.scope == "study":
                group, cohort_status = "-", "not_applicable"
                cohort_reasons = []
                if (study_id, ft) in seen_study_level:
                    continue
                seen_study_level.add((study_id, ft))
            items.append(ReviewDataItem(
                **common, group=group, cohort_status=cohort_status,
                field_type=ft, value=value, reasons=cohort_reasons,
                resolution_key=rf.resolution_key,
                resolution_status=("resolved" if rf.status == "authoritative" else rf.status),
            ))
        return items

    # --- rendering + warnings for the long-format checkpoint ---

    @staticmethod
    def _render_cohorts(registry: CohortRegistry) -> str:
        """The arms this review reports, in its own words — for a human to confirm."""
        if not registry.labels:
            return "  (this review reports no cohort split)"
        lines = [f"  {len(registry.labels)} cohort(s) found in the review's own table:"]
        for c in registry.labels:
            variants = [v for v in c.raw_variants if v != c.display]
            extra = f"  (also written: {', '.join(variants)})" if variants else ""
            lines.append(f"    {c.key:<22} “{c.display}”{extra}   [{c.source}]")
        return "\n".join(lines)

    @staticmethod
    def _render_resolutions(
        resolutions: list[FieldResolutionRecord], limit: int = 30,
    ) -> str:
        """One concept-level line per unique field question."""
        if not resolutions:
            return "  (no measurement fields required resolution)"
        counts: dict[str, int] = {}
        for r in resolutions:
            counts[r.status] = counts.get(r.status, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        model_attempts = sum(len(r.attempts) for r in resolutions)
        lines = [f"  {len(resolutions)} unique resolution question(s): {summary}; "
                 f"{model_attempts} model attempt(s)"]
        for r in resolutions[:limit]:
            target = r.field_type or "UNRESOLVED"
            bad = [name for name, passed in r.checks.items() if not passed]
            suffix = f"; failed: {', '.join(bad)}" if bad else ""
            sampled = (f"; {r.stability} {r.consensus_count}/{len(r.attempts)}"
                       if r.attempts else "")
            lines.append(
                f"    {r.raw_field_name!r:<30} -> {target:<22} "
                f"[{r.status}; {r.source}{sampled}; "
                f"{len(r.affected_cells)} cell(s){suffix}]")
        if len(resolutions) > limit:
            lines.append(f"    … {len(resolutions) - limit} more")
        return "\n".join(lines)

    @staticmethod
    def _render_knowledge(
        imports: list[KnowledgeImportRecord], fingerprint: str, concept_count: int,
    ) -> str:
        lines = [f"  knowledge base: {concept_count} concept(s); "
                 f"fingerprint={fingerprint}"]
        if not imports:
            lines.append("    seed only; no ontology slices loaded")
            return "\n".join(lines)
        for record in imports:
            lines.append(
                f"    {record.source}: +{record.added}, merged={record.merged}, "
                f"conflicts={len(record.conflicts)}; {record.path}")
        return "\n".join(lines)

    @staticmethod
    def _knowledge_warnings(imports: list[KnowledgeImportRecord]) -> list[str]:
        n_conflicts = sum(len(record.conflicts) for record in imports)
        if not n_conflicts:
            return []
        return [
            f"{n_conflicts} seed/ontology field conflict(s) were resolved by "
            "the explicit ontology_override policy; inspect the import records"]

    @staticmethod
    def _resolution_warnings(resolutions: list[FieldResolutionRecord]) -> list[str]:
        counts: dict[str, int] = {}
        for r in resolutions:
            counts[r.status] = counts.get(r.status, 0) + 1
        warnings = []
        if counts.get("candidate"):
            warnings.append(
                f"{counts['candidate']} provisional concept mapping(s) require review")
        if counts.get("unresolved"):
            warnings.append(
                f"{counts['unresolved']} field question(s) could not be resolved")
        if counts.get("mixed"):
            warnings.append(
                f"{counts['mixed']} field question(s) produced mixed row outcomes")
        return warnings

    @staticmethod
    def _render_items(items: list[ReviewDataItem], limit: int = 25) -> str:
        if not items:
            return "  (no rows produced)"
        lines = [f"  {len(items)} row(s)"]
        for i in items[:limit]:
            concept = i.field_type or f"UNRESOLVED «{i.raw_field_name}»"
            cohort = i.cohort_label or i.group
            claim_id = i.review_data_id or "-"
            lines.append(
                f"    [{claim_id:<6}] {i.study_id:<24} {cohort:<12} "
                f"{concept:<18} {i.value!r}")
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
