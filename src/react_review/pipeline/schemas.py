"""Data models for the pipeline orchestration layer.

The student does a systematic review where they:
  1. Define a multi-database search strategy and run it.
  2. Screen results and select 6-10 papers for inclusion.
  3. Extract data from those selected papers into a Table 1 / characteristics
     table.
  4. Submit everything for integrity checking.

This pipeline checks each of those steps. The schema below captures the
"fully-understood student submission" that Step 0 (Ingestion) produces, plus
the per-step outputs aggregated by the orchestrator.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from react_review.core.enums import PipelineStep, ValidationSeverity, VerificationStatus
from react_review.steps.data_extraction.schemas import ExtractedTable
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.steps.search_validation.schemas import SearchValidationResult
from react_review.steps.paper_verification.schemas import ReferenceVerificationResult
from react_review.steps.table_comparison.schemas import (
    EvaluationReport,
    TableComparisonResult,
)


# ======================================================================
# Step 0 (Ingestion) outputs — the "rich" understanding of the submission
# ======================================================================


class DatabaseQuery(BaseModel):
    """A search query for one specific bibliographic database.

    Step 0's strategy-extraction sub-task produces one of these per database
    the student claims to have searched. ``source`` distinguishes student
    verbatim text from LLM-translated approximations so the report can show
    both transparently.
    """

    database: str
    query: str
    source: Literal["verbatim", "llm_translated"] = "llm_translated"
    notes: str = ""


class SearchStrategy(BaseModel):
    """Structured representation of the review's search strategy.

    Replaces the old flat ``search_strategy_text`` field. Carries:
      * the verbatim paragraph from the Methods section (for audit/transparency);
      * a per-database structured query (executed in Step 1);
      * search date / filters / reported counts the student declared.
    """

    raw_text: str = ""
    extracted_per_database: list[DatabaseQuery] = Field(default_factory=list)
    search_date: str = ""
    filters: dict[str, str] = Field(default_factory=dict)
    reported_total_count: int | None = None
    reported_per_db_count: dict[str, int] = Field(default_factory=dict)


class EvidenceFieldSchema(BaseModel):
    """Rich schema for a single column of the student's data extraction table.

    Step 0 produces one of these per column in the student's Table 1. The
    schema travels through the pipeline and is consumed by:
      * Step 3 — uses ``canonical_concept`` to drive extraction prompts;
      * Step 4 — uses ``student_field_name`` to align student / model values,
        ``type`` to dispatch the comparator, and the threshold pair to decide
        MATCH / PARTIAL_MATCH / DIFF.

    Attributes:
        student_field_name: Verbatim column name as the student wrote it
            (e.g. "N", "Age (years)", "Method"). Used as the join key in Step 4.
        canonical_concept: Snake_case abstract concept (e.g. ``sample_size``,
            ``age_mean``, ``measurement_tool``). Used by Step 3 to instruct
            the LLM extractor what to look for in source papers.
        type: Comparison family — selects the comparator in Step 4.
        threshold_match: For numeric type this is a relative-error bound for
            MATCH (e.g. ``0.01`` = 1%). For text/categorical/author types it
            is a similarity bound (e.g. ``0.90``).
        threshold_partial: Same units as ``threshold_match`` but a looser
            bound used for PARTIAL_MATCH. Must be > ``threshold_match`` for
            numeric, < ``threshold_match`` for similarity-based types.
        is_metadata: True for fields like ``author`` / ``year`` / ``doi``
            that are already validated by Step 2; Step 4 skips these to
            avoid duplicate verification.
        synonym_check: True for fields like ``measurement_tool`` that should
            be normalised through the synonym table before comparison.
        description: Optional human-readable note (shown in the report).
    """

    student_field_name: str
    canonical_concept: str
    type: Literal["numeric", "text", "categorical", "author", "year", "doi"]
    threshold_match: float | None = None
    threshold_partial: float | None = None
    is_metadata: bool = False
    synonym_check: bool = False
    description: str = ""


class StudentReviewInput(BaseModel):
    """Output of Step 0 (Ingestion) — the fully-understood student submission.

    Constructed by ``steps/pdf_parsing/parser.py`` for PDF inputs or by the
    YAML loader for structured inputs. Downstream steps consume this object
    and never re-parse the original review.

    Attributes:
        student_id: Student identifier.
        review_title: Title of the systematic review.
        research_context: One-sentence description of what the review studies,
            in the review's own terms (shape: "<primary measure> in <exposed
            cohort> vs <comparison cohort>"). Used as research-context hint by
            Step 3 extractor prompts.
        search_strategy: Structured search strategy (replaces the old flat
            ``search_strategy_text``).
        selected_papers: The 6-10 papers the student selected after screening.
            These are the papers that go through Steps 2-4.
        evidence_schema: One entry per column of the student's data extraction
            table, with type / tolerance / metadata flags. Replaces the old
            flat ``extraction_fields`` list.
        submitted_tables: The student's data extraction tables (one per
            selected paper). Step 4 compares these against AI-extracted tables.
        review_full_text: Full text of the review. Kept for fallback parsing
            and for Step 1's strategy translation when ``search_strategy.raw_text``
            is insufficient.
    """

    student_id: str
    review_title: str
    research_context: str = ""
    search_strategy: SearchStrategy = Field(default_factory=SearchStrategy)
    selected_papers: list[ReferenceEntry] = Field(default_factory=list)
    evidence_schema: list[EvidenceFieldSchema] = Field(default_factory=list)
    submitted_tables: list[ExtractedTable] = Field(default_factory=list)
    review_full_text: str = ""


# ======================================================================
# Step 1 (Search Validation) — informational PRISMA-style identification
# ======================================================================


class DatabaseCountCheck(BaseModel):
    """One row of the PRISMA Identification table comparison.

    Step 1 produces one ``DatabaseCountCheck`` per database the student
    claims to have searched, comparing the student's declared count
    against an independently reproduced count where possible.

    ``verdict`` semantics:
        - ``MATCH``: AI count is within tolerance of student count.
        - ``WARN``: AI count is reproducible but differs from student count
          beyond tolerance.
        - ``REFERENCE``: We have no student count to compare to, but we ran
          the query for cross-coverage context (e.g. Europe PMC / OpenAlex).
        - ``UNVERIFIED``: We have no free API access to this database
          (e.g. Embase, CINAHL, Cochrane, Web of Science). Student count is
          shown but not independently verified.
    """

    database: str
    student_reported: int | None = None
    ai_reproduced: int | None = None
    delta_pct: float | None = None
    verdict: Literal["MATCH", "WARN", "REFERENCE", "UNVERIFIED"] = "UNVERIFIED"
    note: str = ""


class MultiDBIdentificationCheck(BaseModel):
    """Step 1 output — informational only, does not contribute to FAIL verdict.

    Captures the PRISMA "Records identified from databases" row: per-database
    student counts vs reproduced counts where verifiable, and explicit
    UNVERIFIED markers for paywalled databases.
    """

    per_database: list[DatabaseCountCheck] = Field(default_factory=list)
    student_reported_total: int | None = None
    ai_total_unique: int | None = None
    note: str = ""


# ======================================================================
# Step 2 gate — per-paper verification status
# ======================================================================


def is_paper_eligible_for_extraction(status: VerificationStatus) -> bool:
    """Step 2 → Step 3 gate predicate.

    Skip extraction only when CrossRef positively could not find the
    paper (``NOT_FOUND``). Other outcomes — including ``UNCERTAIN``
    (soft match) and ``ACCESS_RESTRICTED`` (paywalled but real) — still
    proceed to extraction with a flag, since the paper plausibly exists
    and the LLM may still get useful content from PMC / Unpaywall /
    abstract fallback.
    """
    return status != VerificationStatus.NOT_FOUND


# ======================================================================
# Cross-step types
# ======================================================================


class ValidationFlag(BaseModel):
    """A flag raised at any point in the pipeline.

    Attributes:
        step: Which pipeline step raised this flag.
        severity: How serious the issue is.
        code: Machine-readable code (e.g. COUNT_MISMATCH).
        message: Human-readable description.
        details: Optional extra context.
    """

    step: PipelineStep
    severity: ValidationSeverity
    code: str
    message: str
    details: str = ""


class PipelineRunResult(BaseModel):
    """Complete result of a single pipeline run.

    Attributes:
        run_id: Unique identifier for this run.
        student_input: The original input (Step 0 output).
        multi_db_identification_check: Step 1 informational output — PRISMA
            Identification count comparison across reachable databases.
        search_result: Step 1 output — search reproducibility check on the
            primary database (PubMed).
        papers_in_search: Step 1 check — which selected papers appear in
            the reproduced PubMed query results.
        papers_in_search_detail: Per-paper diagnostic for Step 1's
            "papers found in search" check (DOI hit, title fuzzy score, etc.).
        verification_results: Step 2 output — one per selected paper.
        paper_verification_statuses: Map of paper_id → VerificationStatus,
            used as the gate for Step 3/4.
        extracted_tables: Step 3 output — multiple extractors × papers.
        extractor_ids: Names of all LLM extractors used in Step 3.
        comparison_results: Step 4 per-paper comparison output.
        report: Step 4 final report.
        all_flags: Aggregated flags from all steps.
    """

    run_id: str
    student_input: StudentReviewInput

    # Step 1 outputs
    multi_db_identification_check: MultiDBIdentificationCheck | None = None
    search_result: SearchValidationResult | None = None
    papers_in_search: dict[str, bool] = Field(default_factory=dict)
    papers_in_search_detail: dict[str, dict] = Field(default_factory=dict)

    # Step 2 outputs
    verification_results: list[ReferenceVerificationResult] = Field(
        default_factory=list
    )
    paper_verification_statuses: dict[str, VerificationStatus] = Field(
        default_factory=dict
    )

    # Step 3 outputs
    extracted_tables: list[ExtractedTable] = Field(default_factory=list)
    extractor_ids: list[str] = Field(default_factory=list)

    # Step 4 outputs
    comparison_results: list[TableComparisonResult] = Field(default_factory=list)
    report: EvaluationReport | None = None

    # Aggregated
    all_flags: list[ValidationFlag] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
