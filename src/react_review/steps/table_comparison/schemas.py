"""Data models for step 4: table comparison and reporting."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from react_review.core.enums import (
    FieldStatus,
    ReportVerdict,
    ValidationSeverity,
)


class FieldDiff(BaseModel):
    """A difference found in a single field between tables.

    Attributes:
        field_name: Display field name — under the new direct-join
            architecture this is the student's verbatim column header
            (e.g. "N", "Age (years)"). Step 4 joins by this exact string.
        canonical_concept: Snake_case abstract concept name from
            ``EvidenceFieldSchema.canonical_concept`` (e.g.
            ``sample_size``, ``age_mean``). Used by the report to
            show students how their column maps to the cross-paper
            extraction concept the AI looked for.
        student_raw_name: Original column header used by the student
            (e.g. "N", "Age (yrs)"). Empty if the student had no value.
        model_raw_names: Original field names used by the model(s)
            (e.g. "sample_size", "age_mean").
        student_value: Value from the student's table (raw).
        student_value_normalized: Normalised version of ``student_value``.
        student_evidence: Evidence snippet from the student side
            (often empty; PDF-extracted tables usually lack quotes).
        model_values: Values from model-generated tables (raw).
        model_values_normalized: Normalised versions of ``model_values``.
        model_evidence: Evidence snippets from the model extractor(s),
            one entry per model value.
        status: Rich FieldStatus describing the comparison outcome.
        is_consistent: True only when status == MATCH or PARTIAL_MATCH.
        explanation: Human-readable explanation of the outcome.
        source_type: Summary of where data came from, e.g.
            "student+llm" / "llm-only" / "student-only".
    """

    field_name: str
    canonical_concept: str = ""
    student_raw_name: str = ""
    model_raw_names: list[str] = Field(default_factory=list)
    student_value: str | int | float | bool | list | None = None
    student_value_normalized: str | int | float | bool | list | None = None
    student_evidence: str = ""
    model_values: list[str | int | float | bool | list | None] = Field(
        default_factory=list
    )
    model_values_normalized: list[str | int | float | bool | list | None] = Field(
        default_factory=list
    )
    model_evidence: list[str] = Field(default_factory=list)
    status: FieldStatus = FieldStatus.NOT_COMPARABLE
    is_consistent: bool = False
    explanation: str = ""
    source_type: str = ""


class ComparisonFlag(BaseModel):
    """A flag raised during table comparison."""

    code: str
    severity: ValidationSeverity
    message: str


class TableComparisonResult(BaseModel):
    """Comparison result for one paper's tables.

    Attributes:
        paper_id: The paper being compared.
        field_diffs: Per-field comparison details.
        agreement_rate: Fraction of *comparable* fields that agree
            (MATCH or PARTIAL_MATCH) out of all fields where both sides
            have a value. Undefined → 0.0 but see ``compared_count``.
        coverage_rate: Fraction of fields that could be compared at all
            (both sides had a value). Orthogonal to agreement_rate.
        compared_count: Number of fields counted in the agreement denominator.
        total_count: Total number of fields considered (all statuses).
        flags: Issues found during comparison.
        skipped: True if this paper could not be meaningfully compared
            (e.g. extractor produced nothing).
    """

    paper_id: str
    field_diffs: list[FieldDiff] = Field(default_factory=list)
    agreement_rate: float = 0.0
    coverage_rate: float = 0.0
    compared_count: int = 0
    total_count: int = 0
    flags: list[ComparisonFlag] = Field(default_factory=list)
    skipped: bool = False


class EvaluationReport(BaseModel):
    """Final evaluation report aggregating all pipeline results.

    Attributes:
        run_id: Unique identifier for this pipeline run.
        timestamp: When the report was generated.
        comparison_results: Per-paper comparison results.
        overall_flags: Aggregated flags from all steps.
        summary: Human-readable summary of findings.
        verdict: Machine-readable verdict (PASS / PARTIAL / FAIL / INCOMPLETE).
        avg_agreement: Mean agreement rate across compared papers.
        avg_coverage: Mean coverage rate across papers.
        compared_papers: Number of papers that produced an agreement number.
        skipped_papers: Number of papers skipped due to extractor / data gaps.
    """

    run_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    comparison_results: list[TableComparisonResult] = Field(default_factory=list)
    overall_flags: list[ComparisonFlag] = Field(default_factory=list)
    summary: str = ""
    verdict: ReportVerdict = ReportVerdict.INCOMPLETE
    avg_agreement: float = 0.0
    avg_coverage: float = 0.0
    compared_papers: int = 0
    skipped_papers: int = 0
