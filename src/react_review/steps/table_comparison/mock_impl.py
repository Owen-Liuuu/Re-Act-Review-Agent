"""Mock implementations for step 4: table comparison and reporting.

Delegates to :class:`RealTableComparator` so the mock output has the
same shape as the real comparator (single row per field, no duplicate
student/model columns) and exercises the same dispatch logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from react_review.core.enums import ReportVerdict, ValidationSeverity
from react_review.steps.data_extraction.schemas import ExtractedTable
from react_review.steps.table_comparison.interfaces import (
    ReportGenerator,
    TableComparator,
)
from react_review.steps.table_comparison.real_impl import RealTableComparator
from react_review.steps.table_comparison.schemas import (
    ComparisonFlag,
    EvaluationReport,
    TableComparisonResult,
)

if TYPE_CHECKING:
    from react_review.pipeline.schemas import EvidenceFieldSchema


class MockTableComparator(TableComparator):
    """Returns comparison results with the same shape as the real implementation.

    Deterministic and suitable for tests — it does not call any external
    services. The ``schema`` parameter is passed through to the real
    comparator so type-aware comparison still happens in mock mode.
    """

    def __init__(self) -> None:
        self._inner = RealTableComparator()

    async def compare(
        self,
        student_table: ExtractedTable,
        model_tables: list[ExtractedTable],
        schema: "list[EvidenceFieldSchema] | None" = None,
    ) -> TableComparisonResult:
        result = await self._inner.compare(student_table, model_tables, schema)
        # Tag with a mock marker so tests can distinguish if needed.
        result.flags.append(
            ComparisonFlag(
                code="MOCK_MINOR_DIFF",
                severity=ValidationSeverity.INFO,
                message="Minor differences detected in mock comparison.",
            )
        )
        # Note: previous versions hard-set agreement_rate=0.85 when
        # ``compared_count == 0`` to keep legacy tests happy. Removed in
        # the P0 cleanup — under the new schema, ``compared_count == 0``
        # legitimately means "no comparable fields" (e.g. all
        # MISSING_STUDENT) and the 0.0 default is the right signal.
        return result


class MockReportGenerator(ReportGenerator):
    """Returns a fake evaluation report for testing."""

    async def generate(
        self,
        comparison_results: list[TableComparisonResult],
        run_id: str,
    ) -> EvaluationReport:
        all_flags = []
        for cr in comparison_results:
            all_flags.extend(cr.flags)

        total = len(comparison_results)
        compared_results = [cr for cr in comparison_results if not cr.skipped]
        skipped = total - len(compared_results)

        avg_agreement = (
            sum(cr.agreement_rate for cr in compared_results) / len(compared_results)
            if compared_results else 0.0
        )
        avg_coverage = (
            sum(cr.coverage_rate for cr in comparison_results) / total
            if total else 0.0
        )

        return EvaluationReport(
            run_id=run_id,
            comparison_results=comparison_results,
            overall_flags=all_flags,
            summary=(
                f"Mock evaluation complete. "
                f"Reviewed {total} paper(s). "
                f"Average agreement rate: {avg_agreement:.1%}."
            ),
            verdict=ReportVerdict.PASS if compared_results else ReportVerdict.INCOMPLETE,
            avg_agreement=avg_agreement,
            avg_coverage=avg_coverage,
            compared_papers=len(compared_results),
            skipped_papers=skipped,
        )
