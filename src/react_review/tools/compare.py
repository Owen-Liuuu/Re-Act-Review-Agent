"""Compare stage: the compare_values tool (wraps audit.compare_values)."""
from __future__ import annotations

from react_review.audit import ToleranceTable, compare_values
from react_review.schemas.audit import MatchResult
from react_review.tools.base import Tool, ToolStage
from react_review.tools.models import CompareInput


class CompareValuesTool(Tool):
    """Audit one review↔source value pair using the dual-band tolerance."""

    name = "compare_values"
    stage = ToolStage.COMPARE
    input_model = CompareInput
    output_model = MatchResult

    def __init__(self, tolerance: ToleranceTable) -> None:
        self._tol = tolerance

    async def run(self, payload: CompareInput) -> MatchResult:
        ft = payload.field_type
        return compare_values(
            field_type=ft,
            review_value=payload.review_value,
            source_value=payload.source_value,
            review_unit=payload.review_unit,
            source_unit=payload.source_unit,
            rel_tolerance=self._tol.rel_tolerance(ft),
            sd_rel_tolerance=self._tol.sd_rel_tolerance(ft),
        )
