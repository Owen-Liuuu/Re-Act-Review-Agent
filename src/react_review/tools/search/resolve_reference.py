"""The ``resolve_reference`` tool — wrap the reconciler behind the Tool contract.

Agent 1 calls this when a review's included-study reference has NO printed DOI:
it resolves the citation to a gated DOI via online services. An already-known DOI
is a passthrough (no lookup).
"""
from __future__ import annotations

from react_review.normalize.doi import normalize_doi
from react_review.tools.base import Tool, ToolStage
from react_review.tools.search.models import (
    ReferenceQuery,
    ResolveReferenceInput,
    ResolveReferenceResult,
)
from react_review.tools.search.reconciler import ReferenceReconciler


class ResolveReferenceTool(Tool):
    """Resolve a citation (no printed DOI) to a gated DOI via online services."""

    name = "resolve_reference"
    stage = ToolStage.SEARCH
    input_model = ResolveReferenceInput
    output_model = ResolveReferenceResult

    def __init__(self, reconciler: ReferenceReconciler) -> None:
        self._reconciler = reconciler

    async def run(self, payload: ResolveReferenceInput) -> ResolveReferenceResult:
        # Already have a DOI → passthrough, no online lookup.
        if payload.doi:
            return ResolveReferenceResult(
                status="resolved", doi=normalize_doi(payload.doi),
                source="given", confidence=1.0, passthrough=True,
            )
        q = ReferenceQuery(
            citation=payload.citation, title=payload.title,
            authors=payload.authors, year=payload.year, journal=payload.journal,
        )
        res = await self._reconciler.resolve(q)
        return ResolveReferenceResult(**res.model_dump())
