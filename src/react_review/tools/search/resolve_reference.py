"""The ``resolve_reference`` tool — wrap the reconciler behind the Tool contract.

Agent 1 calls this when a review's included-study reference has NO printed DOI:
it resolves the citation to a gated DOI via online services. An already-known DOI
is a passthrough (no lookup). A PMID is an identifier lookup (no title-search gate).
"""
from __future__ import annotations

from react_review.normalize.doi import normalize_doi, printed_pmid
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
        pmid = (payload.pmid or "").strip() or printed_pmid(
            payload.citation) or printed_pmid(payload.title)
        q = ReferenceQuery(
            citation=payload.citation, title=payload.title,
            authors=payload.authors, year=payload.year, journal=payload.journal,
            pmid=pmid,
        )
        res = await self._reconciler.resolve(q)
        return ResolveReferenceResult(**res.model_dump())


def bind_identifier_resolve(collector):
    """Inject DOI/PMID from the open study into resolve_reference.

    ``collector.py`` is inside the evidence-adequacy hash boundary and does not
    pass ``pmid`` on ``ResolveReferenceInput``. Production wraps the tool so a
    PMID-only reference takes the identifier path instead of title search.
    """
    inner = collector._resolve
    if inner is None:
        return collector
    current: dict[str, object] = {"ref": None}

    class _Bound:
        async def run(self, payload: ResolveReferenceInput) -> ResolveReferenceResult:
            ref = current["ref"]
            pmid = payload.pmid or ""
            doi = payload.doi or ""
            if ref is not None:
                pmid = pmid or str(getattr(ref, "pmid", "") or "")
                doi = doi or str(getattr(ref, "doi", "") or "")
            updates = {}
            if pmid and payload.pmid != pmid:
                updates["pmid"] = pmid
            if doi and payload.doi != doi:
                updates["doi"] = doi
            if updates:
                payload = payload.model_copy(update=updates)
            return await inner.run(payload)

    original_open = collector.open_study

    async def open_study(reference, *args, **kwargs):
        current["ref"] = reference
        try:
            return await original_open(reference, *args, **kwargs)
        finally:
            current["ref"] = None

    collector._resolve = _Bound()
    collector.open_study = open_study
    return collector
