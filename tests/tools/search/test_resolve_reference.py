"""The resolve_reference tool: DOI passthrough + reconcile-on-miss."""
from __future__ import annotations

import pytest

from react_review.tools.search import (
    CandidateWork,
    ReferenceReconciler,
    ResolveReferenceInput,
    ResolveReferenceTool,
    StaticResolver,
)

GOOD = CandidateWork(doi="10.1/x", title="Epicardial fat thickness in type 1 diabetes",
                     authors=["Ahmad A", "Jones C"], year=2022,
                     journal="Journal of Cardiology", source="crossref")


def _tool(cands) -> ResolveReferenceTool:
    return ResolveReferenceTool(ReferenceReconciler([StaticResolver("crossref", cands)]))


@pytest.mark.asyncio
async def test_passthrough_when_doi_already_present():
    out = await _tool([]).run(  # reconciler never consulted
        ResolveReferenceInput(doi="https://doi.org/10.5/Y", title="whatever"))
    assert out.passthrough is True and out.status == "resolved"
    assert out.doi == "10.5/y" and out.source == "given"


@pytest.mark.asyncio
async def test_resolves_via_reconciler_when_no_doi():
    out = await _tool([GOOD]).run(ResolveReferenceInput(
        title="Epicardial fat thickness in type 1 diabetes",
        authors=["Ahmad A", "Smith B"], year=2022, journal="Journal of Cardiology"))
    assert out.passthrough is False and out.status == "resolved" and out.doi == "10.1/x"


@pytest.mark.asyncio
async def test_unresolved_when_no_match():
    out = await _tool([]).run(
        ResolveReferenceInput(title="An uncited grey-literature report", year=2010))
    assert out.status == "unresolved_source" and out.doi == ""
