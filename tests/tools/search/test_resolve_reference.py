"""The resolve_reference tool: DOI passthrough + reconcile-on-miss."""
from __future__ import annotations

import pytest

from react_review.tools.search import (
    CandidateWork,
    ReferenceReconciler,
    ResolveReferenceInput,
    ResolveReferenceResult,
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


@pytest.mark.asyncio
async def test_pmid_lookup_skips_journal_gate():
    mismatch = CandidateWork(
        doi="10.3389/fonc.2023.1104109",
        title="Minimally invasive esophagectomy after neoadjuvant therapy",
        year=2023, journal="Wrong Journal That Does Not Abbreviate",
        pmid="37251945", source="europepmc")
    out = await _tool([mismatch]).run(ResolveReferenceInput(
        title="Minimally invasive esophagectomy after neoadjuvant therapy",
        year=2023, journal="Front Oncol", pmid="37251945"))
    assert out.status == "resolved" and out.doi == "10.3389/fonc.2023.1104109"
    assert out.passthrough is False


@pytest.mark.asyncio
async def test_bind_identifier_resolve_injects_pmid_from_the_open_study():
    from react_review.steps.paper_verification.schemas import ReferenceEntry
    from react_review.tools.search.resolve_reference import bind_identifier_resolve

    seen: list[ResolveReferenceInput] = []

    class _Inner:
        async def run(self, payload: ResolveReferenceInput) -> ResolveReferenceResult:
            seen.append(payload)
            return ResolveReferenceResult(status="resolved", doi="10.1/x")

    class _Collector:
        def __init__(self) -> None:
            self._resolve = _Inner()

        async def open_study(self, reference):
            return await self._resolve.run(ResolveReferenceInput(title=reference.title))

    wrapped = bind_identifier_resolve(_Collector())
    await wrapped.open_study(ReferenceEntry(
        title="Capovilla G. Front Oncol. 2023;13:1104109.", pmid="37251945"))
    assert seen[0].pmid == "37251945"
    assert seen[0].title.startswith("Capovilla")

