"""ReferenceReconciler — gated citation→DOI across (stubbed) resolvers."""
from __future__ import annotations

import pytest

from react_review.tools.search import (
    CandidateWork,
    ReferenceQuery,
    ReferenceReconciler,
    StaticResolver,
)

Q = ReferenceQuery(title="Epicardial fat thickness in type 1 diabetes",
                   authors=["Ahmad A", "Smith B"], year=2022, journal="Journal of Cardiology")
GOOD = CandidateWork(doi="10.1/x", title="Epicardial fat thickness in type 1 diabetes",
                     authors=["Ahmad A", "Jones C"], year=2022,
                     journal="Journal of Cardiology", source="crossref")
BAD = CandidateWork(doi="10.9/z", title="Kidney-stone lithotripsy trial",
                    authors=["Zed Z"], year=1999, journal="Urology")


@pytest.mark.asyncio
async def test_single_service_good_match_resolves():
    r = await ReferenceReconciler([StaticResolver("crossref", [GOOD])]).resolve(Q)
    assert r.status == "resolved" and r.doi == "10.1/x" and r.source == "crossref"
    assert r.confidence >= 0.72


@pytest.mark.asyncio
async def test_poor_match_is_unresolved_source():
    r = await ReferenceReconciler([StaticResolver("crossref", [BAD])]).resolve(Q)
    assert r.status == "unresolved_source" and r.doi == ""


@pytest.mark.asyncio
async def test_candidate_without_doi_is_unresolved():
    nodoi = GOOD.model_copy(update={"doi": ""})
    r = await ReferenceReconciler([StaticResolver("crossref", [nodoi])]).resolve(Q)
    assert r.status == "unresolved_source"          # a perfect title but no id to fetch


@pytest.mark.asyncio
async def test_cross_source_agreement_boosts_confidence():
    one = await ReferenceReconciler([StaticResolver("crossref", [GOOD])]).resolve(Q)
    two = await ReferenceReconciler([StaticResolver("crossref", [GOOD]),
                                     StaticResolver("openalex", [GOOD])]).resolve(Q)
    assert two.agreed_sources == ["crossref", "openalex"]
    assert two.confidence > one.confidence          # +agreement bonus


@pytest.mark.asyncio
async def test_no_candidates_is_unresolved():
    r = await ReferenceReconciler([StaticResolver("crossref", [])]).resolve(Q)
    assert r.status == "unresolved_source"


@pytest.mark.asyncio
async def test_failing_resolver_is_skipped_not_fatal():
    class _Boom:
        name = "boom"
        async def resolve(self, query):
            raise RuntimeError("service down")

    r = await ReferenceReconciler([_Boom(), StaticResolver("crossref", [GOOD])]).resolve(Q)
    assert r.status == "resolved"                    # the healthy service still resolves
