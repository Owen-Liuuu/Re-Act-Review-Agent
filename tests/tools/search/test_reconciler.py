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


@pytest.mark.asyncio
async def test_wrong_journal_candidate_is_unresolved_with_mismatch_note():
    query = ReferenceQuery(
        title="Minimally invasive esophagectomy after neoadjuvant therapy",
        year=2023, journal="Front Oncol")
    wrong = CandidateWork(
        doi="10.1093/dote/doad052.248",
        title="Minimally invasive esophagectomy after neoadjuvant therapy",
        year=2023, journal="Diseases of the Esophagus", source="crossref")
    r = await ReferenceReconciler([StaticResolver("crossref", [wrong])]).resolve(query)
    assert r.status == "unresolved_source"
    assert r.doi == ""
    assert r.candidates_seen == 1
    assert r.note == "retrieved 1 candidates, none matched the citation"
    from react_review.steps.paper_verification.schemas import ReferenceEntry
    from react_review.tools.search.reconciler import unresolved_note_for
    note = unresolved_note_for(ReferenceEntry(
        title=query.title, year=2023, journal="Front Oncol"))
    assert note == r.note


@pytest.mark.asyncio
async def test_pmid_identifier_skips_the_candidate_gate():
    query = ReferenceQuery(
        title="Minimally invasive esophagectomy after neoadjuvant therapy",
        year=2023, journal="Front Oncol", pmid="37251945")
    mismatch_journal = CandidateWork(
        doi="10.3389/fonc.2023.1104109",
        title="Minimally invasive esophagectomy after neoadjuvant therapy",
        year=2023, journal="Wrong Journal That Does Not Abbreviate",
        pmid="37251945", source="europepmc")
    r = await ReferenceReconciler(
        [StaticResolver("europepmc", [mismatch_journal])]).resolve(query)
    assert r.status == "resolved" and r.doi == "10.3389/fonc.2023.1104109"
    assert r.note == ""


@pytest.mark.asyncio
async def test_doi_identifier_skips_the_candidate_gate():
    query = ReferenceQuery(
        title="Robotic esophagectomy", year=2025,
        journal="Langenbecks Arch Surg", doi="10.1007/s00423-025-03877-4")
    mismatch_journal = CandidateWork(
        doi="10.1007/s00423-025-03877-4", title="Robotic esophagectomy",
        year=2025, journal="Completely Unrelated Journal", source="crossref")
    r = await ReferenceReconciler(
        [StaticResolver("crossref", [mismatch_journal])]).resolve(query)
    assert r.status == "resolved" and r.doi == "10.1007/s00423-025-03877-4"


@pytest.mark.asyncio
async def test_title_search_mismatch_note_is_absent_on_identifier_path():
    query = ReferenceQuery(
        title="Capovilla", year=2023, journal="Front Oncol", pmid="37251945")
    r = await ReferenceReconciler([StaticResolver("crossref", [])]).resolve(query)
    assert r.status == "unresolved_source"
    assert r.note == ""
    assert "candidates, none matched" not in (r.note or "")

