"""Live citation resolvers — response mapping, with httpx mocked (no network)."""
from __future__ import annotations

import pytest

from react_review.tools.search import (
    CrossRefResolver,
    EuropePMCResolver,
    OpenAlexResolver,
    ReferenceQuery,
)
from react_review.tools.search.live_clients import (
    _from_crossref_item,
    _from_europepmc_result,
    _from_openalex_work,
)


class _FakeResp:
    def __init__(self, payload) -> None:
        self._p = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._p


class _FakeClient:
    def __init__(self, payload) -> None:
        self._p = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        return _FakeResp(self._p)


def _patch(monkeypatch, payload) -> None:
    monkeypatch.setattr("react_review.tools.search.live_clients.httpx.AsyncClient",
                        lambda **kw: _FakeClient(payload))


CROSSREF = {"message": {"items": [
    {"DOI": "10.1/X", "title": ["Epicardial fat in T1DM"],
     "author": [{"given": "A", "family": "Ahmad"}],
     "issued": {"date-parts": [[2022]]}, "container-title": ["Journal of Cardiology"]}]}}

OPENALEX = {"results": [
    {"doi": "https://doi.org/10.1/X", "title": "Epicardial fat in T1DM",
     "authorships": [{"author": {"display_name": "A Ahmad"}}],
     "publication_year": 2022,
     "primary_location": {"source": {"display_name": "Journal of Cardiology"}},
     "ids": {"pmcid": "PMC9"}}]}

EUROPEPMC = {"resultList": {"result": [
    {"doi": "10.1/x", "title": "Epicardial fat in T1DM", "authorString": "Ahmad A, Smith B.",
     "pubYear": "2022", "journalTitle": "J Cardiol", "pmcid": "PMC9"}]}}


@pytest.mark.asyncio
async def test_crossref_maps_items(monkeypatch):
    _patch(monkeypatch, CROSSREF)
    cands = await CrossRefResolver(mailto="e@x.com").resolve(
        ReferenceQuery(title="Epicardial fat in T1DM", authors=["Ahmad A"], year=2022))
    assert len(cands) == 1
    c = cands[0]
    assert c.doi == "10.1/x" and c.title == "Epicardial fat in T1DM"     # DOI lower-cased
    assert c.year == 2022 and c.journal == "Journal of Cardiology"
    assert c.authors == ["A Ahmad"] and c.source == "crossref"


@pytest.mark.asyncio
async def test_openalex_maps_results(monkeypatch):
    _patch(monkeypatch, OPENALEX)
    cands = await OpenAlexResolver().resolve(ReferenceQuery(title="Epicardial fat in T1DM"))
    c = cands[0]
    assert c.doi == "10.1/x" and c.journal == "Journal of Cardiology"    # DOI URL stripped
    assert c.authors == ["A Ahmad"] and c.pmcid == "PMC9" and c.source == "openalex"


@pytest.mark.asyncio
async def test_europepmc_maps_results(monkeypatch):
    _patch(monkeypatch, EUROPEPMC)
    cands = await EuropePMCResolver().resolve(ReferenceQuery(title="Epicardial fat in T1DM"))
    c = cands[0]
    assert c.doi == "10.1/x" and c.authors == ["Ahmad A", "Smith B"]     # authorString split
    assert c.year == 2022 and c.pmcid == "PMC9" and c.source == "europepmc"


@pytest.mark.asyncio
async def test_network_error_degrades_to_empty(monkeypatch):
    class _Boom:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            raise RuntimeError("network down")

    monkeypatch.setattr("react_review.tools.search.live_clients.httpx.AsyncClient",
                        lambda **kw: _Boom())
    assert await CrossRefResolver().resolve(ReferenceQuery(title="x")) == []


@pytest.mark.asyncio
async def test_europepmc_identifier_query_uses_pmid(monkeypatch):
    _patch(monkeypatch, EUROPEPMC)
    cands = await EuropePMCResolver().resolve_identifier(
        ReferenceQuery(title="ignored title search", pmid="25249141"))
    assert cands and cands[0].doi == "10.1/x"


@pytest.mark.asyncio
async def test_crossref_identifier_query_uses_doi_path(monkeypatch):
    item = CROSSREF["message"]["items"][0]
    _patch(monkeypatch, {"message": item})
    cands = await CrossRefResolver().resolve_identifier(
        ReferenceQuery(title="ignored", doi="10.1/X"))
    assert cands and cands[0].doi == "10.1/x"


def test_crossref_unreadable_item_returns_none_and_warns():
    from react_review.observe import records, reset

    reset()
    assert _from_crossref_item(
        {"title": ["x"], "issued": {"date-parts": [[[2022]]]}}, "crossref") is None
    blob = " ".join(str(r["message"]) for r in records())
    assert "crossref" in blob
    assert any(r["event"] == "live_clients_parse_failed" for r in records())


def test_openalex_unreadable_item_returns_none_and_warns():
    from react_review.observe import records, reset

    reset()
    assert _from_openalex_work({"authorships": "not-a-list", "title": "x"},
                               "openalex") is None
    blob = " ".join(str(r["message"]) for r in records())
    assert "openalex" in blob
    assert any(r["event"] == "live_clients_parse_failed" for r in records())


def test_europepmc_unreadable_item_returns_none_and_warns():
    from react_review.observe import records, reset

    reset()
    assert _from_europepmc_result({"title": "x", "pubYear": "not-a-year"},
                                  "europepmc") is None
    blob = " ".join(str(r["message"]) for r in records())
    assert "europepmc" in blob
    assert any(r["event"] == "live_clients_parse_failed" for r in records())


@pytest.mark.asyncio
async def test_openalex_repeat_failure_warns_once_then_counts(monkeypatch):
    from react_review.observe import records, reset

    class _Boom:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, params=None):
            raise RuntimeError("429 too many requests")

    monkeypatch.setattr("react_review.tools.search.live_clients.httpx.AsyncClient",
                        lambda **kw: _Boom())
    reset()
    assert await OpenAlexResolver().resolve(ReferenceQuery(title="x")) == []
    assert await OpenAlexResolver().resolve(ReferenceQuery(title="y")) == []
    hits = [r for r in records() if r["event"] == "openalex_resolve_failed"]
    assert len(hits) == 2
    assert hits[0]["count"] == 1
    assert hits[1]["count"] == 2


