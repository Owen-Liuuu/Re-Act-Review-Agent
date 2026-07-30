"""Live citation resolvers — response mapping, with httpx mocked (no network)."""
from __future__ import annotations

import pytest

from react_review.tools.search import (
    CrossRefResolver,
    EuropePMCResolver,
    OpenAlexResolver,
    ReferenceQuery,
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
