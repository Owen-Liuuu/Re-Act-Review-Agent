"""Reconciler per-run cache + live-client rate limiting / 429 backoff (offline)."""
from __future__ import annotations

import pytest

import react_review.tools.search.live_clients as live_clients
from react_review.tools.search import (
    CandidateWork,
    CrossRefResolver,
    ReferenceQuery,
    ReferenceReconciler,
)

Q = ReferenceQuery(title="Epicardial fat thickness in type 1 diabetes",
                   authors=["Ahmad A", "Smith B"], year=2022, journal="Journal of Cardiology")
GOOD = CandidateWork(doi="10.1/x", title="Epicardial fat thickness in type 1 diabetes",
                     authors=["Ahmad A", "Jones C"], year=2022,
                     journal="Journal of Cardiology", source="crossref")


class _Counting:
    name = "crossref"

    def __init__(self, cands):
        self.calls = 0
        self._c = cands

    async def resolve(self, query):
        self.calls += 1
        return list(self._c)


# --- reconciler cache ---

@pytest.mark.asyncio
async def test_reconciler_caches_repeated_query():
    r = _Counting([GOOD])
    rec = ReferenceReconciler([r])
    await rec.resolve(Q)
    await rec.resolve(Q)                      # identical citation → served from cache
    assert r.calls == 1                       # the service was hit once, not twice


@pytest.mark.asyncio
async def test_reconciler_does_not_cache_different_queries():
    r = _Counting([GOOD])
    rec = ReferenceReconciler([r])
    await rec.resolve(Q)
    await rec.resolve(ReferenceQuery(title="A completely different paper", year=1999))
    assert r.calls == 2


# --- rate limiter ---

@pytest.mark.asyncio
async def test_rate_limiter_spaces_consecutive_calls(monkeypatch):
    clock = [100.0]
    slept: list[float] = []
    monkeypatch.setattr(live_clients.time, "monotonic", lambda: clock[0])

    async def fake_sleep(d):
        slept.append(d)
        clock[0] += d

    monkeypatch.setattr(live_clients.asyncio, "sleep", fake_sleep)
    lim = live_clients._RateLimiter(min_interval=1.0)
    await lim.acquire()                       # first call: no wait
    await lim.acquire()                       # immediate: waits the full interval
    assert slept == [1.0]


@pytest.mark.asyncio
async def test_rate_limiter_zero_interval_never_sleeps(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(d):
        slept.append(d)

    monkeypatch.setattr(live_clients.asyncio, "sleep", fake_sleep)
    lim = live_clients._RateLimiter(min_interval=0.0)
    await lim.acquire()
    await lim.acquire()
    assert slept == []


# --- 429 backoff ---

class _Resp:
    def __init__(self, code, payload):
        self.status_code = code
        self._p = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._p


class _SharedClient:
    """One httpx client instance per attempt, but sharing a response queue index."""

    def __init__(self, resps, idx):
        self._resps = resps
        self._idx = idx

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        r = self._resps[min(self._idx[0], len(self._resps) - 1)]
        self._idx[0] += 1
        return r


_CROSSREF = {"message": {"items": [
    {"DOI": "10.1/x", "title": ["T"], "author": [],
     "issued": {"date-parts": [[2022]]}, "container-title": ["J"]}]}}


def _patch_http(monkeypatch, resps, idx):
    monkeypatch.setattr(live_clients.httpx, "AsyncClient",
                        lambda **kw: _SharedClient(resps, idx))

    async def fast_sleep(d):
        pass

    monkeypatch.setattr(live_clients.asyncio, "sleep", fast_sleep)


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds(monkeypatch):
    idx = [0]
    _patch_http(monkeypatch, [_Resp(429, {}), _Resp(200, _CROSSREF)], idx)
    cands = await CrossRefResolver().resolve(ReferenceQuery(title="x"))
    assert len(cands) == 1 and idx[0] == 2         # retried once, then 200


@pytest.mark.asyncio
async def test_gives_up_after_retries_and_degrades(monkeypatch):
    idx = [0]
    _patch_http(monkeypatch, [_Resp(429, {})], idx)   # always throttled
    cands = await CrossRefResolver().resolve(ReferenceQuery(title="x"))
    assert cands == [] and idx[0] == 3               # exhausted retries → graceful []


@pytest.mark.asyncio
async def test_openalex_does_not_retry_429(monkeypatch):
    from react_review.tools.search import OpenAlexResolver

    idx = [0]
    _patch_http(monkeypatch, [_Resp(429, {})], idx)
    cands = await OpenAlexResolver().resolve(ReferenceQuery(title="x"))
    assert cands == [] and idx[0] == 1
