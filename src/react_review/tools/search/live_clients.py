"""Live citation resolvers over the public bibliographic APIs.

Each turns a :class:`ReferenceQuery` into candidate works from ONE service, for
the reconciler's gate to score. No API key needed; passing an email joins the
polite pool. Requests are spaced by a per-service rate limiter and retried with
exponential backoff on HTTP 429/503; any other network/parse error degrades to
``[]`` (one service down must not sink the rest). Unit tests mock ``httpx``; a
live smoke exercises the real API.
"""
from __future__ import annotations

import asyncio
import re
import time

import httpx
import structlog

from react_review.normalize.doi import normalize_doi
from react_review.tools.search.models import CandidateWork, ReferenceQuery

logger = structlog.get_logger(__name__)


class _RateLimiter:
    """Minimum-interval async gate — one per service (services differ in limits)."""

    def __init__(self, min_interval: float = 0.0) -> None:
        self._min = min_interval
        self._last: float | None = None
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._min <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if self._last is not None:
                wait = self._min - (now - self._last)
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.monotonic()
            self._last = now


async def _get_json(
    url: str, params: dict, *, timeout: float, limiter: _RateLimiter,
    retries: int = 3, backoff_cap: float = 8.0,
) -> dict:
    """Rate-limited GET → JSON, retrying with exponential backoff on 429/503."""
    resp = None
    for attempt in range(retries):
        await limiter.acquire()
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params)
        if getattr(resp, "status_code", 200) in (429, 503) and attempt < retries - 1:
            await asyncio.sleep(min(2 ** attempt, backoff_cap))
            continue
        break
    resp.raise_for_status()
    return resp.json()


def _surnames(authors: list[str]) -> list[str]:
    out: list[str] = []
    for a in authors:
        m = re.search(r"[A-Za-z][A-Za-z\-']+", a or "")
        if m:
            out.append(m.group(0))
    return out


class CrossRefResolver:
    """Resolve a citation via CrossRef ``/works`` (bibliographic / title query)."""

    name = "crossref"

    def __init__(self, *, base_url: str = "https://api.crossref.org",
                 mailto: str = "", timeout: float = 30.0, rows: int = 5,
                 min_interval: float = 0.05) -> None:
        self._base_url = base_url.rstrip("/")
        self._mailto = mailto
        self._timeout = timeout
        self._rows = rows
        self._limiter = _RateLimiter(min_interval)

    async def resolve(self, query: ReferenceQuery) -> list[CandidateWork]:
        params: dict[str, str | int] = {"rows": self._rows}
        if query.citation:
            params["query.bibliographic"] = query.citation
        else:
            params["query.title"] = query.title
            if query.authors:
                params["query.author"] = " ".join(_surnames(query.authors))
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            data = await _get_json(f"{self._base_url}/works", params,
                                   timeout=self._timeout, limiter=self._limiter)
            items = data.get("message", {}).get("items", [])
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("crossref_resolve_failed", error=str(exc)[:120])
            return []
        out: list[CandidateWork] = []
        for it in items:
            try:
                titles = it.get("title") or [""]
                journals = it.get("container-title") or [""]
                authors = [" ".join(x for x in (a.get("given"), a.get("family")) if x).strip()
                           for a in (it.get("author") or [])]
                parts = (it.get("issued") or it.get("published") or {}).get("date-parts") or [[None]]
                year = parts[0][0] if parts and parts[0] else None
                out.append(CandidateWork(
                    doi=normalize_doi(it.get("DOI")), title=titles[0], authors=authors,
                    year=int(year) if year else None, journal=journals[0], source=self.name))
            except Exception:                                      # noqa: BLE001
                continue
        return out


class OpenAlexResolver:
    """Resolve a citation via OpenAlex ``/works?search=``."""

    name = "openalex"

    def __init__(self, *, base_url: str = "https://api.openalex.org",
                 mailto: str = "", timeout: float = 30.0, per_page: int = 5,
                 min_interval: float = 0.1) -> None:
        self._base_url = base_url.rstrip("/")
        self._mailto = mailto
        self._timeout = timeout
        self._per_page = per_page
        self._limiter = _RateLimiter(min_interval)

    async def resolve(self, query: ReferenceQuery) -> list[CandidateWork]:
        params: dict[str, str | int] = {
            "search": query.title or query.citation, "per_page": self._per_page}
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            data = await _get_json(f"{self._base_url}/works", params,
                                   timeout=self._timeout, limiter=self._limiter)
            results = data.get("results", [])
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("openalex_resolve_failed", error=str(exc)[:120])
            return []
        out: list[CandidateWork] = []
        for w in results:
            try:
                loc = w.get("primary_location") or w.get("host_venue") or {}
                src = loc.get("source") or loc
                authors = [(a.get("author") or {}).get("display_name", "")
                           for a in (w.get("authorships") or [])]
                out.append(CandidateWork(
                    doi=normalize_doi(w.get("doi")),
                    title=w.get("title") or w.get("display_name") or "",
                    authors=[a for a in authors if a],
                    year=w.get("publication_year"),
                    journal=src.get("display_name", "") if isinstance(src, dict) else "",
                    pmcid=(w.get("ids") or {}).get("pmcid", ""), source=self.name))
            except Exception:                                      # noqa: BLE001
                continue
        return out


class EuropePMCResolver:
    """Resolve a citation via Europe PMC REST ``/search`` (also yields PMCID)."""

    name = "europepmc"

    def __init__(self, *, base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest",
                 timeout: float = 30.0, page_size: int = 5, min_interval: float = 0.1) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._page_size = page_size
        self._limiter = _RateLimiter(min_interval)

    async def resolve(self, query: ReferenceQuery) -> list[CandidateWork]:
        params = {"query": query.title or query.citation, "format": "json",
                  "pageSize": self._page_size}
        try:
            data = await _get_json(f"{self._base_url}/search", params,
                                   timeout=self._timeout, limiter=self._limiter)
            results = (data.get("resultList") or {}).get("result", [])
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("europepmc_resolve_failed", error=str(exc)[:120])
            return []
        out: list[CandidateWork] = []
        for r in results:
            try:
                authors = [a.strip() for a in (r.get("authorString") or "").rstrip(".").split(",")
                           if a.strip()]
                year = r.get("pubYear")
                out.append(CandidateWork(
                    doi=normalize_doi(r.get("doi")), title=r.get("title") or "",
                    authors=authors, year=int(year) if year else None,
                    journal=r.get("journalTitle") or "", pmcid=r.get("pmcid") or "",
                    source=self.name))
            except Exception:                                      # noqa: BLE001
                continue
        return out
