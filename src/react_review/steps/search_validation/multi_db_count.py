"""Lightweight count-only providers for Step 1's multi-database identification check.

Step 1 only needs the **total hit count** for each database — we do not
re-download every paper, only verify that the student's reported counts
are reproducible. These providers therefore expose a single
``count_results(query)`` method and skip the heavier sample-result /
PMID-list fetches that ``SearchProvider`` performs.

Three free providers are implemented:

  * ``PubMedCountProvider``      — NCBI E-utilities (``rettype=count``).
  * ``EuropePMCCountProvider``   — Europe PMC REST API.
  * ``OpenAlexCountProvider``    — OpenAlex /works endpoint.

Subscription-only databases (Embase, CINAHL, Cochrane, Web of Science)
have no free API; per project decision #6, Step 1 marks any of these
appearing in the student's strategy as ``UNVERIFIED`` and does NOT
attempt to translate or run their queries.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import httpx
import structlog

from react_review.core.config import PubMedSettings

logger = structlog.get_logger(__name__)


# ----------------------------------------------------------------------
# Common interface
# ----------------------------------------------------------------------


class IdentificationCounter(ABC):
    """Count-only interface used by Step 1's multi-database check.

    Implementations should return ``None`` on transient failures rather
    than raise — Step 1 treats a missing count as "could not verify"
    and surfaces it as a flag instead of aborting.
    """

    name: str = ""

    @abstractmethod
    async def count_results(self, query: str) -> int | None:
        """Return the total result count for the given query, or None on error."""


# ----------------------------------------------------------------------
# PubMed — reuses the same E-utilities endpoint as PubMedSearchProvider
# but only requests the count (rettype=count).
# ----------------------------------------------------------------------


class PubMedCountProvider(IdentificationCounter):
    """Count-only PubMed provider using ``esearch.fcgi?rettype=count``."""

    name = "PubMed"

    def __init__(self, settings: PubMedSettings) -> None:
        self._settings = settings
        self._base_url = settings.base_url.rstrip("/")
        self._common_params: dict[str, str] = {}
        if settings.api_key:
            self._common_params["api_key"] = settings.api_key
        if settings.email:
            self._common_params["email"] = settings.email

    async def count_results(self, query: str) -> int | None:
        if not query.strip():
            return None
        params = {
            "db": "pubmed",
            "term": query,
            "rettype": "count",
            "retmode": "json",
            **self._common_params,
        }
        url = f"{self._base_url}/esearch.fcgi"
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            count = data.get("esearchresult", {}).get("count")
            return int(count) if count is not None else None
        except Exception as exc:
            logger.warning("pubmed_count_failed", query=query[:80], error=str(exc))
            return None


# ----------------------------------------------------------------------
# Europe PMC — covers PubMed plus PMC, agricultural, and additional
# biomedical content. No API key required.
# Docs: https://europepmc.org/RestfulWebService
# ----------------------------------------------------------------------


class EuropePMCCountProvider(IdentificationCounter):
    """Count-only Europe PMC provider using the REST search endpoint."""

    name = "Europe PMC"
    _BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    async def count_results(self, query: str) -> int | None:
        if not query.strip():
            return None
        params = {
            "query": query,
            "format": "json",
            "pageSize": "1",  # we only need the hitCount, not actual records
            "resultType": "lite",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(self._BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            count = data.get("hitCount")
            return int(count) if count is not None else None
        except Exception as exc:
            logger.warning(
                "europe_pmc_count_failed", query=query[:80], error=str(exc)
            )
            return None


# ----------------------------------------------------------------------
# OpenAlex — open citation graph, broad coverage including non-English
# journals. Polite pool: pass an email for higher rate limits.
# Docs: https://docs.openalex.org/api-entities/works/search-works
# ----------------------------------------------------------------------


class OpenAlexCountProvider(IdentificationCounter):
    """Count-only OpenAlex provider using the /works endpoint."""

    name = "OpenAlex"
    _BASE_URL = "https://api.openalex.org/works"

    def __init__(self, mailto: str = "") -> None:
        self._mailto = mailto

    async def count_results(self, query: str) -> int | None:
        if not query.strip():
            return None
        # OpenAlex's free-text ``search`` parameter doesn't accept full
        # PubMed Boolean syntax with field tags ([tiab], [MeSH]). We strip
        # field tags so the query degrades gracefully into a bag-of-words
        # search — accurate enough for an order-of-magnitude reference.
        cleaned_query = _strip_field_tags(query)
        params = {
            "search": cleaned_query,
            "per_page": "1",
        }
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(self._BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            count = data.get("meta", {}).get("count")
            return int(count) if count is not None else None
        except Exception as exc:
            logger.warning(
                "openalex_count_failed", query=query[:80], error=str(exc)
            )
            return None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _strip_field_tags(query: str) -> str:
    """Drop PubMed-style field tags (``[tiab]``, ``[MeSH]`` etc.).

    OpenAlex's free-text search treats ``foo[tiab]`` as the literal token
    ``foo[tiab]``, dropping recall. Stripping the tags converts the query
    into plain words while keeping the Boolean structure (``AND``, ``OR``,
    parentheses) which OpenAlex tolerates.
    """
    import re
    return re.sub(r"\[[A-Za-z]+\]", "", query)


# ----------------------------------------------------------------------
# Database name → counter mapping.
# Only contains entries for databases with free, programmatic access.
# Other databases the student may report (Embase, CINAHL, Cochrane,
# Web of Science) are intentionally absent — Step 1 marks them
# UNVERIFIED rather than guess at counts.
# ----------------------------------------------------------------------


VERIFIABLE_DATABASES = ("PubMed", "Europe PMC", "OpenAlex")
"""Databases Step 1 can independently count (free APIs only)."""

UNVERIFIABLE_DATABASES = (
    "Embase", "CINAHL", "Cochrane", "Web of Science",
)
"""Databases without free APIs; counts shown but not independently verified."""
