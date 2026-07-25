"""Real implementation for step 1: PubMed E-utilities search validation.

Uses three NCBI E-utilities endpoints:
  - esearch.fcgi : search PubMed by query terms, returns PMID list + total count
  - esummary.fcgi: retrieve summary-level metadata for a list of PMIDs
  - efetch.fcgi  : retrieve full XML records for a list of PMIDs

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""
from __future__ import annotations

import httpx
import structlog

from react_review.core.config import PubMedSettings
from react_review.core.enums import ValidationSeverity
from react_review.steps.search_validation.interfaces import SearchProvider
from react_review.steps.search_validation.schemas import (
    PubMedSummary,
    SearchStrategy,
    SearchValidationFlag,
    SearchValidationResult,
)

logger = structlog.get_logger(__name__)


class PubMedSearchProvider(SearchProvider):
    """Validates a student's search strategy by re-running it against PubMed.

    Args:
        settings: PubMed API configuration (api_key, base_url, etc.).
    """

    def __init__(self, settings: PubMedSettings) -> None:
        self._settings = settings
        self._base_url = settings.base_url.rstrip("/")
        self._common_params: dict[str, str] = {}
        if settings.api_key:
            self._common_params["api_key"] = settings.api_key
        if settings.email:
            self._common_params["email"] = settings.email

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def validate_strategy(
        self, strategy: SearchStrategy
    ) -> SearchValidationResult:
        """Reproduce the student's search on PubMed and compare counts."""
        query = self._build_query(strategy)
        logger.info(
            "pubmed_search_start",
            database=strategy.database,
            query=query[:120],
        )

        # Step A: esearch — get total count + PMID list (up to 500 for cross-check)
        search_count, pmid_list = await self._esearch(query, retmax=500)

        # Step B: compare with student-reported count
        flags = self._compare_counts(strategy.reported_result_count, search_count)

        # Step C: esummary for a sample of results (for cross-checking selected papers)
        sample_summaries: list[PubMedSummary] = []
        if pmid_list:
            sample_pmids = pmid_list[:20]
            raw_summaries = await self._esummary(sample_pmids)
            sample_summaries = [
                PubMedSummary(**s) for s in raw_summaries
            ]

        is_reproducible = search_count > 0 and len(flags) == 0

        return SearchValidationResult(
            original_strategy=strategy,
            reconstructed_query=query,
            reported_count=strategy.reported_result_count,
            actual_count=search_count,
            is_reproducible=is_reproducible,
            sample_results=sample_summaries,
            all_pmids=pmid_list,
            flags=flags,
            notes=f"PubMed returned {search_count} results for query: {query[:80]}",
        )

    # ------------------------------------------------------------------
    # PubMed E-utilities calls
    # ------------------------------------------------------------------

    async def _esearch(self, query: str, retmax: int = 500) -> tuple[int, list[str]]:
        """Call esearch.fcgi to search PubMed.

        Returns:
            Tuple of (total_count, list_of_pmids).
        """
        url = f"{self._base_url}/esearch.fcgi"
        params = {
            **self._common_params,
            "db": "pubmed",
            "term": query,
            "rettype": "json",
            "retmode": "json",
            "retmax": retmax,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        esearch_result = data.get("esearchresult", {})
        count = int(esearch_result.get("count", 0))
        id_list = esearch_result.get("idlist", [])

        logger.info("pubmed_esearch_done", count=count, returned_ids=len(id_list))
        return count, id_list

    async def _esummary(self, pmids: list[str]) -> list[dict]:
        """Call esummary.fcgi to get summary metadata for PMIDs."""
        if not pmids:
            return []

        url = f"{self._base_url}/esummary.fcgi"
        params = {
            **self._common_params,
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        result_block = data.get("result", {})
        summaries = []
        for pmid in pmids:
            if pmid in result_block:
                entry = result_block[pmid]
                summaries.append({
                    "pmid": pmid,
                    "title": entry.get("title", ""),
                    "source": entry.get("source", ""),
                    "pubdate": entry.get("pubdate", ""),
                    "authors": [
                        a.get("name", "") for a in entry.get("authors", [])
                    ],
                    "doi": self._extract_doi_from_articleids(
                        entry.get("articleids", [])
                    ),
                })

        logger.info("pubmed_esummary_done", count=len(summaries))
        return summaries

    async def _efetch(self, pmids: list[str]) -> str:
        """Call efetch.fcgi to get full XML records."""
        if not pmids:
            return ""

        url = f"{self._base_url}/efetch.fcgi"
        params = {
            **self._common_params,
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()

        logger.info("pubmed_efetch_done", pmid_count=len(pmids))
        return resp.text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(strategy: SearchStrategy) -> str:
        """Reconstruct a PubMed query string from a SearchStrategy."""
        if strategy.query_terms:
            query = " AND ".join(strategy.query_terms)
        elif strategy.raw_strategy_text:
            query = strategy.raw_strategy_text
        else:
            query = ""

        if strategy.date_range:
            query += f" AND ({strategy.date_range}[dp])"

        return query

    @staticmethod
    def _compare_counts(
        reported_count: int | None, actual_count: int
    ) -> list[SearchValidationFlag]:
        """Compare student-reported result count with actual PubMed count."""
        flags: list[SearchValidationFlag] = []

        if reported_count is None:
            return flags

        if reported_count == 0 and actual_count == 0:
            return flags

        if actual_count == 0:
            flags.append(
                SearchValidationFlag(
                    code="ZERO_RESULTS",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"PubMed returned 0 results but student reported {reported_count}. "
                        "The search query may be incorrectly reconstructed."
                    ),
                )
            )
            return flags

        diff_ratio = abs(reported_count - actual_count) / max(reported_count, actual_count)

        if diff_ratio > 0.3:
            flags.append(
                SearchValidationFlag(
                    code="COUNT_MAJOR_MISMATCH",
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Reported count ({reported_count}) differs from actual ({actual_count}) "
                        f"by {diff_ratio:.1%}. Significant discrepancy."
                    ),
                )
            )
        elif diff_ratio > 0.05:
            flags.append(
                SearchValidationFlag(
                    code="COUNT_MINOR_MISMATCH",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Reported count ({reported_count}) differs from actual ({actual_count}) "
                        f"by {diff_ratio:.1%}. Minor — may be due to search date."
                    ),
                )
            )

        return flags

    @staticmethod
    def _extract_doi_from_articleids(article_ids: list[dict]) -> str:
        """Extract DOI from esummary articleids list."""
        for aid in article_ids:
            if aid.get("idtype") == "doi":
                return aid.get("value", "")
        return ""
