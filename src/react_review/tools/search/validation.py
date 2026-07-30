"""Search-strategy validation tools: PubMed reproducibility + per-database counts.

These reproduce a review's literature search (legacy Step 1) — verifying the
reported hit counts, NOT fetching source papers. Reference reconciliation (a
citation → a gated DOI) lives alongside in this package but is a separate
operation; see ``reconciler`` / ``resolve_reference``.
"""
from __future__ import annotations

from react_review.steps.search_validation.interfaces import SearchProvider
from react_review.steps.search_validation.multi_db_count import IdentificationCounter
from react_review.steps.search_validation.schemas import (
    SearchStrategy,
    SearchValidationResult,
)
from react_review.tools.base import Tool, ToolStage
from react_review.tools.models import CountInput, CountResult


class SearchPubMedTool(Tool):
    """Reproduce a PubMed search and return counts + sample results."""

    name = "search_pubmed"
    stage = ToolStage.SEARCH
    input_model = SearchStrategy
    output_model = SearchValidationResult

    def __init__(self, provider: SearchProvider) -> None:
        self._provider = provider

    async def run(self, payload: SearchStrategy) -> SearchValidationResult:
        return await self._provider.validate_strategy(payload)


class CountResultsTool(Tool):
    """Return the total hit count for a query against one free database.

    Registered once per counter (``count_pubmed`` / ``count_europepmc`` /
    ``count_openalex``); the instance ``name`` derives from the counter.
    """

    stage = ToolStage.SEARCH
    input_model = CountInput
    output_model = CountResult

    def __init__(self, counter: IdentificationCounter, name: str | None = None) -> None:
        self._counter = counter
        self.name = name or f"count_{(counter.name or 'db').lower().replace(' ', '')}"

    async def run(self, payload: CountInput) -> CountResult:
        count = await self._counter.count_results(payload.query)
        return CountResult(database=self._counter.name, query=payload.query, count=count)
