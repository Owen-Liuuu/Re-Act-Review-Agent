"""Build the typed tool catalogue (a ToolRegistry) from configuration.

Mirrors the old pipeline factory's mock/real split, but produces a registry of
typed tools instead of a monolithic orchestrator. Currently registers 8 tools
across the four stages:

    Search : search_pubmed, count_pubmed, count_europepmc, count_openalex
    Verify : verify_reference
    Extract: fetch_fulltext, extract_fields
    Compare: compare_values

The catalogue grows toward the Proposal's nine as normalize_field (P1.B) and a
resolve/validate split (P2) land.
"""
from __future__ import annotations

from react_review.audit import ToleranceTable
from react_review.core.config import AppConfig
from react_review.steps.search_validation.multi_db_count import IdentificationCounter
from react_review.tools.base import Tool
from react_review.tools.compare import CompareValuesTool
from react_review.tools.extract import ExtractFieldsTool, FetchFullTextTool
from react_review.tools.registry import ToolRegistry
from react_review.tools.search import CountResultsTool, SearchPubMedTool
from react_review.tools.verify import VerifyReferenceTool


class _StubCounter(IdentificationCounter):
    """A deterministic count provider for mock mode (no network)."""

    def __init__(self, name: str, count: int = 142) -> None:
        self.name = name
        self._count = count

    async def count_results(self, query: str) -> int | None:
        return self._count


def build_catalogue(
    config: AppConfig,
    *,
    tolerance: ToleranceTable | None = None,
) -> ToolRegistry:
    """Construct and register the tool catalogue for ``config``."""
    tol = tolerance or ToleranceTable()
    reg = ToolRegistry()

    if config.mock_mode:
        from react_review.steps.search_validation.mock_impl import MockSearchProvider
        from react_review.steps.paper_verification.mock_impl import (
            MockPaperRetriever,
            MockReferenceVerifier,
        )
        from react_review.steps.data_extraction.mock_impl import MockExtractorA

        search_provider = MockSearchProvider()
        verifier = MockReferenceVerifier()
        retriever = MockPaperRetriever()
        extractor = MockExtractorA()
        counters: list[IdentificationCounter] = [
            _StubCounter("PubMed"),
            _StubCounter("Europe PMC"),
            _StubCounter("OpenAlex"),
        ]
    else:
        from react_review.steps.search_validation.pubmed_impl import PubMedSearchProvider
        from react_review.steps.search_validation.multi_db_count import (
            EuropePMCCountProvider,
            OpenAlexCountProvider,
            PubMedCountProvider,
        )
        from react_review.steps.paper_verification.crossref_impl import CrossRefVerifier
        from react_review.steps.paper_verification.fulltext_retriever import (
            FullTextRetriever,
        )
        from react_review.steps.data_extraction.llm_extractor import LLMExtractor
        from react_review.pipeline.factory import _create_llm_backend

        search_provider = PubMedSearchProvider(settings=config.pubmed)
        openalex_mailto = (
            config.unpaywall.email or config.pubmed.email or config.crossref.mailto
        )
        counters = [
            PubMedCountProvider(settings=config.pubmed),
            EuropePMCCountProvider(),
            OpenAlexCountProvider(mailto=openalex_mailto),
        ]
        verifier = CrossRefVerifier(settings=config.crossref, thresholds=config.thresholds)
        retriever = FullTextRetriever(
            pubmed_settings=config.pubmed,
            unpaywall_email=config.unpaywall.email or config.pubmed.email,
        )
        backend = _create_llm_backend(config)
        extractor = LLMExtractor(
            backend=backend, extractor_name=f"{backend.model_id}-extractor"
        )

    tools: list[Tool] = [
        SearchPubMedTool(search_provider),
        *(CountResultsTool(c) for c in counters),
        VerifyReferenceTool(verifier),
        FetchFullTextTool(retriever),
        ExtractFieldsTool(extractor),
        CompareValuesTool(tol),
    ]
    for t in tools:
        reg.register(t)
    return reg
