"""Build the typed tool catalogue (a ToolRegistry) from configuration.

Splits mock from real implementations and produces a registry of typed tools
rather than a monolithic orchestrator. Registers the tools across the four stages
(field_type mapping is the Parser's FieldResolver, not a tool):

    Search : search_pubmed, count_pubmed, count_europepmc, count_openalex, resolve_reference
    Verify : verify_reference
    Extract: fetch_fulltext, extract_source_value, extract_source_batch, ocr_forest_plot
    Compare: compare_values
"""
from __future__ import annotations

from pathlib import Path

from react_review.audit import ToleranceTable
from react_review.core.config import AppConfig
from react_review.steps.search_validation.multi_db_count import IdentificationCounter
from react_review.tools.base import Tool
from react_review.tools.compare import CompareValuesTool
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_batch import ExtractSourceBatchTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.forest_ocr import ForestOcrTool
from react_review.tools.registry import ToolRegistry
from react_review.tools.search import (
    CountResultsTool,
    CrossRefResolver,
    EuropePMCResolver,
    OpenAlexResolver,
    ReferenceReconciler,
    ResolveReferenceTool,
    SearchPubMedTool,
    StaticResolver,
)
from react_review.tools.verify import VerifyReferenceTool

class _StubCounter(IdentificationCounter):
    """A deterministic count provider for mock mode (no network)."""

    def __init__(self, name: str, count: int = 142) -> None:
        self.name = name
        self._count = count

    async def count_results(self, query: str) -> int | None:
        return self._count


def _default_tolerance() -> ToleranceTable:
    """Load the shipped configs/tolerances.yaml if present, else code defaults.

    Loading the repo config by default avoids drift when someone edits the YAML
    but calls ``build_catalogue`` without an explicit tolerance. Degrades to the
    built-in 1% / 3% defaults if the file is unavailable (e.g. installed wheel).
    """
    cfg = Path(__file__).resolve().parents[3] / "configs" / "tolerances.yaml"
    try:
        if cfg.exists():
            return ToleranceTable.from_yaml(cfg)
    except Exception:
        pass
    return ToleranceTable()


def build_catalogue(
    config: AppConfig,
    *,
    tolerance: ToleranceTable | None = None,
) -> ToolRegistry:
    """Construct and register the tool catalogue for ``config``."""
    tol = tolerance or _default_tolerance()
    reg = ToolRegistry()

    if config.mock_mode:
        from react_review.steps.search_validation.mock_impl import MockSearchProvider
        from react_review.steps.paper_verification.mock_impl import (
            MockPaperRetriever,
            MockReferenceVerifier,
        )
        from react_review.llm.mock_backend import MockLLMBackend

        search_provider = MockSearchProvider()
        verifier = MockReferenceVerifier()
        retriever = MockPaperRetriever()
        norm_backend = MockLLMBackend()
        counters: list[IdentificationCounter] = [
            _StubCounter("PubMed"),
            _StubCounter("Europe PMC"),
            _StubCounter("OpenAlex"),
        ]
        reconciler = ReferenceReconciler([StaticResolver("mock", [])])  # offline: no network
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
        from react_review.llm.factory import create_llm_backend, create_vision_backend

        search_provider = PubMedSearchProvider(settings=config.pubmed)
        openalex_mailto = (
            config.unpaywall.email or config.pubmed.email or config.crossref.mailto
        )
        counters = [
            PubMedCountProvider(settings=config.pubmed),
            EuropePMCCountProvider(),
            OpenAlexCountProvider(mailto=openalex_mailto),
        ]
        reconciler = ReferenceReconciler([
            CrossRefResolver(base_url=config.crossref.base_url, mailto=openalex_mailto,
                             timeout=config.crossref.timeout),
            OpenAlexResolver(mailto=openalex_mailto, timeout=config.crossref.timeout),
            EuropePMCResolver(timeout=config.crossref.timeout),
        ])
        verifier = CrossRefVerifier(settings=config.crossref, thresholds=config.thresholds)
        retriever = FullTextRetriever(
            pubmed_settings=config.pubmed,
            unpaywall_email=config.unpaywall.email or config.pubmed.email,
            unpaywall_enabled=config.unpaywall.enabled,
        )
        norm_backend = create_llm_backend(config)

    tools: list[Tool] = [
        SearchPubMedTool(search_provider),
        *(CountResultsTool(c) for c in counters),
        VerifyReferenceTool(verifier),
        FetchFullTextTool(retriever),
        ExtractSourceValueTool(norm_backend),
        # Registered here too, so a contract that routes to it finds
        # it. This module builds tools; the runtime that clears an
        # aggregation is injected where a Collector is built.
        ExtractSourceBatchTool(norm_backend),
        ResolveReferenceTool(reconciler),
        CompareValuesTool(tol),
        ForestOcrTool(norm_backend, vision_backend=(
            None if config.mock_mode else create_vision_backend(config))),
    ]
    for t in tools:
        reg.register(t)
    return reg
