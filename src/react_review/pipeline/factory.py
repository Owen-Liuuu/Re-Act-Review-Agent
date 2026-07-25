"""Factory: creates a PipelineOrchestrator from configuration.

Reads config.mock_mode to decide between mock and real implementations.
When mock_mode is False, wires in:
  - PubMedSearchProvider   (Step 1)
  - CrossRefVerifier       (Step 2)
  - FullTextRetriever      (Step 2/3) — PMC full-text → Unpaywall OA → PubMed abstract
  - LLMExtractor × 2       (Step 3) — actual LLM call depends on backend
  - RealTableComparator    (Step 4) — field-level diff comparison
  - RealReportGenerator    (Step 4) — aggregated evaluation report
"""
from __future__ import annotations

from react_review.core.config import AppConfig
from react_review.llm.base import LLMBackend
from react_review.llm.mock_backend import MockLLMBackend
from react_review.pipeline.orchestrator import PipelineOrchestrator

# Mock implementations
from react_review.steps.search_validation.mock_impl import MockSearchProvider
from react_review.steps.paper_verification.mock_impl import (
    MockPaperRetriever,
    MockReferenceVerifier,
)
from react_review.steps.data_extraction.mock_impl import (
    MockExtractorA,
    MockExtractorB,
)
from react_review.steps.table_comparison.mock_impl import (
    MockReportGenerator,
    MockTableComparator,
)

# Real implementations
from react_review.steps.search_validation.pubmed_impl import PubMedSearchProvider
from react_review.steps.search_validation.multi_db_count import (
    EuropePMCCountProvider,
    IdentificationCounter,
    OpenAlexCountProvider,
    PubMedCountProvider,
)
from react_review.steps.paper_verification.crossref_impl import CrossRefVerifier
from react_review.steps.data_extraction.llm_extractor import LLMExtractor


def _create_llm_backend(config: AppConfig) -> LLMBackend:
    """Create the primary LLM backend from ``config.llm``."""
    return _create_backend_from_settings(config.llm)


def _create_backend_from_settings(settings: "LLMSettings") -> LLMBackend:
    """Create an LLM backend from explicit settings.

    This is the shared factory used for both ``llm`` and ``llm2``.
    """
    from react_review.core.config import LLMSettings  # noqa: F811

    provider = settings.provider.lower()

    if provider == "mock":
        return MockLLMBackend()

    if provider == "openai":
        from react_review.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(settings)

    if provider == "anthropic":
        from react_review.llm.claude_backend import ClaudeBackend
        return ClaudeBackend(settings)

    if provider == "gemini":
        from react_review.llm.gemini_backend import GeminiBackend
        return GeminiBackend(settings)

    if provider == "qwen":
        from react_review.llm.qwen_backend import QwenBackend
        return QwenBackend(settings)

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        "Supported: mock, openai, anthropic, gemini, qwen."
    )


def create_pipeline(config: AppConfig) -> PipelineOrchestrator:
    """Create a pipeline orchestrator wired with the right implementations.

    When ``config.mock_mode`` is True, all steps use mock implementations.
    When False:
      - Step 1: PubMedSearchProvider (real PubMed API)
      - Step 2: CrossRefVerifier (real CrossRef API)
      - Step 3: LLMExtractor (real LLM — depends on llm.provider)
      - Step 4: MockTableComparator / MockReportGenerator (placeholder)

    Args:
        config: Application configuration.

    Returns:
        Fully wired PipelineOrchestrator.
    """
    if config.mock_mode:
        return PipelineOrchestrator(
            search_provider=MockSearchProvider(),
            reference_verifier=MockReferenceVerifier(),
            paper_retriever=MockPaperRetriever(),
            extractors=[MockExtractorA(), MockExtractorB()],
            table_comparator=MockTableComparator(),
            report_generator=MockReportGenerator(),
            counters=[],  # mock mode: all databases marked UNVERIFIED
            enabled_steps=config.enabled_steps,
        )

    # --- Real implementations ---

    # Step 1: PubMed reproducibility provider (single-DB sample-result check)
    search_provider = PubMedSearchProvider(settings=config.pubmed)

    # Step 1: multi-DB identification counters (PubMed + Europe PMC + OpenAlex).
    # Subscription-only databases (Embase, CINAHL, Cochrane, Web of Science)
    # have no free API and are intentionally not represented here.
    openalex_mailto = (
        config.unpaywall.email or config.pubmed.email or config.crossref.mailto
    )
    counters: list[IdentificationCounter] = [
        PubMedCountProvider(settings=config.pubmed),
        EuropePMCCountProvider(),
        OpenAlexCountProvider(mailto=openalex_mailto),
    ]

    # Step 2: CrossRef reference verification
    reference_verifier = CrossRefVerifier(
        settings=config.crossref,
        thresholds=config.thresholds,
    )

    # Step 2/3: Full-text retriever (PMC → Unpaywall → PubMed abstract)
    from react_review.steps.paper_verification.fulltext_retriever import FullTextRetriever
    unpaywall_email = config.unpaywall.email or config.pubmed.email
    paper_retriever = FullTextRetriever(
        pubmed_settings=config.pubmed,
        unpaywall_email=unpaywall_email,
    )

    # Step 3: LLM-based data extraction (dual-LLM cross-validation)
    llm_backend = _create_llm_backend(config)
    extractors = [
        LLMExtractor(backend=llm_backend, extractor_name=f"{llm_backend.model_id}-extractor"),
    ]

    # If llm2 is configured, add a second extractor with a different LLM
    # for genuine cross-validation (e.g. Qwen + Gemini)
    llm2_backend: LLMBackend | None = None
    if config.llm2 and config.llm2.provider != "mock":
        llm2_backend = _create_backend_from_settings(config.llm2)
        extractors.append(
            LLMExtractor(backend=llm2_backend, extractor_name=f"{llm2_backend.model_id}-extractor"),
        )

    # Step 4: Table comparison (real field-level diff)
    from react_review.steps.table_comparison.real_impl import (
        RealTableComparator,
        RealReportGenerator,
    )
    table_comparator = RealTableComparator()
    report_generator = RealReportGenerator()

    return PipelineOrchestrator(
        search_provider=search_provider,
        reference_verifier=reference_verifier,
        paper_retriever=paper_retriever,
        extractors=extractors,
        table_comparator=table_comparator,
        report_generator=report_generator,
        counters=counters,
        llm_backend=llm_backend,
        enabled_steps=config.enabled_steps,
    )
