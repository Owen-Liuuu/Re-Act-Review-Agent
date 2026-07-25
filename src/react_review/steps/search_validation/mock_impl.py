"""Mock implementation for step 1: search strategy validation."""
from __future__ import annotations

from react_review.core.enums import ValidationSeverity
from react_review.steps.search_validation.interfaces import SearchProvider
from react_review.steps.search_validation.schemas import (
    PubMedSummary,
    SearchStrategy,
    SearchValidationFlag,
    SearchValidationResult,
)


class MockSearchProvider(SearchProvider):
    """Returns fake validation results for testing."""

    async def validate_strategy(
        self, strategy: SearchStrategy
    ) -> SearchValidationResult:
        reconstructed = (
            " AND ".join(strategy.query_terms)
            if strategy.query_terms
            else strategy.raw_strategy_text or "cancer AND treatment"
        )

        return SearchValidationResult(
            original_strategy=strategy,
            reconstructed_query=reconstructed,
            reported_count=strategy.reported_result_count or 150,
            actual_count=142,
            is_reproducible=True,
            sample_results=[
                PubMedSummary(
                    pmid="12345678",
                    title="Mock paper about cancer treatment",
                    authors=["Author A"],
                    doi="10.1234/mock-sample-001",
                ),
            ],
            all_pmids=["12345678", "12345679", "12345680"],
            flags=[
                SearchValidationFlag(
                    code="COUNT_MINOR_MISMATCH",
                    severity=ValidationSeverity.WARNING,
                    message="Reported count (150) differs from actual (142) by 5.3%.",
                )
            ],
            notes="Mock validation: search strategy appears largely reproducible.",
        )
