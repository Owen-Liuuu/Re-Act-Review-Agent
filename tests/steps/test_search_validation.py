"""Tests for step 1: search validation."""
from __future__ import annotations

import pytest

from react_review.steps.search_validation.mock_impl import MockSearchProvider
from react_review.steps.search_validation.schemas import SearchStrategy


@pytest.mark.asyncio
async def test_mock_search_provider():
    """MockSearchProvider should return a valid SearchValidationResult."""
    provider = MockSearchProvider()
    strategy = SearchStrategy(
        database="PubMed",
        query_terms=["cancer", "immunotherapy"],
        raw_strategy_text="Search for cancer AND immunotherapy",
    )

    result = await provider.validate_strategy(strategy)

    assert result.original_strategy == strategy
    assert result.reconstructed_query == "cancer AND immunotherapy"
    assert result.is_reproducible is True
    assert result.reported_count is not None
    assert len(result.flags) > 0
