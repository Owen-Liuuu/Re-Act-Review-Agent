"""Abstract interface for step 1: search strategy validation."""
from __future__ import annotations

from abc import ABC, abstractmethod

from react_review.steps.search_validation.schemas import (
    SearchStrategy,
    SearchValidationResult,
)


class SearchProvider(ABC):
    """Interface for validating a student's search strategy.

    Implementations may query PubMed, Cochrane, Embase, or other
    databases to reproduce the search and compare result counts.
    """

    @abstractmethod
    async def validate_strategy(
        self, strategy: SearchStrategy
    ) -> SearchValidationResult:
        """Validate a search strategy by attempting to reproduce it.

        Args:
            strategy: The student's reported search strategy.

        Returns:
            Validation result with reproducibility assessment.
        """
