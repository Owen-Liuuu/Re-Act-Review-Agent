"""Abstract interfaces for step 2: paper existence verification."""
from __future__ import annotations

from abc import ABC, abstractmethod

from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import (
    ReferenceEntry,
    ReferenceVerificationResult,
)


class ReferenceVerifier(ABC):
    """Interface for verifying whether a cited reference exists.

    Implementations may query CrossRef, OpenAlex, PubMed, etc.
    """

    @abstractmethod
    async def verify(self, reference: ReferenceEntry) -> ReferenceVerificationResult:
        """Verify a single reference entry.

        Args:
            reference: The bibliographic reference to verify.

        Returns:
            Verification result with status and confidence.
        """


class PaperRetriever(ABC):
    """Interface for retrieving full paper content.

    Implementations may use Unpaywall, Semantic Scholar, or
    institutional access to retrieve paper text.
    """

    @abstractmethod
    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Attempt to retrieve the full text of a paper.

        Args:
            reference: The reference to retrieve.

        Returns:
            PaperDocument if retrieval succeeds, None otherwise.
        """
