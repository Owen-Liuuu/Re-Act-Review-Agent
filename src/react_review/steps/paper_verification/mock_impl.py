"""Mock implementations for step 2: paper existence verification."""
from __future__ import annotations

from react_review.core.enums import VerificationStatus
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.interfaces import (
    PaperRetriever,
    ReferenceVerifier,
)
from react_review.steps.paper_verification.schemas import (
    ReferenceEntry,
    ReferenceVerificationResult,
)


class MockReferenceVerifier(ReferenceVerifier):
    """Returns fake verification results for testing."""

    async def verify(self, reference: ReferenceEntry) -> ReferenceVerificationResult:
        return ReferenceVerificationResult(
            reference=reference,
            status=VerificationStatus.VERIFIED,
            matched_metadata={
                "title": reference.title,
                "source": "mock-database",
            },
            confidence=0.95,
            access_note="",
            flags=[],
        )


class MockPaperRetriever(PaperRetriever):
    """Returns fake paper documents for testing."""

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        return PaperDocument(
            paper_id=reference.doi or f"mock-{reference.title[:20]}",
            reference=reference,
            full_text=(
                f"This is a mock full text for the paper: {reference.title}. "
                "It contains sample content for testing the extraction pipeline. "
                "Methods: A systematic review was conducted. "
                "Results: The treatment showed significant improvement (p<0.05). "
                "Sample size: 200 patients were enrolled."
            ),
            document_scope=DocumentScope.FULL_TEXT,
            sections={"methods": "A systematic review was conducted.", "results": "Significant improvement found."},
            metadata={"source": "mock-retriever"},
        )
