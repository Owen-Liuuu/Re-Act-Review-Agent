"""Tests for step 2: paper verification."""
from __future__ import annotations

import pytest

from react_review.core.enums import VerificationStatus
from react_review.steps.paper_verification.mock_impl import (
    MockPaperRetriever,
    MockReferenceVerifier,
)
from react_review.steps.paper_verification.schemas import ReferenceEntry


@pytest.mark.asyncio
async def test_mock_reference_verifier(sample_reference: ReferenceEntry):
    """MockReferenceVerifier should return VERIFIED status."""
    verifier = MockReferenceVerifier()
    result = await verifier.verify(sample_reference)

    assert result.status == VerificationStatus.VERIFIED
    assert result.confidence > 0.0
    assert result.reference == sample_reference


@pytest.mark.asyncio
async def test_mock_paper_retriever(sample_reference: ReferenceEntry):
    """MockPaperRetriever should return a PaperDocument."""
    retriever = MockPaperRetriever()
    doc = await retriever.retrieve(sample_reference)

    assert doc is not None
    assert doc.reference == sample_reference
    assert len(doc.full_text) > 0
