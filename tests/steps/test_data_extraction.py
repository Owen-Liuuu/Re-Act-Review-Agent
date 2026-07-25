"""Tests for step 3: data extraction."""
from __future__ import annotations

import pytest

from react_review.pipeline.schemas import EvidenceFieldSchema
from react_review.steps.data_extraction.mock_impl import MockExtractorA, MockExtractorB
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry


@pytest.fixture
def sample_document() -> PaperDocument:
    """Provide a sample paper document."""
    return PaperDocument(
        paper_id="test-paper-001",
        reference=ReferenceEntry(title="Test Paper"),
        full_text="Sample paper text for testing extraction.",
    )


def _schema(*names: str) -> list[EvidenceFieldSchema]:
    """Build a minimal evidence schema for the given field names."""
    return [
        EvidenceFieldSchema(
            student_field_name=n,
            canonical_concept=n,
            type="numeric" if n == "sample_size" else "categorical",
        )
        for n in names
    ]


@pytest.mark.asyncio
async def test_mock_extractor_a(sample_document: PaperDocument):
    """MockExtractorA should return an ExtractedTable keyed by field name."""
    extractor = MockExtractorA()
    schema = _schema("sample_size", "study_design")

    result = await extractor.extract(sample_document, schema)

    assert result.paper_id == "test-paper-001"
    assert result.extractor_id == "mock-extractor-a"
    assert len(result.fields) == 2
    assert result.fields[0].field_name == "sample_size"


@pytest.mark.asyncio
async def test_extractors_produce_different_ids(sample_document: PaperDocument):
    """Different extractors should have different extractor_ids."""
    a = MockExtractorA()
    b = MockExtractorB()
    schema = _schema("sample_size")

    result_a = await a.extract(sample_document, schema)
    result_b = await b.extract(sample_document, schema)

    assert result_a.extractor_id != result_b.extractor_id
