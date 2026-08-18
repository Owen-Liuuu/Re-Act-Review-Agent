"""Tests for Pydantic data models."""
from __future__ import annotations

from react_review.steps.data_extraction.schemas import ExtractedField, ExtractedTable
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.steps.search_validation.schemas import SearchStrategy


def test_reference_entry_minimal():
    """ReferenceEntry should work with just a title."""
    ref = ReferenceEntry(title="Test Paper")
    assert ref.title == "Test Paper"
    assert ref.authors == []
    assert ref.doi is None


def test_reference_entry_full():
    """ReferenceEntry should accept all fields."""
    ref = ReferenceEntry(
        title="Test Paper",
        authors=["Author A", "Author B"],
        journal="Test Journal",
        year=2023,
        doi="10.1234/test",
        pmid="12345678",
    )
    assert len(ref.authors) == 2
    assert ref.year == 2023


def test_search_strategy():
    """SearchStrategy should store query terms and filters."""
    strategy = SearchStrategy(
        database="PubMed",
        query_terms=["cancer", "immunotherapy"],
        filters={"language": "English"},
        raw_strategy_text="Search PubMed for cancer AND immunotherapy",
    )
    assert len(strategy.query_terms) == 2
    assert strategy.database == "PubMed"


def test_extracted_field():
    """ExtractedField should store value and evidence."""
    field = ExtractedField(
        field_name="sample_size",
        value=200,
        evidence="200 patients were enrolled",
        confidence=0.95,
    )
    assert field.value == 200
    assert field.confidence == 0.95


def test_extracted_table():
    """ExtractedTable should contain multiple fields."""
    table = ExtractedTable(
        paper_id="test-paper",
        fields=[
            ExtractedField(field_name="sample_size", value=100),
            ExtractedField(field_name="study_design", value="RCT"),
        ],
        extractor_id="test-extractor",
    )
    assert len(table.fields) == 2
    assert table.extractor_id == "test-extractor"


def test_model_serialization():
    """Pydantic models should round-trip through dict."""
    ref = ReferenceEntry(title="Test", authors=["A"], year=2023)
    data = ref.model_dump()
    ref2 = ReferenceEntry(**data)
    assert ref == ref2
