"""Shared test fixtures for react_review tests."""
from __future__ import annotations

import pytest

from react_review.core.config import AppConfig
from react_review.pipeline.schemas import (
    DatabaseQuery,
    EvidenceFieldSchema,
    SearchStrategy,
    StudentReviewInput,
)
from react_review.steps.data_extraction.schemas import ExtractedField, ExtractedTable
from react_review.steps.paper_verification.schemas import ReferenceEntry


@pytest.fixture
def sample_config() -> AppConfig:
    """Provide a default test configuration."""
    return AppConfig()


@pytest.fixture
def sample_reference() -> ReferenceEntry:
    """Provide a sample reference entry."""
    return ReferenceEntry(
        title="Effectiveness of immunotherapy in advanced lung cancer",
        authors=["Zhang Y", "Li X"],
        journal="Journal of Clinical Oncology",
        year=2023,
        doi="10.1234/test-doi-001",
    )


@pytest.fixture
def sample_references(sample_reference: ReferenceEntry) -> list[ReferenceEntry]:
    """Provide a list of sample references."""
    return [
        sample_reference,
        ReferenceEntry(
            title="Chemotherapy outcomes in breast cancer",
            authors=["Smith A"],
            journal="The Lancet Oncology",
            year=2022,
            doi="10.1234/test-doi-002",
        ),
    ]


@pytest.fixture
def sample_student_input(
    sample_references: list[ReferenceEntry],
) -> StudentReviewInput:
    """Provide a sample review input built with the current rich schema.

    Mirrors the shape produced by Step 0 (Ingestion): a structured
    ``SearchStrategy`` with a PubMed query, a rich ``evidence_schema``,
    and one submitted table. This is what downstream steps consume.
    """
    return StudentReviewInput(
        student_id="test-student-001",
        review_title="Test Systematic Review",
        research_context="Immunotherapy vs chemotherapy in advanced cancer.",
        search_strategy=SearchStrategy(
            raw_text="immunotherapy AND cancer",
            extracted_per_database=[
                DatabaseQuery(
                    database="PubMed",
                    query="immunotherapy AND cancer",
                    source="verbatim",
                ),
            ],
            reported_total_count=150,
            reported_per_db_count={"PubMed": 150},
        ),
        selected_papers=sample_references,
        evidence_schema=[
            EvidenceFieldSchema(
                student_field_name="sample_size",
                canonical_concept="sample_size",
                type="numeric",
                threshold_match=0.0,
                threshold_partial=0.05,
            ),
            EvidenceFieldSchema(
                student_field_name="study_design",
                canonical_concept="study_design",
                type="categorical",
                threshold_match=0.95,
                threshold_partial=0.75,
            ),
        ],
        submitted_tables=[
            ExtractedTable(
                paper_id="10.1234/test-doi-001",
                fields=[
                    ExtractedField(field_name="sample_size", value=100, confidence=1.0),
                    ExtractedField(field_name="study_design", value="RCT", confidence=1.0),
                ],
                extractor_id="student",
            ),
        ],
    )
