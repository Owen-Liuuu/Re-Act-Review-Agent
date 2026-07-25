"""Mock implementations for step 3: data extraction.

Both mock extractors honour the rich ``EvidenceFieldSchema`` interface
so the pipeline shape is identical between mock and real modes — the
difference is just where values come from.
"""
from __future__ import annotations

from react_review.pipeline.schemas import EvidenceFieldSchema
from react_review.steps.data_extraction.interfaces import Extractor
from react_review.steps.data_extraction.schemas import (
    ExtractedField,
    ExtractedTable,
    PaperDocument,
)


def _mock_value(prefix: str, schema_entry: EvidenceFieldSchema) -> str | int | float:
    """Generate a deterministic mock value appropriate for the field type.

    Returning typed values (numbers for numeric fields, strings for text
    fields) lets the comparator's type-routing logic exercise the same
    code paths as a real LLM run.
    """
    name = schema_entry.student_field_name
    if schema_entry.type == "numeric":
        # Hash-based stable number derived from the field name so two
        # extractors land on slightly different values without becoming
        # nondeterministic across runs.
        seed = sum(ord(c) for c in name) + (1 if prefix == "a" else 7)
        return float(50 + (seed % 25))
    if schema_entry.type == "year":
        return 2020 + (sum(ord(c) for c in name) % 5)
    return f"value-{prefix}-{name}"


class MockExtractorA(Extractor):
    """Mock extractor A: returns deterministic fake data."""

    @property
    def extractor_id(self) -> str:
        return "mock-extractor-a"

    async def extract(
        self,
        document: PaperDocument,
        schema: list[EvidenceFieldSchema],
        *,
        research_context: str = "",
    ) -> ExtractedTable:
        extracted_fields = [
            ExtractedField(
                field_name=s.student_field_name,
                value=_mock_value("a", s),
                evidence=(
                    f"Mock evidence A for {s.student_field_name} "
                    f"(concept={s.canonical_concept}) from {document.paper_id}."
                ),
                confidence=0.90,
            )
            for s in schema
        ]
        return ExtractedTable(
            paper_id=document.paper_id,
            fields=extracted_fields,
            extractor_id=self.extractor_id,
        )


class MockExtractorB(Extractor):
    """Mock extractor B: slightly different fake data for cross-validation."""

    @property
    def extractor_id(self) -> str:
        return "mock-extractor-b"

    async def extract(
        self,
        document: PaperDocument,
        schema: list[EvidenceFieldSchema],
        *,
        research_context: str = "",
    ) -> ExtractedTable:
        extracted_fields = [
            ExtractedField(
                field_name=s.student_field_name,
                value=_mock_value("b", s),
                evidence=(
                    f"Mock evidence B for {s.student_field_name} "
                    f"(concept={s.canonical_concept}) from {document.paper_id}."
                ),
                confidence=0.85,
            )
            for s in schema
        ]
        return ExtractedTable(
            paper_id=document.paper_id,
            fields=extracted_fields,
            extractor_id=self.extractor_id,
        )
