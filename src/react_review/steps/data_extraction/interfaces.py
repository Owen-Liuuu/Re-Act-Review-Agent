"""Abstract interface for step 3: data extraction.

Step 3 takes the rich ``evidence_schema`` produced by Step 0 and asks the
LLM (or a mock backend) to find each field's value in a source paper. The
schema carries the canonical concept name and value type, which the
extractor uses to build a more informative prompt; the result table keys
back to the student's verbatim field names so Step 4 can join cleanly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from react_review.steps.data_extraction.schemas import ExtractedTable, PaperDocument

if TYPE_CHECKING:
    from react_review.pipeline.schemas import EvidenceFieldSchema


class Extractor(ABC):
    """Interface for extracting structured data from a paper.

    Multiple extractor implementations (e.g. using different LLMs)
    can be run in parallel for cross-validation.
    """

    @property
    @abstractmethod
    def extractor_id(self) -> str:
        """Unique identifier for this extractor."""

    @abstractmethod
    async def extract(
        self,
        document: PaperDocument,
        schema: "list[EvidenceFieldSchema]",
        *,
        research_context: str = "",
    ) -> ExtractedTable:
        """Extract specified fields from a paper document.

        Args:
            document: The paper to extract from.
            schema: One entry per field the student extracted, carrying
                ``student_field_name`` (used as the output key),
                ``canonical_concept`` (the abstract target the extractor
                should look for in the paper), ``type`` (numeric / text /
                categorical / author / year / doi), and an optional
                ``description``. The extractor MUST return its values
                keyed by ``student_field_name`` so Step 4 can join.
            research_context: One-sentence description of the review's
                topic, e.g. "EAT thickness/volume in T1DM vs controls".
                Helps the LLM understand what data to look for.

        Returns:
            Extraction results as a structured table whose
            ``ExtractedField.field_name`` matches each schema entry's
            ``student_field_name``.
        """
