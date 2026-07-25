"""Data models for step 3: data extraction and table generation."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from react_review.steps.paper_verification.schemas import ReferenceEntry


class PaperDocument(BaseModel):
    """A retrieved paper with its full text and metadata.

    Attributes:
        paper_id: Unique identifier (DOI or generated).
        reference: The original reference entry.
        full_text: Complete text content of the paper.
        sections: Named sections (e.g. methods, results).
        metadata: Additional metadata from the source.
    """

    paper_id: str
    reference: ReferenceEntry
    full_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)


class ExtractedField(BaseModel):
    """A single field extracted from a paper.

    Attributes:
        field_name: Name of the extracted field.
        value: The extracted value (any JSON-serialisable type).
        evidence: Supporting quote from the paper text for this value.
            For model-extracted fields, this is how the report surfaces
            per-value snippets back to the user.
        confidence: Extraction confidence (0.0 to 1.0).
        extractor_failed: True when the extractor failed to produce a
            value because of an internal error (timeout, parse failure,
            API error), as opposed to legitimately finding no value in
            the paper. Must NOT be blamed on the student.
    """

    field_name: str
    value: str | int | float | bool | list | None = None
    evidence: str = ""
    confidence: float = 0.0
    extractor_failed: bool = False


class ExtractedTable(BaseModel):
    """A complete extraction result for one paper from one extractor.

    Attributes:
        paper_id: Which paper this extraction is from.
        fields: List of extracted fields.
        extractor_id: Identifier of the extractor that produced this.
        extraction_timestamp: When the extraction was performed.
    """

    paper_id: str
    fields: list[ExtractedField] = Field(default_factory=list)
    extractor_id: str = ""
    extraction_timestamp: datetime = Field(default_factory=datetime.now)
