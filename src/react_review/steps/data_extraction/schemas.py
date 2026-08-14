"""Data models for step 3: data extraction and table generation."""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_serializer

from react_review.steps.paper_verification.schemas import ReferenceEntry


class DocumentScope(str, Enum):
    """How much primary-source content a retriever actually obtained.

    ``UNKNOWN`` is reserved for artifacts written before this field existed.
    New retrievers must declare one of the other three values explicitly; text
    length is deliberately not used to infer it.
    """

    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"
    METADATA_ONLY = "metadata_only"
    UNKNOWN = "unknown"


class PaperDocument(BaseModel):
    """A retrieved paper with its full text and metadata.

    Attributes:
        paper_id: Unique identifier (DOI or generated).
        reference: The original reference entry.
        full_text: Complete text content of the paper.
        sections: Named sections (e.g. methods, results).
        metadata: Additional metadata from the source.
        document_scope: Explicit extent of the retrieved source document.
    """

    paper_id: str
    reference: ReferenceEntry
    full_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, str] = Field(default_factory=dict)
    document_scope: DocumentScope = DocumentScope.UNKNOWN

    @model_serializer(mode="wrap")
    def _omit_legacy_unknown_scope(self, handler):
        """Do not change the bytes of legacy documents with no scope field."""
        body = handler(self)
        if self.document_scope is DocumentScope.UNKNOWN:
            body.pop("document_scope", None)
        return body


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
