"""Small input/output models for tools that don't reuse an existing schema."""
from __future__ import annotations

from pydantic import BaseModel, Field

from react_review.pipeline.schemas import EvidenceFieldSchema
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry

Value = str | int | float | bool | list | None


class CompareInput(BaseModel):
    """Input to the compare_values tool (one review↔source value pair)."""

    field_type: str
    review_value: Value = None
    source_value: Value = None
    review_unit: str = ""
    source_unit: str = ""


class CountInput(BaseModel):
    """Input to a count_results tool: a database query string."""

    query: str


class CountResult(BaseModel):
    """Output of a count_results tool."""

    database: str
    query: str
    count: int | None = None


class FetchResult(BaseModel):
    """Output of the fetch_fulltext tool.

    ``retrieved`` is False when all retrieval tiers failed (``document`` is then
    a metadata-only fallback or None).
    """

    reference: ReferenceEntry
    retrieved: bool = False
    document: PaperDocument | None = None


class ExtractInput(BaseModel):
    """Input to the extract_fields tool."""

    document: PaperDocument
    evidence_schema: list[EvidenceFieldSchema] = Field(default_factory=list)
    research_context: str = ""


class NormalizeInput(BaseModel):
    """Input to the normalize_field tool (Tier-2 semantic mapping)."""

    raw_field_name: str
    unit: str = ""
    modality: str = ""          # optional signal for DKB disambiguation (CT/echo)
    research_context: str = ""


class NormalizeResult(BaseModel):
    """Output of the normalize_field tool.

    ``source`` is where the answer came from: ``cache`` / ``vocabulary`` / ``llm``.
    ``is_new`` is True when the LLM added a new field_type to the vocabulary.
    """

    field_type: str
    source: str = "vocabulary"
    is_new: bool = False

