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
    # Context for the semantic fallback. Two text values can only be judged
    # equivalent against what they are values OF and where the source one came
    # from — "ICU" and "intensive care unit" under a Setting column, quoted from
    # the paper, is a different question from the strings in isolation.
    column_header: str = ""
    source_quote: str = ""
    research_context: str = ""
    # The verified parts of the source value. Passed alongside the verbatim
    # string rather than instead of it: the audit still displays what the paper
    # printed, and compares what the extraction actually read.
    source_components: dict | None = None
    # Which population each side counts. Compared before the arithmetic,
    # because it decides whether the arithmetic is about the same people.
    review_scope: dict | None = None
    source_scope: dict | None = None


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

