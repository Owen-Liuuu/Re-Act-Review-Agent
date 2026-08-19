"""Small input/output models for tools that don't reuse an existing schema."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from react_review.schemas.adequacy import AdequacyStatus, EvidenceAdequacy
from react_review.schemas.table import CapturedTable
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry

Value = str | int | float | bool | list | None


class CompareInput(BaseModel):
    """Input to the compare_values tool (one review↔source value pair)."""

    audit_id: str = ""
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
    # Defence in depth: the orchestrator owns the adequacy gate, but the tool
    # input also refuses an explicitly non-sufficient claim. That makes it
    # impossible for another caller to bypass the gate accidentally.
    evidence_adequacy: EvidenceAdequacy | None = None

    @model_validator(mode="after")
    def _only_sufficient_evidence_may_be_compared(self):
        if (self.evidence_adequacy is not None
                and self.evidence_adequacy.status is not AdequacyStatus.SUFFICIENT):
            raise ValueError(
                f"{self.evidence_adequacy.status.value} evidence must not enter "
                "value comparison")
        return self


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

    ``tables`` carries structured PMC ``<table-wrap>`` captures when the
    retriever produced them. They sit here rather than on ``PaperDocument`` so
    the evidence-adequacy evaluator hash (which includes that schema) does not
    have to be re-frozen for a source-side table object.
    """

    reference: ReferenceEntry
    retrieved: bool = False
    document: PaperDocument | None = None
    tables: list[CapturedTable] = Field(default_factory=list)


class ForestOcrInput(BaseModel):
    """One forest-plot figure. Tests may inject a grid instead of reading a PDF."""

    pdf_path: str = ""
    figure_id: str = ""
    caption: str = ""
    page_hint: str = ""
    outcomes: list[str] = []
    # 0-based forest-plot hit among the review's forest displays.
    figure_ordinal: int = 0
    # Deprecated page-local index; used only if caption pairing cannot run.
    image_index: int = 0
    # When set, the tool returns this table and does not invent cells.
    injected_table: dict | None = None


class ForestOcrResult(BaseModel):
    """A forest plot as a CapturedTable, or an empty grid plus difficulties."""

    table: CapturedTable
    difficulties: list[str] = []

