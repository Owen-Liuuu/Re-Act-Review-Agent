"""Data models for step 1: search strategy validation."""
from __future__ import annotations

from pydantic import BaseModel, Field

from react_review.core.enums import ValidationSeverity


class SearchStrategy(BaseModel):
    """A student's reported search strategy to be validated.

    Attributes:
        database: Target database (e.g. PubMed, Cochrane, Embase).
        query_terms: Key search terms used.
        filters: Applied filters (e.g. date range, language).
        date_range: Reported date range for the search.
        raw_strategy_text: The original free-text description from
                           the student's systematic review.
        reported_result_count: How many results the student claims to have found.
    """

    database: str
    query_terms: list[str] = Field(default_factory=list)
    filters: dict[str, str] = Field(default_factory=dict)
    date_range: str | None = None
    raw_strategy_text: str = ""
    reported_result_count: int | None = None


class SearchValidationFlag(BaseModel):
    """A single flag raised during search validation."""

    code: str
    severity: ValidationSeverity
    message: str


class PubMedSummary(BaseModel):
    """Summary-level metadata for a paper found in PubMed search results.

    Used to check whether the student's selected papers appear in the
    reproduced search results.
    """

    pmid: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    source: str = ""
    pubdate: str = ""
    doi: str = ""


class SearchValidationResult(BaseModel):
    """Output of step 1: search strategy validation.

    Attributes:
        original_strategy: The student's original strategy.
        reconstructed_query: The query rebuilt by the system.
        reported_count: Number of results the student reported.
        actual_count: Number of results the system found via PubMed.
        is_reproducible: Whether the search could be reproduced.
        sample_results: Summary of the first few PubMed results
            (used to cross-check selected papers).
        all_pmids: Full list of PMIDs returned by the search
            (for cross-checking selected papers).
        flags: Validation issues found.
        notes: Additional human-readable notes.
    """

    original_strategy: SearchStrategy
    reconstructed_query: str = ""
    reported_count: int | None = None
    actual_count: int | None = None
    is_reproducible: bool = False
    sample_results: list[PubMedSummary] = Field(default_factory=list)
    all_pmids: list[str] = Field(default_factory=list)
    flags: list[SearchValidationFlag] = Field(default_factory=list)
    notes: str = ""
