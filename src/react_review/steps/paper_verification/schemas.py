"""Data models for step 2: paper existence verification."""
from __future__ import annotations

from pydantic import BaseModel, Field

from react_review.core.enums import ValidationSeverity, VerificationStatus


class ReferenceEntry(BaseModel):
    """A single bibliographic reference from the student's review.

    Attributes:
        title: Paper title as cited.
        authors: List of author names.
        journal: Journal name.
        year: Publication year.
        doi: Digital Object Identifier (if available).
        pmid: PubMed ID (if available).
    """

    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int | None = None
    doi: str | None = None
    pmid: str | None = None


class VerificationFlag(BaseModel):
    """A single flag raised during reference verification."""

    code: str
    severity: ValidationSeverity
    message: str


class ReferenceVerificationResult(BaseModel):
    """Verification outcome for a single reference.

    Attributes:
        reference: The original reference being verified.
        status: Verification outcome.
        matched_metadata: Metadata returned from the lookup source.
        confidence: Confidence score (0.0 to 1.0).
        access_note: Explanation if access is restricted.
        flags: Issues found during verification.
    """

    reference: ReferenceEntry
    status: VerificationStatus = VerificationStatus.UNCERTAIN
    matched_metadata: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0
    access_note: str = ""
    flags: list[VerificationFlag] = Field(default_factory=list)
