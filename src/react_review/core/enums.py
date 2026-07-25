"""Shared enumerations used across pipeline steps."""
from __future__ import annotations

from enum import Enum


class PipelineStep(str, Enum):
    """Identifiers for each step in the 4-step pipeline."""

    SEARCH_VALIDATION = "search_validation"
    PAPER_VERIFICATION = "paper_verification"
    DATA_EXTRACTION = "data_extraction"
    TABLE_COMPARISON = "table_comparison"


class VerificationStatus(str, Enum):
    """Outcome of a single reference verification check."""

    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    UNCERTAIN = "uncertain"
    ACCESS_RESTRICTED = "access_restricted"


class ValidationSeverity(str, Enum):
    """Severity levels for validation flags."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FieldStatus(str, Enum):
    """Per-field comparison status.

    - MATCH:           student and model values agree after normalization.
    - PARTIAL_MATCH:   values are close (fuzzy text similarity / numeric
                       tolerance within a broader band) but not identical.
    - DIFF:            student and model both have values, but they differ.
    - MISSING_MODEL:   model has no value — treat as extractor gap, NOT
                       a student error.
    - MISSING_STUDENT: student has no value while the model has one.
    - NOT_COMPARABLE:  neither side has a value, or types don't allow a
                       meaningful comparison.
    - NEEDS_REVIEW:    comparison could not be decided automatically;
                       requires a human look.
    """

    MATCH = "match"
    PARTIAL_MATCH = "partial_match"
    DIFF = "diff"
    MISSING_MODEL = "missing_model"
    MISSING_STUDENT = "missing_student"
    NOT_COMPARABLE = "not_comparable"
    NEEDS_REVIEW = "needs_review"


class ComparisonFlagCode(str, Enum):
    """Canonical flag codes used by Step 4.

    FIELD_MISMATCH is reserved for cases where both sides have values
    and they disagree. EXTRACTOR_GAP is used when the model failed to
    extract a field — this must never be blamed on the student.
    """

    FIELD_MISMATCH = "FIELD_MISMATCH"
    EXTRACTOR_GAP = "EXTRACTOR_GAP"
    STUDENT_GAP = "STUDENT_GAP"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NO_STUDENT_TABLE = "NO_STUDENT_TABLE"
    NO_MODEL_TABLE = "NO_MODEL_TABLE"
    NO_TABLES = "NO_TABLES"


class ReportVerdict(str, Enum):
    """Overall verdicts for the final evaluation report."""

    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"
