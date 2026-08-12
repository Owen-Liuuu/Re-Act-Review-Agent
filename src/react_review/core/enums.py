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


class ReflectionDecision(str, Enum):
    """What to do after a stage (Proposal E6).

    - ACCEPT:   the result is trustworthy; move on.
    - RETRY:    re-attempt (e.g. a different backend / a different source repo),
                while attempts remain.
    - ESCALATE: hand to a human review flag — too uncertain, or retries spent.
    """

    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"


class AuditLabel(str, Enum):
    """Per-value audit verdict (review value vs source value).

    This is the label set the benchmark's ``expected_label`` uses.

    - MATCH:          primary values agree within the field's tolerance.
    - MISMATCH:       primary values differ beyond tolerance.
    - UNIT_MISMATCH:  the reported units differ (independent of value closeness).
    - NOT_COMPARABLE: one/both sides have no parseable value.
    """

    MATCH = "match"
    MISMATCH = "mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    NOT_COMPARABLE = "not_comparable"


class CollectionOutcome(str, Enum):
    """How the Collector's source lookup ended — drives the human-review label.

    Splits the old single "no source value" case into two very different
    signals:

    - FOUND:                the source value was located in the paper.
    - SOURCE_ACCESS_FAILED: the source full text could not be retrieved at all
                            (an access/coverage gap — NOT the review's fault).
    - MISSING_SOURCE:       the paper WAS retrieved but the value is not stated
                            in it — a potential fabrication in the review.
    - UNRESOLVED_SOURCE:    the source paper could not even be identified from its
                            citation (no DOI printed and no confident online match).
    - UNKNOWN_COHORT:       the review's claim could not be tied to a cohort, so
                            there was nothing specific to look for. Kept separate
                            from MISSING_SOURCE, which reads as possible
                            fabrication and would be the wrong accusation here.
    - EXTRACTION_FAILED:    the paper was retrieved and the READING failed — the
                            response never arrived, or arrived malformed, or
                            every reading in it failed a deterministic check.
                            This is a fact about the response. Reporting it as
                            MISSING_SOURCE would put an accusation on the review
                            for a fault in the extractor.
    - EXTRACTION_UNRESOLVED: the reading was fine and the ANSWER is not unique —
                            the arm, the population or the timepoint could not
                            be pinned to exactly one candidate, or the paper
                            contradicts itself. Nothing was hidden and nothing
                            may be released.
    """

    FOUND = "found"
    SOURCE_ACCESS_FAILED = "source_access_failed"
    MISSING_SOURCE = "missing_source"
    UNRESOLVED_SOURCE = "unresolved_source"
    UNKNOWN_COHORT = "unknown_cohort"
    EXTRACTION_FAILED = "extraction_failed"
    EXTRACTION_UNRESOLVED = "extraction_unresolved"
