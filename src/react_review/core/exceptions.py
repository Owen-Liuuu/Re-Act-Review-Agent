"""Exception hierarchy for react_review.

All custom exceptions inherit from LitInspectorError so callers can
catch a single base class when they want to handle any project error.
"""
from __future__ import annotations


class LitInspectorError(Exception):
    """Base exception for all react_review errors."""


class ConfigError(LitInspectorError):
    """Raised when configuration loading or validation fails."""


class LLMError(LitInspectorError):
    """Raised when an LLM call or response parsing fails."""


class SearchValidationError(LitInspectorError):
    """Raised during step 1: search strategy validation."""


class VerificationError(LitInspectorError):
    """Raised during step 2: paper existence verification."""


class ExtractionError(LitInspectorError):
    """Raised during step 3: data extraction."""


class ComparisonError(LitInspectorError):
    """Raised during step 4: table comparison."""


class RunStopped(LitInspectorError):
    """A human stopped the run at a checkpoint.

    Not a failure: the reviewer looked at a step and decided the run should not
    continue (e.g. the review's main table was not extracted correctly). Carries
    where it happened so the caller can persist a partial package and say why.
    """

    def __init__(self, *, stage: str = "", index: int = 0, reason: str = "") -> None:
        self.stage = stage
        self.index = index
        self.reason = reason or f"run stopped at {stage or 'a checkpoint'}"
        super().__init__(self.reason)
