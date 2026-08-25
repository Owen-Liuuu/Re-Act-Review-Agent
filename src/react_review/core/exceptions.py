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


#: Auth / billing refusals. Retrying cannot succeed; the HTTP layer already
#: does not retry them. The extraction loop must not either.
PERMANENT_HTTP = frozenset({401, 402, 403})


def http_status_from_error(exc: BaseException) -> int | None:
    """Parse ``HTTP 4xx`` out of an ``LLMError`` (or its message)."""
    import re
    match = re.search(r"HTTP (\d{3})", str(exc))
    if match is None:
        return None
    return int(match.group(1))


class PermanentProviderError(LLMError):
    """The provider refused the request; another attempt cannot succeed.

    Distinct from a paper that omits a value, and from a transient 429/500.
    The run must stop rather than walk every remaining claim into the same wall.
    """

    def __init__(
        self,
        message: str = "",
        *,
        status: int = 0,
        not_found_reason: str = "",
    ) -> None:
        self.status = status
        self.not_found_reason = not_found_reason or message
        super().__init__(message)


def raise_if_permanent(exc: BaseException) -> None:
    """Re-raise ``exc`` as :class:`PermanentProviderError` when it is 401/402/403."""
    if isinstance(exc, PermanentProviderError):
        raise exc
    status = http_status_from_error(exc)
    if status not in PERMANENT_HTTP:
        return
    reason = f"the extraction call failed: {exc}"[:300]
    raise PermanentProviderError(
        "the provider refused the request "
        f"(HTTP {status}); the run stopped rather than repeating a call "
        "that cannot succeed. "
        f"Original error: {exc}"[:400],
        status=status,
        not_found_reason=reason,
    ) from exc


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


class ModelUnavailable(LitInspectorError):
    """Every model call a stage made failed, so the stage produced nothing.

    Not the same as a review with no extractable table: both leave zero items,
    and only this one means the run never received an answer to judge. Carries
    the counts so the artifact can say what was observed rather than only that
    the run gave up.
    """

    def __init__(self, *, stage: str, requests: int) -> None:
        self.stage = stage
        self.requests = requests
        super().__init__(
            f"all {requests} model call(s) during {stage} failed, so there is "
            f"no parsed review to audit. This is a provider or transport "
            f"failure, not a review whose table could not be found")
