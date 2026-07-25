"""LLM backend abstract base class and response parsing utilities.

Pattern adapted from metascreener.llm.base (小组项目/SRC/).
"""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import structlog

from react_review.core.exceptions import LLMError

if TYPE_CHECKING:
    import httpx

logger = structlog.get_logger(__name__)


class LLMBackend(ABC):
    """Abstract interface for any LLM provider.

    Subclasses (OpenAI / Anthropic / Gemini / Qwen / Mock / …) MUST
    invoke ``super().__init__(...)`` so the per-backend semaphore and
    retry settings are initialised. Subclasses' ``complete()``
    implementations SHOULD:

      * wrap each HTTP attempt in ``async with self._sem:`` so the
        semaphore enforces the configured concurrency cap;
      * use ``self._compute_retry_delay(response, attempt)`` together
        with a retry loop bounded by ``self._max_retries`` to back off
        on HTTP 429 (rate limited) responses, respecting the provider's
        ``Retry-After`` header when present.

    Per project decisions (2026-05-10):
      * The semaphore caps in-flight requests at ``max_concurrency``
        — addresses provider concurrency caps (e.g. Moonshot's "max 3
        in-flight").
      * The retry loop covers RPM-style limits (e.g. Zhipu free tier's
        per-minute limit) which the semaphore cannot prevent.
    """

    def __init__(
        self,
        max_concurrency: int = 3,
        *,
        max_retries: int = 5,
        retry_base_delay: float = 2.0,
    ) -> None:
        # asyncio.Semaphore is created lazily on first ``async with``
        # so it binds to whatever event loop the caller is using.
        # Capacity 1 minimum — anything lower would deadlock.
        self._sem: asyncio.Semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._max_concurrency: int = max(1, max_concurrency)
        self._max_retries: int = max(0, max_retries)
        self._retry_base_delay: float = max(0.0, retry_base_delay)

    @property
    def max_concurrency(self) -> int:
        """The configured concurrency cap for this backend instance."""
        return self._max_concurrency

    @property
    def max_retries(self) -> int:
        """Maximum 429 retry attempts (0 disables retry)."""
        return self._max_retries

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the identifier of the underlying model."""

    @abstractmethod
    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt and return the raw text response.

        Implementations should wrap their actual HTTP call in
        ``async with self._sem:`` and use a retry loop with
        ``self._compute_retry_delay`` to handle HTTP 429 responses.
        """

    # ------------------------------------------------------------------
    # Shared retry helper
    # ------------------------------------------------------------------

    def _compute_retry_delay(
        self, response: "httpx.Response | None", attempt: int
    ) -> float:
        """Compute the delay (seconds) before the next retry attempt.

        Resolution order:

          1. ``Retry-After`` response header (RFC 7231) — only the
             integer-seconds form is parsed. Only applies when a
             ``response`` is available (i.e. an HTTP 429); for network
             errors ``response`` is ``None`` and we skip straight to (2).
          2. Exponential backoff: ``retry_base_delay * 2 ** attempt``
             (so attempt=0 → base, attempt=1 → 2*base, …).

        Args:
            response: The HTTP response that returned 429, or ``None``
                when the failure was a network error (no response).
            attempt: Zero-indexed attempt number that *just failed*
                (so the first failure is ``attempt=0`` and the next
                wait corresponds to that index).

        Returns:
            Non-negative float — the number of seconds to sleep.
        """
        if response is not None:
            retry_after_raw = (
                response.headers.get("Retry-After", "") or ""
            ).strip()
            if retry_after_raw:
                try:
                    retry_after = float(retry_after_raw)
                    if retry_after > 0:
                        return retry_after
                except ValueError:
                    # Non-numeric (HTTP-date) — fall back to exponential.
                    pass
        # Exponential backoff. Attempt 0 → base, attempt 1 → 2*base, …
        return self._retry_base_delay * (2 ** max(0, attempt))

    def _log_rate_limited(
        self,
        response: "httpx.Response",
        *,
        attempt: int,
        delay: float,
    ) -> None:
        """Emit a structured warning when we back off on HTTP 429."""
        used_header = bool(response.headers.get("Retry-After"))
        logger.warning(
            "llm_rate_limited_retry",
            model=self.model_id,
            attempt=attempt + 1,
            max_retries=self._max_retries,
            delay_s=round(delay, 2),
            source="Retry-After" if used_header else "exponential_backoff",
            status=response.status_code,
        )

    def _log_transient_retry(
        self,
        *,
        attempt: int,
        delay: float,
        detail: str = "",
    ) -> None:
        """Emit a structured warning when we back off on a network error.

        Network errors (connection reset, read timeout, remote protocol
        error) are transient — typically a side-effect of the provider
        being hammered while rate-limited. Treating them like a 429 and
        retrying with backoff recovers extractions that would otherwise
        be lost as a hard failure.
        """
        logger.warning(
            "llm_network_error_retry",
            model=self.model_id,
            attempt=attempt + 1,
            max_retries=self._max_retries,
            delay_s=round(delay, 2),
            detail=(detail or "")[:160],
        )


def parse_llm_response(raw: str, model_id: str) -> dict:
    """Extract a JSON object from a raw LLM response.

    Handles common cases where the model wraps JSON in markdown
    code fences or adds extra text around it.

    Args:
        raw: Raw text response from the LLM.
        model_id: Model identifier (for error messages).

    Returns:
        Parsed dictionary.

    Raises:
        LLMError: If no valid JSON can be extracted.
    """
    # Strip <think>...</think> reasoning blocks (deepseek-reasoner, etc.)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMError(
        f"Failed to parse JSON from {model_id} response. "
        f"Raw output (first 200 chars): {raw[:200]}"
    )
