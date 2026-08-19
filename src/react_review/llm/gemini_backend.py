"""Google Gemini LLM backend — connects via REST API (no SDK needed).

Uses the generativelanguage.googleapis.com endpoint directly with httpx.
Docs: https://ai.google.dev/gemini-api/docs/text-generation
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from react_review.core.config import LLMSettings
from react_review.core.exceptions import LLMError
from react_review.llm.base import LLMBackend

logger = structlog.get_logger(__name__)

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiBackend(LLMBackend):
    """LLM backend using the Google Gemini REST API.

    Args:
        settings: LLM configuration (api_key, model, temperature, etc.).
    """

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(
            max_concurrency=settings.max_concurrency,
            max_retries=settings.max_retries,
            retry_base_delay=settings.retry_base_delay,
        )
        self._settings = settings
        if not settings.api_key:
            raise LLMError("Gemini backend requires an api_key in config.")
        self._model = settings.model or "gemini-2.5-flash"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt to Gemini and return the response text.

        Endpoint: POST /v1beta/models/{model}:generateContent?key={key}

        Throttled by ``self._sem`` (see ``LLMBackend``).
        """
        url = f"{_GEMINI_BASE}/models/{self._model}:generateContent"
        params = {"key": self._settings.api_key}

        generation_config: dict = {
            "temperature": self._settings.temperature,
            "maxOutputTokens": self._settings.max_tokens,
        }
        # Resolve thinking budget:
        #   - explicit value in config → always respect it (incl. 0 = off, -1 = dynamic)
        #   - not set (None) → for 2.5-series, default to 0 to avoid silent
        #     truncation; other models don't support thinkingConfig so skip.
        budget = self._settings.thinking_budget
        if budget is None and "2.5" in self._model:
            budget = 0
        if budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": budget}

        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": generation_config,
        }

        logger.info(
            "gemini_request",
            model=self._model,
            prompt_len=len(prompt),
        )

        # Unified retry loop — retries both HTTP 429 and transient
        # network errors with backoff (see OpenAIBackend for rationale).
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
                data = None
                last_failure = "no attempts made"
                for attempt in range(self._max_retries + 1):
                    try:
                        async with self._sem:
                            resp = await client.post(
                                url, params=params, json=payload
                            )
                    except httpx.RequestError as exc:
                        last_failure = f"network error: {exc!r}"
                        if attempt >= self._max_retries:
                            self._log_transient_retry(
                                attempt=attempt, delay=0, detail=repr(exc),
                                exhausted=True)
                            break
                        delay = self._compute_retry_delay(None, attempt)
                        self._log_transient_retry(
                            attempt=attempt, delay=delay, detail=repr(exc)
                        )
                        await asyncio.sleep(delay)
                        continue

                    if resp.status_code == 429:
                        last_failure = f"HTTP 429: {resp.text[:300]}"
                        if attempt >= self._max_retries:
                            self._log_rate_limited(
                                resp, attempt=attempt, delay=0, exhausted=True)
                            break
                        delay = self._compute_retry_delay(resp, attempt)
                        self._log_rate_limited(resp, attempt=attempt, delay=delay)
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    break

                if data is None:
                    raise LLMError(
                        f"Gemini API error after {self._max_retries + 1} "
                        f"attempts: {last_failure}"
                    )

            # Extract text from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMError(f"Gemini returned no candidates: {data}")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise LLMError(f"Gemini candidate has no parts: {candidates[0]}")

            text = parts[0].get("text", "")
            finish_reason = candidates[0].get("finishReason", "")
            logger.info(
                "gemini_response",
                response_len=len(text),
                finish_reason=finish_reason,
            )
            if finish_reason == "MAX_TOKENS":
                logger.warning(
                    "gemini_truncated",
                    msg="Response hit maxOutputTokens — raise max_tokens in config.",
                )
            return text

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMError(
                f"Gemini API error (HTTP {exc.response.status_code}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Gemini network error: {exc!r}") from exc
