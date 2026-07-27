"""OpenAI-compatible LLM backend.

Works with any OpenAI-compatible `/v1/chat/completions` endpoint:

    - Real OpenAI          (base_url: https://api.openai.com/v1)
    - OpenRouter           (base_url: https://openrouter.ai/api/v1)
    - DeepSeek             (base_url: https://api.deepseek.com/v1)
    - Azure OpenAI (proxy) / local LLM servers, etc.

Uses ``httpx`` directly so no extra SDK dependency is required.
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from react_review.core.config import LLMSettings
from react_review.core.exceptions import LLMError
from react_review.llm.base import LLMBackend

logger = structlog.get_logger(__name__)

_DEFAULT_BASE = "https://api.openai.com/v1"


class OpenAIBackend(LLMBackend):
    """LLM backend for any OpenAI-compatible endpoint.

    Args:
        settings: LLM configuration (api_key, model, temperature,
            base_url, etc.). ``base_url`` is optional and defaults
            to OpenAI's official URL; set it to point at OpenRouter,
            DeepSeek, Azure, local servers, etc.
    """

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(
            max_concurrency=settings.max_concurrency,
            max_retries=settings.max_retries,
            retry_base_delay=settings.retry_base_delay,
        )
        self._settings = settings
        if not settings.api_key:
            raise LLMError("OpenAI backend requires an api_key in config.")
        self._base_url = (settings.base_url or _DEFAULT_BASE).rstrip("/")
        self._model = settings.model or "gpt-4o-mini"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt and return the response text.

        Endpoint: POST ``{base_url}/chat/completions``

        Concurrency is capped by ``self._sem``. HTTP 429 responses are
        retried up to ``self._max_retries`` times with delays computed
        by :meth:`LLMBackend._compute_retry_delay` (Retry-After header
        if present, exponential backoff otherwise). The semaphore slot
        is released **between** attempts so other in-flight tasks can
        proceed during the back-off sleep.
        """
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "seed": seed,
        }
        # Provider-specific extras (e.g. GLM {"thinking": {"type": "disabled"}}).
        if self._settings.extra_body:
            payload.update(self._settings.extra_body)

        logger.info(
            "openai_request",
            model=self._model,
            base_url=self._base_url,
            prompt_len=len(prompt),
        )

        # Unified retry loop. Both HTTP 429 (rate limited) AND
        # ``httpx.RequestError`` (connection reset / read timeout /
        # remote protocol error) are treated as transient and retried
        # with backoff — the latter is common collateral damage while a
        # provider is being rate-limited, and was previously a hard
        # failure that lost the whole extraction.
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0)
            ) as client:
                data = None
                last_failure = "no attempts made"
                for attempt in range(self._max_retries + 1):
                    try:
                        async with self._sem:
                            resp = await client.post(
                                url, headers=headers, json=payload
                            )
                    except httpx.RequestError as exc:
                        last_failure = f"network error: {exc!r}"
                        if attempt >= self._max_retries:
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
                            break
                        delay = self._compute_retry_delay(resp, attempt)
                        self._log_rate_limited(resp, attempt=attempt, delay=delay)
                        await asyncio.sleep(delay)
                        continue

                    # Non-429 response — raise for other 4xx/5xx, else parse.
                    resp.raise_for_status()
                    data = resp.json()
                    break

                if data is None:
                    raise LLMError(
                        f"OpenAI API error after {self._max_retries + 1} "
                        f"attempts: {last_failure}"
                    )

            choices = data.get("choices", [])
            if not choices:
                raise LLMError(f"OpenAI returned no choices: {data}")

            text = choices[0].get("message", {}).get("content", "")
            finish_reason = choices[0].get("finish_reason", "")
            logger.info(
                "openai_response",
                response_len=len(text),
                finish_reason=finish_reason,
            )
            if finish_reason == "length":
                logger.warning(
                    "openai_truncated",
                    msg="Response hit max_tokens — raise max_tokens in config.",
                )
            return text

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMError(
                f"OpenAI API error (HTTP {exc.response.status_code}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            # Reached only if a RequestError escapes the retry loop in an
            # unexpected way; the loop normally converts these to LLMError.
            raise LLMError(f"OpenAI network error: {exc!r}") from exc
