"""Qwen (通义千问) LLM backend — connects via DashScope OpenAI-compatible API.

Uses the OpenAI-compatible endpoint provided by Alibaba Cloud DashScope.
Docs: https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from react_review.core.config import LLMSettings
from react_review.core.exceptions import LLMError
from react_review.llm.base import LLMBackend

logger = structlog.get_logger(__name__)

_DASHSCOPE_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


class QwenBackend(LLMBackend):
    """LLM backend using the Qwen / DashScope OpenAI-compatible API.

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
            raise LLMError("Qwen backend requires an api_key in config.")
        configured_base = (settings.base_url or "").rstrip("/")
        if configured_base and configured_base != _DASHSCOPE_BASE:
            logger.warning(
                "qwen_base_url_overridden",
                configured_base=configured_base,
                forced_base_url=_DASHSCOPE_BASE,
            )
        self._base_url = _DASHSCOPE_BASE
        self._model = settings.model or "qwen-plus"

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        """Send a prompt to Qwen via the OpenAI-compatible endpoint.

        Endpoint: POST /v1/chat/completions

        Throttled by ``self._sem`` (see ``LLMBackend``).
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

        logger.info(
            "qwen_request",
            model=self._model,
            base_url=self._base_url,
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

                    resp.raise_for_status()
                    data = resp.json()
                    break

                if data is None:
                    raise LLMError(
                        f"Qwen API error after {self._max_retries + 1} "
                        f"attempts: {last_failure}"
                    )

            # OpenAI-compatible response format
            choices = data.get("choices", [])
            if not choices:
                raise LLMError(f"Qwen returned no choices: {data}")

            text = choices[0].get("message", {}).get("content", "")
            logger.info("qwen_response", response_len=len(text))
            return text

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMError(
                f"Qwen API error (HTTP {exc.response.status_code}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Qwen network error: {exc!r}") from exc
