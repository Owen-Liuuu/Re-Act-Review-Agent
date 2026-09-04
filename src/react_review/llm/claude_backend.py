"""Anthropic Claude LLM backend via the Messages HTTP API (no SDK)."""
from __future__ import annotations

import asyncio
import base64

import httpx
import structlog

from react_review.core.config import LLMSettings
from react_review.core.exceptions import LLMError
from react_review.llm.base import LLMBackend

logger = structlog.get_logger(__name__)

_DEFAULT_BASE = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


def _image_block(blob: bytes) -> dict:
    if blob.startswith(b"\x89PNG"):
        mime = "image/png"
    elif blob.startswith(b"\xff\xd8"):
        mime = "image/jpeg"
    elif blob[:6] in (b"GIF87a", b"GIF89a"):
        mime = "image/gif"
    elif blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = "image/png"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": base64.standard_b64encode(blob).decode("ascii"),
        },
    }


def _messages_url(base_url: str) -> str:
    base = (base_url or _DEFAULT_BASE).rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


class ClaudeBackend(LLMBackend):
    """LLM backend using Anthropic's Messages API."""

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(
            max_concurrency=settings.max_concurrency,
            max_retries=settings.max_retries,
            retry_base_delay=settings.retry_base_delay,
        )
        self._settings = settings
        if not settings.api_key:
            raise LLMError("Anthropic backend requires an api_key in config.")
        self._url = _messages_url(settings.base_url)
        self._model = settings.model or "claude-sonnet-5"
        self.read_timeout_seconds = settings.read_timeout_seconds

    @property
    def model_id(self) -> str:
        return self._model

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        logger.info(
            "anthropic_request",
            model=self._model,
            prompt_len=len(prompt),
        )
        return await self._post_messages(
            [{"type": "text", "text": prompt}]
        )

    async def complete_vision(
        self, prompt: str, images: list[bytes], *, seed: int = 42,
    ) -> str:
        content = [_image_block(blob) for blob in images]
        content.append({"type": "text", "text": prompt})
        logger.info(
            "anthropic_request",
            model=self._model,
            prompt_len=len(prompt),
            image_count=len(images),
        )
        return await self._post_messages(content)

    async def _post_messages(self, content: list[dict]) -> str:
        headers = {
            "x-api-key": self._settings.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.read_timeout_seconds, connect=30.0)
            ) as client:
                data = None
                last_failure = "no attempts made"
                total_attempts = 0
                network_retries = 0
                rate_retries = 0
                while True:
                    total_attempts += 1
                    try:
                        async with self._sem:
                            resp = await client.post(
                                self._url, headers=headers, json=payload
                            )
                    except httpx.ReadTimeout as exc:
                        logger.warning(
                            "llm_read_timeout",
                            model=self.model_id,
                            attempts=total_attempts,
                            threshold_s=self.read_timeout_seconds,
                            detail=repr(exc)[:160],
                        )
                        raise LLMError(
                            "Anthropic read timeout: "
                            f"tried {total_attempts} time(s); "
                            f"deadline {self.read_timeout_seconds}s; {exc!r}"
                        ) from exc
                    except httpx.RequestError as exc:
                        last_failure = f"network error: {exc!r}"
                        retry_limit = int(isinstance(
                            exc, (httpx.ConnectError, httpx.ProtocolError)))
                        if network_retries >= retry_limit:
                            self._log_transient_retry(
                                attempt=network_retries, delay=0,
                                detail=repr(exc), exhausted=True,
                                retry_limit=retry_limit)
                            break
                        delay = self._compute_retry_delay(None, network_retries)
                        self._log_transient_retry(
                            attempt=network_retries, delay=delay,
                            detail=repr(exc), retry_limit=retry_limit,
                        )
                        network_retries += 1
                        await asyncio.sleep(delay)
                        continue

                    if resp.status_code == 429:
                        self.record_http_429()
                        last_failure = f"HTTP 429: {resp.text[:300]}"
                        if rate_retries >= self._max_retries:
                            self._log_rate_limited(
                                resp, attempt=rate_retries, delay=0,
                                exhausted=True)
                            break
                        delay = self._compute_retry_delay(resp, rate_retries)
                        self._log_rate_limited(
                            resp, attempt=rate_retries, delay=delay)
                        rate_retries += 1
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    break

                if data is None:
                    raise LLMError(
                        f"Anthropic API error after {total_attempts} "
                        f"attempts: {last_failure}"
                    )

            blocks = data.get("content") or []
            texts = [
                b.get("text") or ""
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = "".join(texts)
            logger.info("anthropic_response", response_len=len(text))
            return text

        except httpx.HTTPStatusError as exc:
            error_body = exc.response.text[:500]
            raise LLMError(
                f"Anthropic API error (HTTP {exc.response.status_code}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            raise LLMError(f"Anthropic network error: {exc!r}") from exc
