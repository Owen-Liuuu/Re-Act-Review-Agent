"""finish_reason=length with empty content is truncation, not a parse miss."""
from __future__ import annotations

import pytest

from react_review.core.config import LLMSettings
from react_review.core.exceptions import LLMError
from react_review.llm.openai_backend import OpenAIBackend


def _settings(**kwargs) -> LLMSettings:
    body = dict(
        provider="openai",
        api_key="sk-test",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        max_retries=0,
        retry_base_delay=0.0,
        max_tokens=32768,
    )
    body.update(kwargs)
    return LLMSettings(**body)


def _chat_response(*, content: str, finish_reason: str, reasoning_tokens: int = 123):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


@pytest.mark.asyncio
async def test_length_with_empty_content_raises_truncated(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json=_chat_response(content="", finish_reason="length", reasoning_tokens=123),
    )
    with pytest.raises(LLMError, match=r"truncated.*reasoning_tokens=123.*max_tokens=32768"):
        await OpenAIBackend(_settings()).complete("prompt")


@pytest.mark.asyncio
async def test_length_with_content_returns_text(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json=_chat_response(content='{"ok": true}', finish_reason="length"),
    )
    text = await OpenAIBackend(_settings()).complete("prompt")
    assert text == '{"ok": true}'
