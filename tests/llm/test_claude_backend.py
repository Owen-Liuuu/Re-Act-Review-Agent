"""Anthropic Messages API adapter."""
from __future__ import annotations

import json

import pytest

from react_review.core.config import LLMSettings
from react_review.core.exceptions import LLMError
from react_review.llm.claude_backend import ClaudeBackend

_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes"
_URL = "https://api.anthropic.com/v1/messages"


def _settings(**kwargs) -> LLMSettings:
    body = dict(
        provider="anthropic",
        api_key="sk-ant-test",
        model="claude-sonnet-5",
        max_retries=2,
        retry_base_delay=0.0,
    )
    body.update(kwargs)
    return LLMSettings(**body)


@pytest.mark.asyncio
async def test_complete_reads_text_blocks(httpx_mock):
    httpx_mock.add_response(
        url=_URL,
        json={"content": [{"type": "text", "text": "hello from claude"}]},
    )
    text = await ClaudeBackend(_settings()).complete("prompt")
    assert text == "hello from claude"
    req = httpx_mock.get_request()
    assert req.headers["x-api-key"] == "sk-ant-test"
    payload = json.loads(req.content)
    assert payload["model"] == "claude-sonnet-5"
    assert payload["messages"][0]["content"][0]["text"] == "prompt"


@pytest.mark.asyncio
async def test_complete_vision_sends_image_then_text(httpx_mock):
    httpx_mock.add_response(
        url=_URL,
        json={"content": [{"type": "text", "text": "plot"}]},
    )
    text = await ClaudeBackend(_settings()).complete_vision("read", [_PNG])
    assert text == "plot"
    payload = json.loads(httpx_mock.get_request().content)
    blocks = payload["messages"][0]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[-1] == {"type": "text", "text": "read"}


@pytest.mark.asyncio
async def test_missing_key_raises_before_http():
    with pytest.raises(LLMError, match="api_key"):
        ClaudeBackend(_settings(api_key=""))
