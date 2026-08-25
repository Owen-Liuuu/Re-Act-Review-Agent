"""Token-derived deadlines and failure-class retry budgets."""
from __future__ import annotations

import time

import httpx
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
        max_retries=5,
        retry_base_delay=0.0,
        max_tokens=8192,
    )
    body.update(kwargs)
    return LLMSettings(**body)


@pytest.mark.asyncio
async def test_read_timeout_is_not_retried_and_names_the_deadline(httpx_mock):
    httpx_mock.add_exception(httpx.ReadTimeout("generation stalled"))
    started = time.monotonic()
    with pytest.raises(
        LLMError, match=r"read timeout.*tried 1 time\(s\).*deadline 470s"
    ):
        await OpenAIBackend(_settings()).complete("prompt")
    assert time.monotonic() - started < 700
    assert len(httpx_mock.get_requests()) == 1


@pytest.mark.asyncio
async def test_connect_error_gets_one_retry_then_stops(httpx_mock):
    httpx_mock.add_exception(httpx.ConnectError("connect reset"))
    httpx_mock.add_exception(httpx.ConnectError("connect reset again"))
    with pytest.raises(LLMError, match=r"after 2 attempts"):
        await OpenAIBackend(_settings()).complete("prompt")
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_429_keeps_all_five_retries(httpx_mock):
    for _ in range(6):
        httpx_mock.add_response(status_code=429, text="rate limited")
    with pytest.raises(LLMError, match=r"after 6 attempts.*HTTP 429"):
        await OpenAIBackend(_settings()).complete("prompt")
    assert len(httpx_mock.get_requests()) == 6


def test_read_timeout_follows_max_tokens_instead_of_a_constant():
    backend = OpenAIBackend(_settings(max_tokens=4096))
    assert backend.read_timeout_seconds == 265
