"""Vision is an optional backend path: default raises, OpenAI sends image blocks."""
from __future__ import annotations

import json

import pytest

from react_review.core.config import AppConfig, LLMSettings, load_config
from react_review.llm.factory import create_backend_from_settings, create_vision_backend
from react_review.llm.metered import MeteredBackend
from react_review.llm.mock_backend import MockLLMBackend
from react_review.llm.openai_backend import OpenAIBackend, _image_data_url
from react_review.schemas.telemetry import RunTelemetry

_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes"


def _settings(**kwargs) -> LLMSettings:
    body = dict(
        provider="openai",
        api_key="sk-test",
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        max_retries=2,
        retry_base_delay=0.0,
    )
    body.update(kwargs)
    return LLMSettings(**body)


def test_vision_defaults_to_none():
    assert AppConfig().vision is None


def test_create_vision_backend_returns_none_when_unset():
    assert create_vision_backend(AppConfig()) is None


def test_create_vision_backend_from_glm_settings():
    config = AppConfig(
        vision=LLMSettings(
            provider="glm",
            model="glm-4.6v-flash",
            api_key="sk-test",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
    )
    backend = create_vision_backend(config)
    assert isinstance(backend, OpenAIBackend)
    assert backend.model_id == "glm-4.6v-flash"


def test_load_config_vision_block(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "vision:\n  provider: glm\n  model: glm-4.6v-flash\n  api_key: k\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.vision is not None
    assert config.vision.model == "glm-4.6v-flash"


@pytest.mark.asyncio
async def test_mock_backend_has_no_vision_path():
    with pytest.raises(NotImplementedError, match="has no vision path"):
        await MockLLMBackend().complete_vision("read", [_PNG])


def test_image_data_url_uses_png_prefix():
    url = _image_data_url(_PNG)
    assert url.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_complete_vision_sends_image_then_text_blocks(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": '{"ok": true}'},
                           "finish_reason": "stop"}]},
    )
    text = await OpenAIBackend(_settings()).complete_vision("read this", [_PNG])
    assert text == '{"ok": true}'
    payload = json.loads(httpx_mock.get_request().content)
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1] == {"type": "text", "text": "read this"}
    assert payload["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_complete_still_sends_a_string_content(httpx_mock):
    httpx_mock.add_response(
        url="https://api.openai.com/v1/chat/completions",
        json={"choices": [{"message": {"content": "plain"},
                           "finish_reason": "stop"}]},
    )
    text = await OpenAIBackend(_settings()).complete("hello")
    assert text == "plain"
    payload = json.loads(httpx_mock.get_request().content)
    assert payload["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_complete_vision_retries_429_on_the_shared_loop(httpx_mock):
    url = "https://api.openai.com/v1/chat/completions"
    httpx_mock.add_response(url=url, status_code=429)
    httpx_mock.add_response(
        url=url,
        json={"choices": [{"message": {"content": "after-retry"},
                           "finish_reason": "stop"}]},
    )
    text = await OpenAIBackend(_settings()).complete_vision("read", [_PNG])
    assert text == "after-retry"
    assert len(httpx_mock.get_requests()) == 2


@pytest.mark.asyncio
async def test_metered_backend_counts_vision_calls():
    class _Vision(MockLLMBackend):
        async def complete_vision(self, prompt, images, *, seed=42):
            return "seen"

    telemetry = RunTelemetry()
    out = await MeteredBackend(_Vision(), telemetry).complete_vision("p", [_PNG])
    assert out == "seen"
    assert telemetry.backend_requests == 1


def test_factory_mock_is_not_a_vision_backend():
    backend = create_backend_from_settings(LLMSettings(provider="mock"))
    assert isinstance(backend, MockLLMBackend)
