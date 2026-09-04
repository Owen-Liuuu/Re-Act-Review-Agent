"""Vendor factory aliases and Qwen honouring base_url."""
from __future__ import annotations

import pytest

from react_review.core.config import LLMSettings
from react_review.core.exceptions import LLMError
from react_review.llm.factory import create_backend_from_settings
from react_review.llm.openai_backend import OpenAIBackend
from react_review.llm.qwen_backend import QwenBackend


def test_kimi_defaults_to_moonshot_openai_compat():
    backend = create_backend_from_settings(LLMSettings(
        provider="kimi", api_key="sk-kimi", model="kimi-k2.6"))
    assert isinstance(backend, OpenAIBackend)
    assert backend._base_url == "https://api.moonshot.cn/v1"
    assert backend.model_id == "kimi-k2.6"


def test_qwen_uses_configured_base_url():
    backend = QwenBackend(LLMSettings(
        provider="qwen", api_key="sk-qwen", model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"))
    assert backend._base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_qwen_default_is_mainland_not_intl():
    backend = QwenBackend(LLMSettings(
        provider="qwen", api_key="sk-qwen", model="qwen-plus"))
    assert "dashscope-intl" not in backend._base_url
    assert "dashscope.aliyuncs.com" in backend._base_url


def test_claude_missing_key_raises():
    with pytest.raises(LLMError, match="api_key"):
        create_backend_from_settings(LLMSettings(provider="anthropic"))
