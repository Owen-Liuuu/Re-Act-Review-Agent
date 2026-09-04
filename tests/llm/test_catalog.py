"""OpenRouter catalog slugs resolve to native vendor calls."""
from __future__ import annotations

import pytest

from react_review.core.exceptions import ConfigError
from react_review.llm.catalog import native_model, reject_openrouter_key, resolve


def test_resolve_deepseek_complex_default():
    gear = resolve("DeepSeek", "deepseek/deepseek-v4-pro")
    assert gear.provider == "openai"
    assert gear.model == "deepseek-v4-pro"
    assert gear.base_url == "https://api.deepseek.com"
    assert "openrouter" not in gear.base_url


def test_native_model_strips_slug():
    assert native_model("anthropic/claude-sonnet-5") == "claude-sonnet-5"


def test_unknown_vendor_raises():
    with pytest.raises(ConfigError, match="unknown vendor"):
        resolve("Acme", "acme/model")


def test_empty_slug_raises():
    with pytest.raises(ConfigError, match="empty model slug"):
        native_model("  ")


def test_rejects_openrouter_key():
    with pytest.raises(ConfigError, match="OpenRouter"):
        reject_openrouter_key("sk-or-v1-secret")
    reject_openrouter_key("sk-native")


def test_vision_slug_rejected_on_text_gear():
    with pytest.raises(ConfigError, match="not in the GLM text"):
        resolve("GLM", "z-ai/glm-4.6v")
    gear = resolve("GLM", "z-ai/glm-4.6v", vision=True)
    assert gear.model == "glm-4.6v"
