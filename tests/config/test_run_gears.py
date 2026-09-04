"""Homepage gears clone AppConfig in memory and keep Simple's key distinct."""
from __future__ import annotations

import pytest

from react_review.core.config import (
    AppConfig,
    LLMSettings,
    apply_env_secrets,
    apply_run_gears,
    keys_required,
)
from react_review.core.exceptions import ConfigError
from react_review.llm.catalog import resolve


def _host() -> AppConfig:
    return AppConfig(
        mock_mode=False,
        llm=LLMSettings(provider="openai", model="host-llm", api_key="HOST-LLM",
                        base_url="https://api.deepseek.com"),
        vision=LLMSettings(provider="glm", model="host-vis", api_key="HOST-VIS",
                           base_url="https://open.bigmodel.cn/api/paas/v4"),
        backend_profiles={
            "transcribe": {
                "provider": "openai",
                "model": "host-simple",
                "api_key": "HOST-SIMPLE",
                "base_url": "https://api.deepseek.com",
                "reasoning": "off",
            }
        },
        routing={"table_capture": "transcribe", "claim_origin": "transcribe"},
    )


def test_empty_keys_keep_host_secrets_and_rewrite_models():
    overlay = apply_run_gears(
        _host(),
        complex=resolve("DeepSeek", "deepseek/deepseek-v4-pro"),
        simple=resolve("DeepSeek", "deepseek/deepseek-v4-flash"),
        visual=resolve("GLM", "z-ai/glm-4.6v", vision=True),
    )
    assert overlay.llm.api_key == "HOST-LLM"
    assert overlay.backend_profiles["transcribe"].api_key == "HOST-SIMPLE"
    assert overlay.vision.api_key == "HOST-VIS"
    assert overlay.llm.model == "deepseek-v4-pro"
    assert overlay.backend_profiles["transcribe"].model == "deepseek-v4-flash"
    assert overlay.vision.model == "glm-4.6v"
    assert overlay.routing["claim_origin"] == "transcribe"
    assert overlay.routing["unpivot"] == "transcribe"


def test_three_keys_do_not_let_simple_inherit_complex():
    overlay = apply_run_gears(
        _host(),
        complex=resolve("DeepSeek", "deepseek/deepseek-v4-pro"),
        simple=resolve("GPT", "openai/gpt-5.4"),
        visual=resolve("GLM", "z-ai/glm-4.6v", vision=True),
        complex_key="sk-complex",
        simple_key="sk-simple",
        visual_key="glm-visual",
    )
    assert overlay.llm.api_key == "sk-complex"
    assert overlay.backend_profiles["transcribe"].api_key == "sk-simple"
    assert overlay.vision.api_key == "glm-visual"
    assert overlay.backend_profiles["transcribe"].provider == "openai"
    assert overlay.backend_profiles["transcribe"].base_url == "https://api.openai.com/v1"


def test_partial_keys_are_a_config_error():
    with pytest.raises(ConfigError, match="all three API keys"):
        apply_run_gears(
            _host(),
            complex=resolve("DeepSeek", "deepseek/deepseek-v4-pro"),
            simple=resolve("DeepSeek", "deepseek/deepseek-v4-flash"),
            visual=resolve("GLM", "z-ai/glm-4.6v", vision=True),
            complex_key="sk-only-one",
        )


def test_openrouter_key_rejected():
    with pytest.raises(ConfigError, match="OpenRouter"):
        apply_run_gears(
            _host(),
            complex=resolve("DeepSeek", "deepseek/deepseek-v4-pro"),
            simple=resolve("DeepSeek", "deepseek/deepseek-v4-flash"),
            visual=resolve("GLM", "z-ai/glm-4.6v", vision=True),
            complex_key="sk-or-v1-nope",
            simple_key="sk-or-v1-nope",
            visual_key="sk-or-v1-nope",
        )


def test_claude_simple_clears_reasoning_flag():
    overlay = apply_run_gears(
        _host(),
        complex=resolve("Claude", "anthropic/claude-sonnet-5"),
        simple=resolve("Claude", "anthropic/claude-sonnet-5"),
        visual=resolve("Claude", "anthropic/claude-sonnet-5", vision=True),
        complex_key="sk-ant-c",
        simple_key="sk-ant-s",
        visual_key="sk-ant-v",
    )
    assert overlay.backend_profiles["transcribe"].reasoning is None
    assert overlay.llm.provider == "anthropic"


def test_transcribe_env_key_beats_llm_inherit(monkeypatch):
    monkeypatch.setenv("REACT_REVIEW_TRANSCRIBE_API_KEY", "FROM-TRANSCRIBE")
    monkeypatch.setenv("REACT_REVIEW_LLM_API_KEY", "FROM-LLM")
    cfg = apply_env_secrets(AppConfig(
        backend_profiles={"transcribe": {"provider": "openai", "model": "x"}}
    ))
    assert cfg.backend_profiles["transcribe"].api_key == "FROM-TRANSCRIBE"


def test_keys_required_only_when_live_and_host_empty():
    assert keys_required(AppConfig(mock_mode=True)) is False
    assert keys_required(AppConfig(
        mock_mode=False, llm=LLMSettings(api_key="x"))) is False
    assert keys_required(AppConfig(mock_mode=False)) is True
