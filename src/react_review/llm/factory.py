"""Constructing an LLM backend from configuration.

One place decides which adapter a provider name means, so a run, a tool
catalogue and a live test all reach the same backend for the same config. The
provider name is the only switch; everything else (keys, base URLs, limits)
travels in :class:`~react_review.core.config.LLMSettings`.

An unknown provider raises rather than falling back to a mock: a typo that
silently produced canned answers would look like a cheap, fast, and completely
fictional run.
"""
from __future__ import annotations

from react_review.core.config import AppConfig, LLMSettings
from react_review.llm.base import LLMBackend
from react_review.llm.mock_backend import MockLLMBackend


def create_llm_backend(config: AppConfig) -> LLMBackend:
    """Create the primary LLM backend from ``config.llm``."""
    return create_backend_from_settings(config.llm)


def create_vision_backend(config: AppConfig) -> LLMBackend | None:
    """Create the vision backend from ``config.vision``, or None if unset."""
    if config.vision is None:
        return None
    return create_backend_from_settings(config.vision)


def create_backend_from_settings(settings: LLMSettings) -> LLMBackend:
    """Create an LLM backend from explicit settings.

    This is the shared factory used for ``llm``, ``llm2``, and ``vision``.
    """
    provider = settings.provider.lower()

    if provider == "mock":
        return MockLLMBackend()

    if provider == "openai":
        from react_review.llm.openai_backend import OpenAIBackend
        return OpenAIBackend(settings)

    if provider == "anthropic":
        from react_review.llm.claude_backend import ClaudeBackend
        return ClaudeBackend(settings)

    if provider == "gemini":
        from react_review.llm.gemini_backend import GeminiBackend
        return GeminiBackend(settings)

    if provider == "qwen":
        from react_review.llm.qwen_backend import QwenBackend
        return QwenBackend(settings)

    if provider in ("glm", "zhipu"):
        # GLM / Zhipu expose an OpenAI-compatible endpoint; reuse OpenAIBackend
        # and default the base_url to BigModel if the config didn't set one.
        from react_review.llm.openai_backend import OpenAIBackend
        if not settings.base_url:
            settings = settings.model_copy(
                update={"base_url": "https://open.bigmodel.cn/api/paas/v4"}
            )
        return OpenAIBackend(settings)

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        "Supported: mock, openai, anthropic, gemini, qwen, glm."
    )
