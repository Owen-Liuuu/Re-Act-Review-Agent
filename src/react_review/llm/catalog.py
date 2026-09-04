"""Homepage model catalog: OpenRouter slugs → native vendor calls.

The dropdown IDs follow OpenRouter so the list stays comparable to the public
catalogue. The HTTP call never goes to OpenRouter: ``resolve`` strips the slug
to a native model name and picks that vendor's own base_url.
"""
from __future__ import annotations

from dataclasses import dataclass

from react_review.core.exceptions import ConfigError

VENDORS = ("DeepSeek", "GPT", "Claude", "GLM", "Qwen", "Kimi")

TEXT_MODELS: dict[str, tuple[str, ...]] = {
    "DeepSeek": (
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v3.2",
        "deepseek/deepseek-r1-0528",
    ),
    "GPT": (
        "openai/gpt-5.4",
        "openai/gpt-5.5",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
    ),
    "Claude": (
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-4.6",
    ),
    "GLM": (
        "z-ai/glm-5.3",
        "z-ai/glm-5.2",
        "z-ai/glm-5",
        "z-ai/glm-4.7",
    ),
    "Qwen": (
        "qwen/qwen3.8-max",
        "qwen/qwen3.7-max",
        "qwen/qwen3.8-flash",
        "qwen/qwen3.7-plus",
    ),
    "Kimi": (
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k2.5",
        "moonshotai/kimi-k2",
        "moonshotai/kimi-k2-thinking",
    ),
}

VISION_MODELS: dict[str, tuple[str, ...]] = {
    "DeepSeek": ("deepseek/deepseek-v4-flash-vision-exp",),
    "GPT": ("openai/gpt-5.4", "openai/gpt-5.5", "openai/gpt-5.4-mini"),
    "Claude": (
        "anthropic/claude-sonnet-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-sonnet-4.6",
    ),
    "GLM": ("z-ai/glm-5.3-flash", "z-ai/glm-5v-turbo", "z-ai/glm-4.6v"),
    "Qwen": ("qwen/qwen3.8-max", "qwen/qwen3.8-flash", "qwen/qwen3.7-plus"),
    "Kimi": ("moonshotai/kimi-k3", "moonshotai/kimi-k2.6", "moonshotai/kimi-k2.5"),
}

DEFAULTS = {
    "complex": ("DeepSeek", "deepseek/deepseek-v4-pro"),
    "simple": ("DeepSeek", "deepseek/deepseek-v4-flash"),
    "visual": ("GLM", "z-ai/glm-4.6v"),
}

# Simple / transcribe slots. Complex stays on ``llm`` (unlisted steps).
SIMPLE_STEPS = (
    "table_capture",
    "forest_ocr_text",
    "claim_origin",
    "unpivot",
    "references",
)

_VENDOR = {
    "DeepSeek": {
        "provider": "openai",
        "base_url": "https://api.deepseek.com",
        "placeholder": "sk-…",
        "where": "platform.deepseek.com",
    },
    "GPT": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "placeholder": "sk-…",
        "where": "platform.openai.com",
    },
    "Claude": {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "placeholder": "sk-ant-…",
        "where": "console.anthropic.com",
    },
    "GLM": {
        "provider": "glm",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "placeholder": "…",
        "where": "open.bigmodel.cn",
    },
    "Qwen": {
        "provider": "qwen",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "placeholder": "sk-…",
        "where": "dashscope.console.aliyun.com",
    },
    "Kimi": {
        "provider": "openai",
        "base_url": "https://api.moonshot.cn/v1",
        "placeholder": "sk-…",
        "where": "platform.moonshot.cn",
    },
}


@dataclass(frozen=True)
class ResolvedGear:
    """One homepage gear ready to write onto ``AppConfig``."""

    vendor: str
    slug: str
    provider: str
    model: str
    base_url: str


def native_model(slug: str) -> str:
    """Last path segment of an OpenRouter-style slug (``a/b`` → ``b``)."""
    text = (slug or "").strip()
    if not text:
        raise ConfigError("empty model slug")
    return text.rsplit("/", 1)[-1]


def reject_openrouter_key(key: str) -> None:
    """Customer keys must be native. OpenRouter ``sk-or-…`` is a hard error."""
    if (key or "").strip().lower().startswith("sk-or-"):
        raise ConfigError(
            "OpenRouter keys are not accepted; paste the native vendor key")


def resolve(vendor: str, slug: str, *, vision: bool = False) -> ResolvedGear:
    """Map a homepage vendor + catalog slug to a native provider call."""
    name = (vendor or "").strip()
    if name not in _VENDOR:
        raise ConfigError(
            f"unknown vendor {vendor!r}; known: {', '.join(VENDORS)}")
    text = (slug or "").strip()
    if not text:
        raise ConfigError("empty model slug")
    allowed = VISION_MODELS if vision else TEXT_MODELS
    if text not in allowed[name]:
        raise ConfigError(
            f"model {text!r} is not in the {name} "
            f"{'vision' if vision else 'text'} catalog")
    meta = _VENDOR[name]
    return ResolvedGear(
        vendor=name,
        slug=text,
        provider=meta["provider"],
        model=native_model(text),
        base_url=meta["base_url"],
    )


def catalog_payload() -> dict:
    """JSON-serialisable catalogue for the desk homepage script."""
    return {
        "vendors": list(VENDORS),
        "text": {k: list(v) for k, v in TEXT_MODELS.items()},
        "vision": {k: list(v) for k, v in VISION_MODELS.items()},
        "defaults": {k: {"vendor": a, "model": b} for k, (a, b) in DEFAULTS.items()},
        "hints": {
            name: {"placeholder": meta["placeholder"], "where": meta["where"]}
            for name, meta in _VENDOR.items()
        },
    }
