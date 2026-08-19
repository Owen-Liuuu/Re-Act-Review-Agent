"""Reasoning injection and per-call backend traces.

MeteredBackend is the only place that *decides* the patch; the HTTP backends
read the contextvar when they build a request body. complete() signatures stay
untouched.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_PATCH: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_reasoning_patch", default=None)
_TRACE: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_backend_trace", default=None)


def reasoning_extra_body(
    provider: str, *, reasoning: str | None, model: str = "",
    base_url: str = "",
) -> dict[str, Any]:
    """Provider-specific body keys. Empty when reasoning should not be touched."""
    if reasoning not in {"on", "off"}:
        return {}
    enabled = reasoning == "on"
    kind = _provider_kind(provider, model=model, base_url=base_url)
    if kind == "glm":
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    if kind == "deepseek":
        return {"thinking": {"type": "enabled" if enabled else "disabled"}}
    if kind == "openai":
        return {"reasoning_effort": "medium" if enabled else "none"}
    return {}


def _provider_kind(provider: str, *, model: str = "", base_url: str = "") -> str:
    name = (provider or "").lower()
    blob = f"{model} {base_url}".lower()
    if name in {"glm", "zhipu"} or "bigmodel" in blob:
        return "glm"
    if name == "deepseek" or "deepseek" in blob:
        return "deepseek"
    if name == "openai":
        return "openai"
    return name


def set_reasoning_patch(patch: dict[str, Any] | None):
    return _PATCH.set(patch or None)


def reset_reasoning_patch(token) -> None:
    _PATCH.reset(token)


def current_reasoning_patch() -> dict[str, Any]:
    return dict(_PATCH.get() or {})


def set_backend_trace(trace: dict[str, Any] | None) -> None:
    _TRACE.set(trace)


def take_backend_trace() -> dict[str, Any] | None:
    """Return the last call's trace and clear it so a silent step cannot inherit it."""
    trace = _TRACE.get()
    _TRACE.set(None)
    return trace
