"""A backend that counts what it actually did.

Wrapping the backend rather than instrumenting each caller means every model
call is counted the same way, whoever makes it — and that a replayed or cached
attempt is counted as what it is: not a call.

The wrapper never changes an answer. It records, re-raises, and gets out of the
way; a failed call is still counted, because a failure costs time and a report
that hides it understates what the run did.
"""
from __future__ import annotations

import time

from react_review.llm.base import LLMBackend
from react_review.llm.reasoning import (
    reasoning_extra_body,
    reset_reasoning_patch,
    set_backend_trace,
    set_reasoning_patch,
)
from react_review.schemas.telemetry import RunTelemetry


class MeteredBackend(LLMBackend):
    """Delegates to a real backend and records the cost of each call."""

    def __init__(self, backend: LLMBackend, telemetry: RunTelemetry,
                 stage: str = "", *, profile: str = "",
                 reasoning: str | None = None, provider: str = "") -> None:
        """Optionally labelled with the STAGE whose cost this wrapper measures.

        One backend serves extraction and semantic comparison alike, so a single
        set of counters cannot answer whether batching spent more on output than
        it saved on calls — the question batching exists to settle. Wrapping the
        same backend two or three times, each with a fixed label, keeps that
        answerable without threading a stage through every call site.
        """
        super().__init__()
        self._backend = backend
        self._telemetry = telemetry
        self._stage = stage
        self._profile = profile
        self._reasoning = reasoning
        self._provider = provider

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @property
    def telemetry(self) -> RunTelemetry:
        return self._telemetry

    def __getattr__(self, name: str):
        # Anything the wrapper does not define belongs to the backend it wraps.
        return getattr(self._backend, name)

    def _reasoning_patch(self) -> dict:
        settings = getattr(self._backend, "_settings", None)
        model = getattr(settings, "model", "") or self.model_id
        base_url = getattr(settings, "base_url", "") or ""
        provider = self._provider or getattr(settings, "provider", "")
        return reasoning_extra_body(
            provider, reasoning=self._reasoning, model=model, base_url=base_url)

    def _publish_trace(self) -> None:
        if not self._profile and self._reasoning is None:
            return
        tokens = getattr(self._backend, "last_reasoning_tokens", None)
        set_backend_trace({
            "profile": self._profile,
            "model_id": self.model_id,
            "reasoning": self._reasoning or "",
            "reasoning_tokens": tokens,
        })

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        started = time.perf_counter()
        token = set_reasoning_patch(self._reasoning_patch())
        try:
            output = await self._backend.complete(prompt, seed=seed)
        except Exception:
            self._telemetry.record_call(
                prompt=prompt, output="", failed=True,
                seconds=time.perf_counter() - started, stage=self._stage)
            raise
        finally:
            reset_reasoning_patch(token)
        self._telemetry.record_call(
            prompt=prompt, output=output or "",
            seconds=time.perf_counter() - started,
            usage=getattr(self._backend, "last_usage", None),
            stage=self._stage)
        self._publish_trace()
        return output

    async def complete_vision(
        self, prompt: str, images: list[bytes], *, seed: int = 42,
    ) -> str:
        started = time.perf_counter()
        token = set_reasoning_patch(self._reasoning_patch())
        try:
            output = await self._backend.complete_vision(
                prompt, images, seed=seed)
        except Exception:
            self._telemetry.record_call(
                prompt=prompt, output="", failed=True,
                seconds=time.perf_counter() - started, stage=self._stage)
            raise
        finally:
            reset_reasoning_patch(token)
        self._telemetry.record_call(
            prompt=prompt, output=output or "",
            seconds=time.perf_counter() - started,
            usage=getattr(self._backend, "last_usage", None),
            stage=self._stage)
        self._publish_trace()
        return output
