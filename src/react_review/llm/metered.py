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
from react_review.schemas.telemetry import RunTelemetry


class MeteredBackend(LLMBackend):
    """Delegates to a real backend and records the cost of each call."""

    def __init__(self, backend: LLMBackend, telemetry: RunTelemetry,
                 stage: str = "") -> None:
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

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @property
    def telemetry(self) -> RunTelemetry:
        return self._telemetry

    def __getattr__(self, name: str):
        # Anything the wrapper does not define belongs to the backend it wraps.
        return getattr(self._backend, name)

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        started = time.perf_counter()
        try:
            output = await self._backend.complete(prompt, seed=seed)
        except Exception:
            self._telemetry.record_call(
                prompt=prompt, output="", failed=True,
                seconds=time.perf_counter() - started, stage=self._stage)
            raise
        self._telemetry.record_call(
            prompt=prompt, output=output or "",
            seconds=time.perf_counter() - started,
            usage=getattr(self._backend, "last_usage", None),
            stage=self._stage)
        return output

    async def complete_vision(
        self, prompt: str, images: list[bytes], *, seed: int = 42,
    ) -> str:
        started = time.perf_counter()
        try:
            output = await self._backend.complete_vision(
                prompt, images, seed=seed)
        except Exception:
            self._telemetry.record_call(
                prompt=prompt, output="", failed=True,
                seconds=time.perf_counter() - started, stage=self._stage)
            raise
        self._telemetry.record_call(
            prompt=prompt, output=output or "",
            seconds=time.perf_counter() - started,
            usage=getattr(self._backend, "last_usage", None),
            stage=self._stage)
        return output
