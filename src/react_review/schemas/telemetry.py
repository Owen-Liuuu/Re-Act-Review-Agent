"""What a run cost, in units that are not each other.

D1 will claim a saving against a per-claim baseline, and a claim like that is
only worth making if the baseline was measured in the same units. So the things
that get casually equated are kept apart:

*Tool attempts* are how many times the audit asked. *Backend requests* are how
many of those reached the model — a cached or replayed attempt is not a call.
*Repeated attempts* are asks the audit made again after refusing an answer.
They are counted apart from backend requests because a replay repeats attempts
without reaching any model at all — "10 retries, 0 requests" is only a
contradiction if the two were ever the same thing.

*Characters are not tokens.* Prompt length is recorded as characters, because
that is what can be counted locally; provider token counts are recorded
separately and only when the provider actually reports them. Adding an estimate
to a measurement, under one name, is how a cost report becomes fiction.

*Call seconds are not wall seconds.* Time inside model calls and time the run
took are different numbers, and the gap between them is where concurrency and
local work live.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from pydantic import BaseModel, Field


class RunTelemetry(BaseModel):
    """Counters for one run. Every field is measured, none is inferred."""

    tool_attempts: dict[str, int] = Field(default_factory=dict)
    backend_requests: int = 0
    repeated_attempts: int = 0
    backend_failures: int = 0
    prompt_chars: int = 0
    output_chars: int = 0
    # Only when the provider reports them. None means "not reported", which is
    # different from zero and must not be summed with a character count.
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    call_seconds: float = 0.0
    wall_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    def attempt(self, stage: str) -> None:
        self.tool_attempts[stage] = self.tool_attempts.get(stage, 0) + 1

    def record_call(self, *, prompt: str, output: str, seconds: float,
                    usage: dict | None = None, failed: bool = False) -> None:
        self.backend_requests += 1
        self.prompt_chars += len(prompt or "")
        self.output_chars += len(output or "")
        self.call_seconds += seconds
        if failed:
            self.backend_failures += 1
        if usage:
            for key, field in (("prompt_tokens", "provider_input_tokens"),
                               ("completion_tokens", "provider_output_tokens")):
                value = usage.get(key)
                if isinstance(value, int):
                    setattr(self, field, (getattr(self, field) or 0) + value)

    def record_cache(self, *, hits: int, misses: int) -> None:
        self.cache_hits += hits
        self.cache_misses += misses

    @property
    def attempts_total(self) -> int:
        return sum(self.tool_attempts.values())

    def summary(self) -> str:
        tokens = ("not reported by the provider"
                  if self.provider_input_tokens is None else
                  f"{self.provider_input_tokens} in / {self.provider_output_tokens} out")
        return (
            f"attempts {self.attempts_total} {self.tool_attempts or ''} · "
            f"backend requests {self.backend_requests} "
            f"(failures {self.backend_failures}) · "
            f"repeated attempts {self.repeated_attempts} · "
            f"cache {self.cache_hits} hit / {self.cache_misses} miss · "
            f"chars {self.prompt_chars} in / {self.output_chars} out · "
            f"provider tokens: {tokens} · "
            f"model time {self.call_seconds:.1f}s of {self.wall_seconds:.1f}s wall")


@contextmanager
def wall_clock(telemetry: RunTelemetry):
    started = time.perf_counter()
    try:
        yield telemetry
    finally:
        telemetry.wall_seconds += time.perf_counter() - started
