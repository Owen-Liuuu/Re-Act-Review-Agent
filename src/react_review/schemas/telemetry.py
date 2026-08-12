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

from pydantic import BaseModel, Field, model_serializer


class StageTelemetry(BaseModel):
    """What ONE stage of a run cost.

    The global counters below answer "what did this run cost", which stopped
    being enough the moment a run could read the same paper two different ways:
    extraction and semantic comparison share a backend, so a single output-token
    figure cannot say whether batching spent more on output than it saved on
    calls. That question is the reason batching exists, and it needs its own
    numbers.
    """

    requests: int = 0
    failures: int = 0
    prompt_chars: int = 0
    output_chars: int = 0
    #: None means the provider did not report, which is not zero.
    provider_input_tokens: int | None = None
    provider_output_tokens: int | None = None
    call_seconds: float = 0.0
    #: Counted where the CACHE is, not where the backend is — a hit never
    #: reaches a backend, so a wrapper around one cannot see it.
    cache_hits: int = 0
    cache_misses: int = 0


#: The stages a run can spend in. Closed, so a typo becomes a bucket nobody
#: reads rather than a silent third of the cost.
SINGLE_EXTRACTION = "single_extraction"
BATCH_EXTRACTION = "batch_extraction"
SEMANTIC = "semantic"
STAGES = (SINGLE_EXTRACTION, BATCH_EXTRACTION, SEMANTIC)


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
    #: Per-stage costs, present only for a run that recorded any. A run that
    #: never used a stage gains no key for it, and a legacy run gains nothing at
    #: all — every recorded artifact would otherwise change shape for a fact it
    #: does not have. See tests/test_legacy_bytes.py.
    stages: dict[str, StageTelemetry] | None = None

    @model_serializer(mode="wrap")
    def _omit_unused_stages(self, handler):
        body = handler(self)
        if not body.get("stages"):
            body.pop("stages", None)
        return body

    def _stage(self, name: str) -> StageTelemetry:
        if name not in STAGES:
            raise ValueError(
                f"unknown telemetry stage {name!r} (known: {', '.join(STAGES)}); "
                "a typo here becomes a bucket nobody reads rather than a "
                "visible third of the cost")
        if self.stages is None:
            self.stages = {}
        return self.stages.setdefault(name, StageTelemetry())

    def record_stage_cache(self, stage: str, *, hits: int = 0,
                           misses: int = 0) -> None:
        """Cache activity for ONE stage, and only there.

        Deliberately not `record_cache`: the accuracy harness already folds each
        cache's own hit/miss totals into the global counters when a run ends, so
        adding to them here would double every number it reports.
        """
        bucket = self._stage(stage)
        bucket.cache_hits += hits
        bucket.cache_misses += misses

    def attempt(self, stage: str) -> None:
        self.tool_attempts[stage] = self.tool_attempts.get(stage, 0) + 1

    def record_call(self, *, prompt: str, output: str, seconds: float,
                    usage: dict | None = None, failed: bool = False,
                    stage: str = "") -> None:
        """One backend call. The global totals are unchanged; a stage is extra.

        No double counting: the globals were always updated exactly once per
        call and still are. The stage bucket is a second, narrower record of the
        same event.
        """
        if stage:
            bucket = self._stage(stage)
            bucket.requests += 1
            bucket.prompt_chars += len(prompt or "")
            bucket.output_chars += len(output or "")
            bucket.call_seconds += seconds
            if failed:
                bucket.failures += 1
            if usage:
                for key, name in (("prompt_tokens", "provider_input_tokens"),
                                  ("completion_tokens", "provider_output_tokens")):
                    value = usage.get(key)
                    if isinstance(value, int):
                        setattr(bucket, name,
                                (getattr(bucket, name) or 0) + value)
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
