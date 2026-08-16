"""Cost, in units that are not each other.

D1 will claim a saving against a per-claim baseline. That claim is worth exactly
as much as the baseline's units, so these tests hold the units apart: a replayed
attempt is not a model call, a character is not a token, and time spent inside
calls is not the time the run took.
"""
from __future__ import annotations

import asyncio
import time

from react_review.llm.base import LLMBackend
from react_review.llm.metered import MeteredBackend
from react_review.schemas.telemetry import RunTelemetry, wall_clock


class _Backend(LLMBackend):
    def __init__(self, output="answer", usage=None, fail=False):
        super().__init__()
        self._output, self.last_usage, self._fail = output, usage, fail

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        if self._fail:
            raise RuntimeError("provider said no")
        return self._output


def test_a_replayed_attempt_is_not_a_model_call():
    telemetry = RunTelemetry()
    telemetry.attempt("extract_source_value")
    telemetry.attempt("extract_source_value")
    telemetry.repeated_attempts += 1
    telemetry.record_cache(hits=2, misses=0)
    assert telemetry.attempts_total == 2
    assert telemetry.backend_requests == 0      # nothing reached a model
    assert telemetry.cache_hits == 2


def test_characters_are_never_reported_as_tokens():
    telemetry = RunTelemetry()
    asyncio.run(MeteredBackend(_Backend(), telemetry).complete("a prompt"))
    assert telemetry.prompt_chars == len("a prompt")
    # The provider said nothing about tokens, so neither does the report.
    assert telemetry.provider_input_tokens is None
    assert "not reported by the provider" in telemetry.summary()


def test_provider_tokens_are_recorded_only_when_the_provider_reports_them():
    telemetry = RunTelemetry()
    backend = _Backend(usage={"prompt_tokens": 120, "completion_tokens": 34})
    asyncio.run(MeteredBackend(backend, telemetry).complete("a prompt"))
    assert telemetry.provider_input_tokens == 120
    assert telemetry.provider_output_tokens == 34


def test_a_failed_call_still_counts_as_a_call():
    telemetry = RunTelemetry()
    try:
        asyncio.run(MeteredBackend(_Backend(fail=True), telemetry).complete("x"))
    except RuntimeError:
        pass
    assert telemetry.backend_requests == 1 and telemetry.backend_failures == 1


def test_call_time_and_wall_time_are_separate():
    telemetry = RunTelemetry()
    with wall_clock(telemetry):
        time.sleep(0.01)
        asyncio.run(MeteredBackend(_Backend(), telemetry).complete("x"))
    assert telemetry.wall_seconds >= telemetry.call_seconds
    assert telemetry.wall_seconds > 0


def test_the_wrapper_never_changes_an_answer():
    telemetry = RunTelemetry()
    backend = MeteredBackend(_Backend(output="the answer"), telemetry)
    assert asyncio.run(backend.complete("q")) == "the answer"
    assert backend.model_id == "stub"


# --- "the model never answered" is a fact about the counters -----------------

def test_a_run_that_reached_no_backend_did_not_fail_every_call():
    """A fully replayed run makes no calls, which is not the same as failing
    all of them — reading it that way would call every cached run a failure."""
    telemetry = RunTelemetry()
    telemetry.attempt("extract_source_value")
    telemetry.record_cache(hits=4, misses=0)
    assert telemetry.backend_requests == 0
    assert telemetry.every_call_failed() is False


def test_every_call_failing_is_distinguishable_from_some_of_them_failing():
    every = RunTelemetry()
    for _ in range(3):
        try:
            asyncio.run(MeteredBackend(_Backend(fail=True), every).complete("x"))
        except RuntimeError:
            pass
    assert every.backend_requests == 3 and every.backend_failures == 3
    assert every.every_call_failed() is True

    some = RunTelemetry()
    try:
        asyncio.run(MeteredBackend(_Backend(fail=True), some).complete("x"))
    except RuntimeError:
        pass
    asyncio.run(MeteredBackend(_Backend(), some).complete("x"))
    assert some.backend_requests == 2 and some.backend_failures == 1
    assert some.every_call_failed() is False
