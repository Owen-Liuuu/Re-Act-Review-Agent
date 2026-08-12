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
import math
from contextlib import contextmanager

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    model_serializer,
    model_validator,
)


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
#: Reading the REVIEW — the parser and the field resolver, both of which call
#: the model before a single source paper is opened. Its own stage because
#: folding it into extraction would attribute the cost of reading the review to
#: the cost of reading the papers.
REVIEW_PARSING = "review_parsing"
SINGLE_EXTRACTION = "single_extraction"
BATCH_EXTRACTION = "batch_extraction"
SEMANTIC = "semantic"
STAGES = (REVIEW_PARSING, SINGLE_EXTRACTION, BATCH_EXTRACTION, SEMANTIC)


class BatchStats(BaseModel):
    """What batching actually did, as counts rather than as a design claim.

    Batching was justified by an argument about cost and consistency. Neither is
    checkable from the backend's totals: a run that issued one prompt for four
    claims and a run that issued four look identical there. These are the
    numbers that decide whether it bought anything, and what it cost when a
    reading went wrong.
    """

    #: Unknown keys are still refused. `claims_per_batch` is the one exception
    #: and is handled below: it must be readable back, because it is written.
    model_config = {"extra": "forbid"}

    batches: int = 0
    claims: int = 0
    #: Batches that answered exactly one claim. The number that decides whether
    #: batching is buying anything at all.
    singleton_batches: int = 0
    failed_batches: int = 0
    #: Every projection outcome, by name. A run whose claims mostly land in
    #: `scope_unresolved` is failing in a different way from one landing in
    #: `not_reported`, and a single "unresolved" count cannot say which.
    projections: dict[str, int] = Field(default_factory=dict)
    aggregation_attempts: int = 0
    aggregation_derived: int = 0
    aggregation_rejected: int = 0
    aggregation_protocol_errors: int = 0
    #: A printed total and a computed one, where both existed. Agreement is
    #: corroboration; a conflict is the paper disagreeing with itself, and the
    #: two must never be summed into one "checked" figure.
    explicit_vs_derived_agreements: int = 0
    explicit_vs_derived_conflicts: int = 0

    @model_validator(mode="before")
    @classmethod
    def _derived_value_is_a_checksum(cls, data):
        """`claims_per_batch` may arrive, but only as the value it must have.

        It is written into every artifact, so refusing it outright made a
        package that saved cleanly fail to load — which broke the report, since
        the report renders from the reloaded file. But accepting it as state
        would put a second source of truth beside `batches` and `claims`, which
        is the fault this project has spent five rounds removing.

        So it is neither: on the way in it is checked against the value it is
        derived from and then discarded. Agreement means nothing was lost in the
        round trip; disagreement means the file has been edited, and a
        disagreement silently corrected is worse than one that stops the load.
        """
        if not isinstance(data, dict) or "claims_per_batch" not in data:
            return data
        body = dict(data)
        supplied = body.pop("claims_per_batch")
        batches = int(body.get("batches") or 0)
        claims = int(body.get("claims") or 0)
        expected = round((claims / batches) if batches else 0.0, 4)
        if (isinstance(supplied, bool)
                or not isinstance(supplied, (int, float))
                or not math.isfinite(float(supplied))
                or float(supplied) != expected):
            raise ValueError(
                f"claims_per_batch is {supplied}, and {claims} claims over "
                f"{batches} batch(es) is {expected}. It is derived from those "
                "two numbers, so a value that disagrees means the record was "
                "edited")
        return body

    @computed_field
    @property
    def claims_per_batch(self) -> float:
        """Derived at serialisation, never supplied.

        It reaches the JSON — a plain property would not, and the number the
        whole cost argument turns on would have been absent from every artifact
        carrying the counts it comes from. But it is not WRITABLE: a settable
        field would let a caller record 999 beside batches=2 and claims=4, which
        is the second-source-of-truth fault this project has spent four rounds
        removing everywhere else.
        """
        return round((self.claims / self.batches) if self.batches else 0.0, 4)


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
    #: Present only for a run that batched. Same rule, same reason.
    batch: BatchStats | None = None

    @model_serializer(mode="wrap")
    def _omit_unused_sections(self, handler):
        body = handler(self)
        for name in ("stages", "batch"):
            if not body.get(name):
                body.pop(name, None)
        return body

    def has_measurements(self) -> bool:
        """Whether anything was actually measured.

        A zero RunTelemetry serialises to a full dictionary of zeroes, which is
        truthy — so testing the object decided nothing and every package written
        by a run that measured nothing gained the key anyway. What matters is
        the counts.
        """
        return bool(
            self.tool_attempts or self.backend_requests or self.cache_hits
            or self.cache_misses or self.repeated_attempts or self.call_seconds
            or self.wall_seconds or self.stages or self.batch)

    def batch_stats(self) -> BatchStats:
        if self.batch is None:
            self.batch = BatchStats()
        return self.batch

    def record_batch(self, *, claims: int, failed: bool) -> None:
        stats = self.batch_stats()
        stats.batches += 1
        stats.claims += claims
        if claims == 1:
            stats.singleton_batches += 1
        if failed:
            stats.failed_batches += 1

    def record_projection(self, status: str, aggregation_status: str) -> None:
        """One claim's outcome, and what the aggregation did for it.

        Agreement and conflict are read structurally rather than from prose: a
        printed total released while a valid sum existed is corroboration, and a
        contradiction beside a valid sum is the paper disagreeing with itself.
        """
        stats = self.batch_stats()
        stats.projections[status] = stats.projections.get(status, 0) + 1
        if aggregation_status and aggregation_status != "not_applicable":
            stats.aggregation_attempts += 1
        if aggregation_status == "derived":
            stats.aggregation_derived += 1
            if status == "ok":
                stats.explicit_vs_derived_agreements += 1
            elif status == "contradictory":
                stats.explicit_vs_derived_conflicts += 1
        elif aggregation_status == "rejected":
            stats.aggregation_rejected += 1
        elif aggregation_status == "protocol_error":
            stats.aggregation_protocol_errors += 1

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

    def set_cache_totals(self, *, hits: int, misses: int) -> None:
        """The run's cache totals, ASSIGNED rather than added.

        Finalisation can be reached more than once — a clean finish, a stop, an
        interrupt and an error do not share one path — and a total that
        accumulates would count the same run twice on the second visit. A
        snapshot cannot: writing the same numbers again writes the same numbers.
        """
        self.cache_hits, self.cache_misses = hits, misses

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
