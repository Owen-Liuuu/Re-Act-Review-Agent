"""How a production run is assembled — in ONE place, so a test can run it.

The wiring is the part that keeps being wrong while every component is right.
The route was resolved per claim and the CLI passed a single profile; the
runtime was bound and the CLI never built one; the telemetry existed and was
created after the parsing it was supposed to measure. Every one of those passed
a suite in which the tests built their own Collector.

So the assembly is a function, the CLI calls it, and the tests call the same
one. A test that rebuilds the wiring by hand only ever proves that the wiring a
test writes works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from react_review.llm.metered import MeteredBackend
from react_review.schemas.telemetry import (
    BATCH_EXTRACTION,
    REVIEW_PARSING,
    SEMANTIC,
    SINGLE_EXTRACTION,
    RunTelemetry,
)


@dataclass
class ProductionStages:
    """Which stage label each backend view carries.

    Empty labels when the run has only one route to compare: per-stage numbers
    exist to set one reading against another, and a single-route run has
    nothing to set it against while its global counters already say what it
    cost. Labelling it anyway adds a section to every artifact ever replayed.
    """

    parsing: str = ""
    single: str = ""
    batch: str = ""
    semantic: str = ""

    @classmethod
    def of(cls, contract) -> "ProductionStages":
        if contract is None or not getattr(contract, "batching", False):
            return cls()
        return cls(parsing=REVIEW_PARSING, single=SINGLE_EXTRACTION,
                   batch=BATCH_EXTRACTION, semantic=SEMANTIC)


@dataclass
class ProductionBackends:
    """One backend, seen once per stage.

    Parsing is its own view because it is not extraction: the review parser and
    the field resolver call the model before a single source paper is opened,
    and folding those into the extraction bucket would attribute the cost of
    reading the REVIEW to the cost of reading the papers.
    """

    raw: Any
    telemetry: RunTelemetry
    stages: ProductionStages
    parsing: Any = None
    single: Any = None
    batch: Any = None
    semantic: Any = None

    def __post_init__(self) -> None:
        self.parsing = MeteredBackend(self.raw, self.telemetry, self.stages.parsing)
        self.single = MeteredBackend(self.raw, self.telemetry, self.stages.single)
        self.batch = MeteredBackend(self.raw, self.telemetry, self.stages.batch)
        self.semantic = MeteredBackend(self.raw, self.telemetry, self.stages.semantic)


def aggregation_runtime(contract):
    """The policy and the identity that cleared it, bound — or nothing.

    Nothing when the contract does not batch: recording an identity a run never
    used would attribute its answers to code that never ran, and resolving one
    costs a git subprocess a non-batching run has no reason to pay.
    """
    if contract is None or not getattr(contract, "batching", False):
        return None
    from react_review.tools.safe_aggregation import AggregationRuntime

    return AggregationRuntime.resolve(
        policy_id=contract.aggregation_policy_id,
        evaluator_version=contract.evaluator_version)


def build_collector(registry, *, contract, knowledge=None, cohorts=None,
                    knowledge_fingerprint: str = "", telemetry=None,
                    runtime=None):
    """The Collector a production run uses.

    Everything it needs to honour its contract, in one call: the routes, the
    bound runtime, the knowledge fingerprint that goes into a batch question,
    and the telemetry. Each of those was separately missing at some point while
    the tests, which built their own, stayed green.
    """
    from react_review.agents.collector import Collector

    return Collector(
        registry, knowledge=knowledge, cohorts=cohorts, contract=contract,
        aggregation_runtime=(runtime if runtime is not None
                             else aggregation_runtime(contract)),
        knowledge_fingerprint=knowledge_fingerprint, telemetry=telemetry,
        extraction_profile=contract.extraction_profile)


def record_cache_totals(telemetry: RunTelemetry, *, extraction=None,
                        semantic=None, stages: ProductionStages | None = None) -> None:
    """Fold each cache's own totals in, once, when the run ends.

    Both books are kept: the GLOBAL counters, which is where a cache's hits have
    always been reported and what every existing artifact reads, and the stage
    buckets, which is the only way to say whether the batch path or the single
    one was the cheap half. Counting the same event into both is not double
    counting — they answer different questions — but counting it into either
    twice would be, so this is the one place it happens.
    """
    labels = stages or ProductionStages()
    for cache, stage in ((extraction, labels.single), (semantic, labels.semantic)):
        if cache is None:
            continue
        hits, misses = int(getattr(cache, "hits", 0)), int(getattr(cache, "misses", 0))
        telemetry.record_cache(hits=hits, misses=misses)
        if stage:
            telemetry.record_stage_cache(stage, hits=hits, misses=misses)
