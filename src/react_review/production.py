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


def snapshot_cache_totals(telemetry: RunTelemetry, *, extraction=None,
                          semantic=None) -> None:
    """The run's GLOBAL cache totals, assigned once and idempotently.

    Only the global book. The stage buckets are filled where each lookup
    happens, because a shared cache cannot be attributed afterwards: the
    extraction cache serves both the single-target and the batch tool, and
    adding its whole total to either reported three hits where one occurred.

    Assigned rather than added, because finalisation is reachable by four paths
    — finished, stopped, interrupted, failed — and a total that accumulates
    counts the run twice the second time anything calls this.
    """
    hits = sum(int(getattr(c, "hits", 0)) for c in (extraction, semantic)
               if c is not None)
    misses = sum(int(getattr(c, "misses", 0)) for c in (extraction, semantic)
                 if c is not None)
    telemetry.set_cache_totals(hits=hits, misses=misses)


# --- the end of a run, which happens exactly once ---------------------------

RUNNING, FINALISED = "running", "finalised"

COMPLETE = "complete"
STOPPED_BY_USER = "stopped_by_user"
INTERRUPTED = "interrupted"
ERROR = "error"


class FinalisationFailed(RuntimeError):
    """The run finished, and its result could not be made durable.

    Separate from any failure of the audit itself. The audit produced an answer;
    what failed is the promise that the answer is on disk and readable — and a
    run that cannot keep that promise must not report success.
    """


@dataclass
class ProductionSession:
    """Owns the end of a run: the clock, the books, and the one final save.

    Four ways out — finished, stopped, interrupted, failed — and every one of
    them has to leave the same three things behind: an artifact that says how
    the run ended, telemetry covering the whole execution, and a saved cache.
    Spread across four `except` branches they were not the same three things:
    the stop path recorded no telemetry, the reason it printed was never written
    down, and an interrupt before the first paper left no artifact at all, so a
    run that stopped early was indistinguishable from one that never started.

    Two rules give this object its shape:

    * **Finalisation is idempotent.** The paths are not mutually exclusive in
      practice — an error during a stop is ordinary — so a second call must
      change nothing rather than double the books or overwrite the first, truer,
      reason.
    * **Saving the final package is this object's authority alone.** It was the
      Pipeline's, which meant the package was written before the run had
      finished spending; the telemetry inside it was therefore a snapshot from
      before the last paper, and the fix for that was to save a second time —
      two writes, two different files' worth of bytes, one filename.
    """

    store: Any
    run_id: str
    telemetry: RunTelemetry
    manifest: Any = None
    extraction_cache: Any = None
    semantic_cache: Any = None
    emit: Any = print
    state: str = RUNNING
    outcome: str = ""
    _clock: Any = field(default=None, repr=False)

    # --- the clock -----------------------------------------------------------

    def __enter__(self) -> "ProductionSession":
        """Start measuring EXECUTION: parsing and the audit.

        Not the whole process. Rendering a report from a finished package is not
        part of what the run cost, and counting it would make the same audit
        look more expensive for having been reported on.
        """
        from react_review.schemas.telemetry import wall_clock

        self._clock = wall_clock(self.telemetry)
        self._clock.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop_clock()
        return False

    def _stop_clock(self) -> None:
        if self._clock is not None:
            self._clock.__exit__(None, None, None)
            self._clock = None

    # --- the books -----------------------------------------------------------

    def _close_books(self) -> None:
        """Stop the clock and take the cache snapshot. Safe to repeat."""
        self._stop_clock()
        snapshot_cache_totals(self.telemetry, extraction=self.extraction_cache,
                              semantic=self.semantic_cache)
        if self.semantic_cache is not None:
            self.semantic_cache.save()

    # --- the four ways out ---------------------------------------------------

    def finalise_success(self, package):
        """Save the finished package once, then RELOAD it and return that.

        The reload is not a check that can be skipped on a good day. The report
        renders from the file, so a package that saves and does not load again
        is a run that reports success and produces an unreadable result — which
        is exactly what the derived-value round trip turned out to be. Better to
        fail here, holding the file, than downstream holding nothing.
        """
        if self.state == FINALISED:
            return self.store.load(self.run_id)
        self._close_books()
        final = package.model_copy(update={"telemetry": self.telemetry})
        path = self.store.save(final)
        try:
            reloaded = self.store.load(self.run_id)
        except Exception as exc:                                   # noqa: BLE001
            raise FinalisationFailed(
                f"the package was written to {path} and could not be read back: "
                f"{exc}. The run's own result is not loadable, so it cannot be "
                "reported as complete") from exc
        self.state, self.outcome = FINALISED, COMPLETE
        return reloaded

    def finalise_stopped(self, *, stage: str, reason: str) -> None:
        self._finalise_early(STOPPED_BY_USER, stage=stage, reason=reason)

    def finalise_interrupted(self, *, reason: str = "interrupted (Ctrl-C)") -> None:
        self._finalise_early(INTERRUPTED, stage="", reason=reason)

    def finalise_error(self, exc: BaseException) -> None:
        """A run that died of an exception is evidence too, and says so.

        It used to leave a partial marked `in_progress`, which reads as a run
        still going — the one thing it certainly is not.
        """
        self._finalise_early(ERROR, stage="", reason=f"{type(exc).__name__}: {exc}")

    def _finalise_early(self, status: str, *, stage: str, reason: str) -> None:
        """Leave an artifact that says how the run ended — always one.

        Including when nothing had been collected yet. A partial written only
        after the first study meant an interrupt during parsing left a run
        directory with no package in it, and no way to tell that from a run that
        was never started.
        """
        if self.state == FINALISED:
            return
        self._close_books()
        run_dir = self.store.run_dir(self.run_id)
        partial = run_dir / "package.partial.json"
        existed = partial.is_file()
        if existed:
            self._patch_partial(partial, status=status, stage=stage, reason=reason)
        else:
            from react_review.schemas.package import EvidencePackage

            self.store.save_partial(EvidencePackage(
                run_id=self.run_id, run_manifest=self.manifest, status=status,
                stopped_at_stage=stage, stop_reason=reason,
                telemetry=self.telemetry))
        self.state, self.outcome = FINALISED, status
        self.emit(f"\n[{status}] {reason}")
        self.emit(f"[ARTIFACTS] {run_dir.resolve()}")
        if existed:
            self.emit(f"[PARTIAL]   {partial.resolve()} — evidence collected so far")

    def _patch_partial(self, path, *, status: str, stage: str, reason: str) -> None:
        """Add how it ended to the evidence already on disk, without risking it.

        Patched as JSON rather than rebuilt through the model, because the
        evidence in that file is the only copy: a package written by an older
        version that no longer validates would, on the rebuild path, be replaced
        by a minimal one — losing the collected evidence in order to record that
        collection had stopped.
        """
        import json

        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            data["status"], data["stopped_at_stage"] = status, stage
            data["stop_reason"] = reason
            data["telemetry"] = self.telemetry.model_dump(mode="json")
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        except Exception as exc:                                   # noqa: BLE001
            self.emit(f"[WARN] could not record the outcome in {path}: {exc}")
