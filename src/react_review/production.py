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

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from react_review.llm.metered import MeteredBackend
from react_review.schemas.telemetry import (
    BATCH_EXTRACTION,
    REVIEW_PARSING,
    SEMANTIC,
    SINGLE_EXTRACTION,
    RunTelemetry,
)


# Which of the four telemetry buckets a routed step counts toward.
_SLOT_BUCKET = {
    "review_lens": "parsing",
    "evidence_localize": "parsing",
    "table_capture": "parsing",
    "forest_ocr_vision": "parsing",
    "forest_ocr_text": "parsing",
    "claim_origin": "parsing",
    "unpivot": "parsing",
    "references": "parsing",
    "field_resolution": "parsing",
    "extract_locate": "single",
    "extract_transcribe": "single",
    "semantic_compare": "semantic",
}


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
    """One default backend, plus per-step slots that may wrap a different gear.

    Parsing is its own view because it is not extraction: the review parser and
    the field resolver call the model before a single source paper is opened,
    and folding those into the extraction bucket would attribute the cost of
    reading the REVIEW to the cost of reading the papers.

    Unconfigured (no ``backend_profiles`` / ``routing``) every slot wraps
    ``raw`` — the four telemetry views stay the same objects' counting story.
    """

    raw: Any
    telemetry: RunTelemetry
    stages: ProductionStages
    config: Any = None
    vision_raw: Any = None
    parsing: Any = None
    single: Any = None
    batch: Any = None
    semantic: Any = None
    review_lens: Any = None
    evidence_localize: Any = None
    table_capture: Any = None
    forest_ocr_vision: Any = None
    forest_ocr_text: Any = None
    claim_origin: Any = None
    unpivot: Any = None
    references: Any = None
    field_resolution: Any = None
    extract_locate: Any = None
    extract_transcribe: Any = None
    semantic_compare: Any = None

    def __post_init__(self) -> None:
        self.parsing = MeteredBackend(self.raw, self.telemetry, self.stages.parsing)
        extract_raw = self._raw_for("extract_transcribe")
        self.single = MeteredBackend(
            extract_raw, self.telemetry, self.stages.single,
            **self._meter_kw("extract_transcribe"))
        self.batch = MeteredBackend(
            extract_raw, self.telemetry, self.stages.batch,
            **self._meter_kw("extract_transcribe"))
        semantic_raw = self._raw_for("semantic_compare")
        self.semantic = MeteredBackend(
            semantic_raw, self.telemetry, self.stages.semantic,
            **self._meter_kw("semantic_compare"))
        for name, bucket in _SLOT_BUCKET.items():
            if name == "forest_ocr_vision":
                inner = self._vision_for()
                wrapped = (MeteredBackend(
                    inner, self.telemetry, getattr(self.stages, bucket),
                    **self._meter_kw(name)) if inner is not None else None)
            elif name == "extract_transcribe":
                wrapped = self.single
            elif name == "semantic_compare":
                wrapped = self.semantic
            else:
                wrapped = MeteredBackend(
                    self._raw_for(name), self.telemetry,
                    getattr(self.stages, bucket), **self._meter_kw(name))
            setattr(self, name, wrapped)

    def _routing(self) -> dict[str, str]:
        config = self.config
        if config is None:
            return {}
        return dict(getattr(config, "routing", None) or {})

    def _raw_for(self, step: str):
        name = self._routing().get(step)
        if not name or self.config is None:
            return self.raw
        cache = getattr(self, "_profile_raws", None)
        if cache is None:
            cache = {}
            self._profile_raws = cache
        if name not in cache:
            from react_review.core.config import settings_from_profile
            from react_review.llm.factory import create_backend_from_settings
            cache[name] = create_backend_from_settings(
                settings_from_profile(self.config, name))
        return cache[name]

    def _vision_for(self):
        name = self._routing().get("forest_ocr_vision")
        if name and self.config is not None:
            return self._raw_for("forest_ocr_vision")
        return self.vision_raw

    def _meter_kw(self, step: str) -> dict:
        name = self._routing().get(step, "")
        if not name or self.config is None:
            return {}
        profile = self.config.backend_profiles[name]
        return {
            "profile": name,
            "reasoning": profile.reasoning,
            "provider": profile.provider,
        }


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


def evidence_adequacy_runtime(contract):
    """Resolve the registered claim-level gate named by a schema-v4 contract."""
    if contract is None or not getattr(contract, "adequacy_enabled", False):
        return None
    from react_review.audit.evidence_adequacy import EvidenceAdequacyEvaluator
    from react_review.contracts import ContractError

    evaluator = EvidenceAdequacyEvaluator.resolve(
        policy_id=contract.adequacy_policy_id,
        evaluator_version=contract.adequacy_evaluator_version)
    who = evaluator.identity
    expected = (
        contract.adequacy_policy_id,
        contract.adequacy_policy_hash,
        contract.adequacy_evaluator_id,
        contract.adequacy_evaluator_version,
        contract.adequacy_evaluator_hash,
    )
    actual = (
        who.policy_id, who.policy_sha256, who.evaluator_id,
        who.evaluator_version, who.evaluator_hash,
    )
    if actual != expected or not who.release_eligible:
        raise ContractError(
            "the resolved evidence adequacy evaluator is not the registered, "
            "release-eligible identity pinned by the run contract")
    return evaluator


def build_collector(registry, *, contract, knowledge=None, cohorts=None,
                    knowledge_fingerprint: str = "", telemetry=None,
                    runtime=None, adequacy_evaluator=None):
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
        adequacy_evaluator=(
            adequacy_evaluator if adequacy_evaluator is not None
            else evidence_adequacy_runtime(contract)),
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


@dataclass
class ProductionDependencies:
    """The three things a production run reaches OUTSIDE itself for.

    The model, the papers, and the person. `_run_main` builds everything else —
    the contract, the telemetry, the parser, the resolver, the collector, the
    session — and a test that replaced any of those would be testing its own
    wiring, which is how a signature and its only call site drifted apart twice
    while the suite stayed green. Substituting these three is enough to run the
    whole entry point offline, and leaves the parsing, the field resolution and
    the extraction to the code that does them in production.

    The gate is here because it is the human operator rather than a component of
    the pipeline, and because it is the only way to reach the stop path from
    outside: `RunStopped` is raised nowhere but the reporter, on a gate's
    decision, so without this the entry point's stop branch could only ever be
    tested on a session a test constructed for itself.

    Not a place to grow beyond that. A `parser` field here would turn the one
    test that exercises review parsing end to end into a lifecycle test that
    never parses anything.
    """

    backend: Any = None
    retriever: Any = None
    gate: Any = None

    def llm(self, config):
        if self.backend is not None:
            return self.backend
        from react_review.llm.factory import create_llm_backend

        return create_llm_backend(config)

    def papers(self, build):
        return self.retriever if self.retriever is not None else build()

    def checkpoint(self, build):
        return self.gate if self.gate is not None else build()


# --- the end of a run, which happens exactly once ---------------------------

RUNNING = "running"
FINALISING = "finalising"
FINALISED_SUCCESS = "finalised_success"
FINALISED_PARTIAL = "finalised_partial"
FINALISATION_FAILED = "finalisation_failed"
TERMINAL_STATES = (FINALISED_SUCCESS, FINALISED_PARTIAL, FINALISATION_FAILED)

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
    execution: Any = None
    extraction_cache: Any = None
    semantic_cache: Any = None
    emit: Any = print
    state: str = RUNNING
    outcome: str = ""
    _clock: Any = field(default=None, repr=False)
    _artifact: Any = field(default=None, init=False, repr=False)
    _failure: FinalisationFailed | None = field(default=None, init=False, repr=False)

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

    def _close_books(self) -> list[dict[str, str]]:
        """Stop/snapshot first, then try every cache without hiding failures."""
        self._stop_clock()
        snapshot_cache_totals(self.telemetry, extraction=self.extraction_cache,
                              semantic=self.semantic_cache)
        errors: list[dict[str, str]] = []
        seen: set[int] = set()
        for name, cache in (("extraction_cache", self.extraction_cache),
                            ("semantic_cache", self.semantic_cache)):
            if cache is None or id(cache) in seen:
                continue
            seen.add(id(cache))
            try:
                cache.save()
            except Exception as exc:                               # noqa: BLE001
                errors.append(self._error(name, exc))
        return errors

    @staticmethod
    def _error(phase: str, exc: BaseException) -> dict[str, str]:
        return {"phase": phase, "error": f"{type(exc).__name__}: {exc}"}

    def _already_finalised(self):
        if self.state == FINALISATION_FAILED and self._failure is not None:
            raise self._failure
        return self._artifact

    def _fail_success(self, message: str, errors: list[dict[str, str]],
                      exc: BaseException | None = None):
        """Record a failed success-finalisation and raise one stable error."""
        self.state = FINALISATION_FAILED
        self._write_outcome_sidecar(
            status=FINALISATION_FAILED, stage="", reason=message,
            requested_outcome=COMPLETE, errors=errors)
        failure = FinalisationFailed(message)
        if exc is not None:
            failure.__cause__ = exc
        self._failure = failure
        raise failure

    # --- the four ways out ---------------------------------------------------

    def finalise_success(self, package):
        """Validate an unpublished candidate, then atomically make it final."""
        if self.state in TERMINAL_STATES:
            return self._already_finalised()
        if self.state != RUNNING:
            raise FinalisationFailed(f"cannot finalise success from {self.state}")
        self.state = FINALISING
        errors = self._close_books()
        if errors:
            detail = "; ".join(e["error"] for e in errors)
            self._fail_success(f"could not save the run caches: {detail}", errors)

        manifest = self.manifest or getattr(package, "run_manifest", None)
        if manifest is not None:
            manifest = manifest.model_copy(deep=True)
            if self.execution is None:
                error = RuntimeError("ProductionSession has no ExecutionMode")
                self._fail_success(str(error), [self._error("manifest", error)], error)
            try:
                manifest.finalise(self.execution)
            except Exception as exc:                               # noqa: BLE001
                self._fail_success(
                    f"could not finalise the run manifest: {exc}",
                    [self._error("manifest", exc)], exc)

        final = package.model_copy(update={"telemetry": self.telemetry,
                                           "run_manifest": manifest})
        try:
            candidate = self.store.save_finalizing(final)
            reloaded = self.store.load(self.run_id, path=candidate)
        except Exception as exc:                                   # noqa: BLE001
            path = getattr(self.store, "finalizing_path", lambda _: "candidate")(
                self.run_id)
            self._fail_success(
                f"the package candidate at {path} could not be read back after "
                f"writing: {exc}. No package was published as complete",
                [self._error("package_validation", exc)], exc)
        try:
            self.store.publish_finalizing(self.run_id)
        except Exception as exc:                                   # noqa: BLE001
            self._fail_success(
                f"the validated package could not be published: {exc}",
                [self._error("package_publish", exc)], exc)

        cleanup_errors = self._retire_partial()
        if cleanup_errors:
            detail = "; ".join(e["error"] for e in cleanup_errors)
            self._fail_success(
                f"the package is complete, but its progress artifact could not "
                f"be retired: {detail}", cleanup_errors)
        self.state, self.outcome = FINALISED_SUCCESS, COMPLETE
        self._artifact = reloaded
        return reloaded

    def finalise_stopped(self, *, stage: str, reason: str):
        return self._finalise_early(STOPPED_BY_USER, stage=stage, reason=reason)

    def finalise_interrupted(self, *, reason: str = "interrupted (Ctrl-C)"):
        return self._finalise_early(INTERRUPTED, stage="", reason=reason)

    def finalise_error(self, exc: BaseException, *, stage: str = ""):
        """A run that died of an exception is evidence too, and says so.

        It used to leave a partial marked `in_progress`, which reads as a run
        still going — the one thing it certainly is not.

        ``stage`` is optional because most crashes cannot say where they were:
        an arbitrary exception carries a traceback, not a pipeline stage. A
        caller that DOES know — a failure raised by a named stage — passes it,
        so the artifact locates the failure instead of only reporting it.
        """
        return self._finalise_early(
            ERROR, stage=stage, reason=f"{type(exc).__name__}: {exc}")

    def _finalise_early(self, status: str, *, stage: str, reason: str):
        """Leave an artifact that says how the run ended — always one.

        Including when nothing had been collected yet. A partial written only
        after the first study meant an interrupt during parsing left a run
        directory with no package in it, and no way to tell that from a run that
        was never started.
        """
        if self.state in TERMINAL_STATES:
            return self._artifact
        if self.state != RUNNING:
            return self._artifact
        self.state = FINALISING
        errors = self._close_books()
        run_dir = self.store.run_dir(self.run_id)
        partial = run_dir / "package.partial.json"
        existed = partial.is_file()
        if existed:
            try:
                self._artifact = self._patch_partial(
                    partial, status=status, stage=stage, reason=reason,
                    errors=errors)
            except Exception as exc:                               # noqa: BLE001
                errors.append(self._error("partial_patch", exc))
                self._artifact = self._write_outcome_sidecar(
                    status=status, stage=stage, reason=reason,
                    requested_outcome=status, errors=errors)
        else:
            from react_review.schemas.package import EvidencePackage

            try:
                package = EvidencePackage(
                    run_id=self.run_id, run_manifest=self.manifest, status=status,
                    stopped_at_stage=stage, stop_reason=reason,
                    telemetry=self.telemetry, finalisation_errors=errors)
                self.store.save_partial(package)
                self._artifact = package
            except Exception as exc:                               # noqa: BLE001
                errors.append(self._error("partial_write", exc))
                self._artifact = self._write_outcome_sidecar(
                    status=status, stage=stage, reason=reason,
                    requested_outcome=status, errors=errors)
        self.state = FINALISATION_FAILED if errors else FINALISED_PARTIAL
        self.outcome = status
        self.emit(f"\n[{status}] {reason}")
        self.emit(f"[ARTIFACTS] {run_dir.resolve()}")
        if existed:
            self.emit(f"[PARTIAL]   {partial.resolve()} — evidence collected so far")
        for failure in errors:
            self.emit(f"[FINALISATION FAILED] {failure['phase']}: "
                      f"{failure['error']}")
        return self._artifact

    def _patch_partial(self, path, *, status: str, stage: str, reason: str,
                       errors: list[dict[str, str]]):
        """Add how it ended to the evidence already on disk, without risking it.

        Patched as JSON rather than rebuilt through the model, because the
        evidence in that file is the only copy: a package written by an older
        version that no longer validates would, on the rebuild path, be replaced
        by a minimal one — losing the collected evidence in order to record that
        collection had stopped.
        """
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        data["status"], data["stopped_at_stage"] = status, stage
        data["stop_reason"] = reason
        data["telemetry"] = self.telemetry.model_dump(mode="json")
        if errors:
            data["finalisation_errors"] = errors
        self._atomic_json(path, data)
        return path

    def _write_outcome_sidecar(self, *, status: str, stage: str, reason: str,
                               requested_outcome: str,
                               errors: list[dict[str, str]]):
        """Preserve an outcome when the sole partial cannot safely be touched."""
        path = self.store.run_dir(self.run_id) / "run.outcome.json"
        data = {
            "run_id": self.run_id,
            "status": status,
            "requested_outcome": requested_outcome,
            "stopped_at_stage": stage,
            "stop_reason": reason,
            "telemetry": self.telemetry.model_dump(mode="json"),
            "run_manifest": (self.manifest.model_dump(mode="json")
                             if self.manifest is not None else None),
            "finalisation_errors": errors,
        }
        try:
            self._atomic_json(path, data)
            return path
        except Exception as exc:                                   # noqa: BLE001
            self.emit(f"[FINALISATION FAILED] outcome sidecar: "
                      f"{type(exc).__name__}: {exc}")
            return None

    def _retire_partial(self) -> list[dict[str, str]]:
        """Remove stale progress only after a final package is authoritative."""
        partial = self.store.run_dir(self.run_id) / "package.partial.json"
        if not partial.is_file():
            return []
        try:
            partial.unlink()
            return []
        except Exception as unlink_exc:                            # noqa: BLE001
            archived = partial.with_name("package.partial.superseded.json")
            try:
                os.replace(partial, archived)
                return []
            except Exception as replace_exc:                       # noqa: BLE001
                return [self._error("partial_cleanup", unlink_exc),
                        self._error("partial_archive", replace_exc)]

    @staticmethod
    def _atomic_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)
