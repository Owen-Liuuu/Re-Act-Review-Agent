"""A batched run, end to end, and the references that must resolve.

The point of batching is that several answers come out of one act of reading.
That is a claim about an artifact, not about a design, so it has to be checkable
IN the artifact: every batched row names an execution id, every execution id
names a record, and the record says which claims it answered. A reference that
resolves nowhere would be worse than no reference — it would look like evidence.
"""
from __future__ import annotations

import json

import pytest

from tests.conftest import requires_frozen_evaluator

from react_review.agents.collector import Collector
from react_review.audit import ToleranceTable
from react_review.llm.base import LLMBackend
from react_review.normalize.cohorts import CohortLabel, CohortRegistry
from react_review.orchestrator import AuditOrchestrator, AuditPipeline, Judge
from react_review.run_profile import load_run_contract
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.compare import CompareValuesTool
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_batch import ExtractSourceBatchTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry

PAPER = ("A total of 945 patients underwent randomization: 316 patients were "
         "assigned to the nivolumab group, 314 to the nivolumab-plus-ipilimumab "
         "group, and 315 to the ipilimumab group.")
REFERENCE = ReferenceEntry(study_id="larkin", title="A trial", doi="10.1/x")


class _Retriever(PaperRetriever):
    async def retrieve(self, reference):
        return PaperDocument(paper_id="p1", reference=reference, full_text=PAPER)


class _Batch(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "batch-stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return json.dumps({"readings": [
            {"arm_label": "nivolumab group", "value": "316", "quote": PAPER,
             "population_phrase": "underwent randomization"},
            {"arm_label": "ipilimumab group", "value": "315", "quote": PAPER,
             "population_phrase": "underwent randomization"}]})


class _Single(LLMBackend):
    @property
    def model_id(self) -> str:
        return "single-stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        return json.dumps({"found": False, "value": None,
                           "not_found_reason": "not asked for here"})


def _contract(tmp_path):
    body = {"schema_version": 2, "profile_id": "batched",
            "semantic_prompt_profile": "semantic_v1",
            "context_policy": "cli_only", "scope_policy": "off",
            "extraction_routes": {"value": "targeted_v5_batch",
                                  "arm_identity": "targeted_v4"},
            "aggregation_policy_id": "safe_sum_v5",
            "evaluator_version": "1.7.0"}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return load_run_contract(path)


def _pipeline(tmp_path, batch_backend):
    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(ExtractSourceValueTool(_Single()))
    registry.register(ExtractSourceBatchTool(batch_backend))
    registry.register(CompareValuesTool(ToleranceTable()))
    collector = Collector(
        registry, contract=_contract(tmp_path),
        cohorts=CohortRegistry(labels=[
            CohortLabel(key="nivolumab_plus_placebo",
                        display="Nivolumab (3 mg/kg) + placebo"),
            CohortLabel(key="ipilimumab_plus_placebo",
                        display="Ipilimumab (3 mg/kg) + placebo")]))
    return AuditPipeline(collector, AuditOrchestrator(registry), Judge())


def _claims():
    return [
        ReviewDataItem(review_data_id="r1", study_id="larkin",
                       group="nivolumab_plus_placebo", field_type="cohort_n",
                       raw_field_name="Arm, n", value="316",
                       cohort_label="Nivolumab (3 mg/kg) + placebo"),
        ReviewDataItem(review_data_id="r2", study_id="larkin",
                       group="ipilimumab_plus_placebo", field_type="cohort_n",
                       raw_field_name="Arm, n", value="315",
                       cohort_label="Ipilimumab (3 mg/kg) + placebo"),
    ]


@pytest.fixture
def package(tmp_path):
    backend = _Batch()
    pipeline = _pipeline(tmp_path, backend)
    result = pytest.importorskip("asyncio").run(
        pipeline.run(_claims(), lambda _: REFERENCE))
    result_backend_calls = backend.calls
    return result, result_backend_calls


# --- one reading, two answers ---------------------------------------------

def test_two_claims_cost_one_reading(package):
    evidence, calls = package
    assert calls == 1
    assert len(evidence.source_items) == 2
    assert len(evidence.batch_records) == 1


def test_every_row_names_the_reading_it_came_from(package):
    evidence, _ = package
    known = {record.execution_id for record in evidence.batch_records}
    for item in evidence.source_items:
        provenance = item.batch_provenance
        assert provenance is not None
        assert provenance.batch_execution_id in known


def test_the_reading_names_the_claims_it_answered(package):
    """Resolvable in both directions, or one of them is decoration."""
    evidence, _ = package
    record = evidence.batch_records[0]
    assert sorted(record.claim_ids) == ["r1", "r2"]
    named = {item.batch_provenance.claim_id for item in evidence.source_items}
    assert named == set(record.claim_ids)


def test_the_reading_is_kept_once_not_per_claim(package):
    evidence, _ = package
    assert len(evidence.batch_records) == 1
    assert evidence.batch_records[0].usable_readings == 2


def test_the_package_serialises_with_its_readings(package):
    evidence, _ = package
    body = evidence.model_dump(mode="json")
    assert body["batch_records"][0]["claim_ids"] == ["r1", "r2"]
    assert body["source_items"][0]["batch_provenance"]["route"] == \
        "targeted_v5_batch"


def test_the_values_are_the_ones_the_reading_reported(package):
    evidence, _ = package
    assert [item.source_value for item in evidence.source_items] == ["316", "315"]


# --- a run that could not read ---------------------------------------------

def test_a_failed_reading_still_records_the_attempt(tmp_path):
    import asyncio

    class _Broken(_Batch):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.calls += 1
            raise TimeoutError("gateway")

    backend = _Broken()
    pipeline = _pipeline(tmp_path, backend)
    evidence = asyncio.run(pipeline.run(_claims(), lambda _: REFERENCE))
    assert len(evidence.batch_records) == 1
    record = evidence.batch_records[0]
    assert record.failure == "transport" and record.attempts == 3
    # And the claims say what happened, rather than that the paper is silent.
    outcomes = {item.collection_outcome.value for item in evidence.source_items}
    assert outcomes == {"extraction_failed"}


# --- the eval harness reads per study and restores the answer key's order ---

def _rows():
    """Deliberately interleaved, so grouping MUST reorder them internally."""
    return [
        {"audit_id": "a1", "study_id": "larkin", "group": "nivolumab_plus_placebo",
         "field_type": "cohort_n", "review_value": "316", "source_value": "316"},
        {"audit_id": "a2", "study_id": "other", "group": "nivolumab_plus_placebo",
         "field_type": "cohort_n", "review_value": "10", "source_value": "10"},
        {"audit_id": "a3", "study_id": "larkin", "group": "ipilimumab_plus_placebo",
         "field_type": "cohort_n", "review_value": "315", "source_value": "315"},
    ]


def test_run_rows_returns_rows_in_the_answer_keys_order(tmp_path):
    import asyncio

    from react_review.eval_accuracy import run_rows

    pipeline_backend = _Batch()
    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(ExtractSourceValueTool(_Single()))
    registry.register(ExtractSourceBatchTool(pipeline_backend))
    collector = Collector(
        registry, contract=_contract(tmp_path),
        cohorts=CohortRegistry(labels=[
            CohortLabel(key="nivolumab_plus_placebo",
                        display="Nivolumab (3 mg/kg) + placebo"),
            CohortLabel(key="ipilimumab_plus_placebo",
                        display="Ipilimumab (3 mg/kg) + placebo")]))
    results = asyncio.run(run_rows(
        _rows(), collector, ToleranceTable(), lambda _: REFERENCE))
    assert [r.audit_id for r in results] == ["a1", "a2", "a3"]


def test_run_rows_carries_the_readings_beside_the_rows(tmp_path):
    """A row names an execution id; this is where that reference resolves."""
    import asyncio

    from react_review.eval_accuracy import run_rows

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(ExtractSourceValueTool(_Single()))
    registry.register(ExtractSourceBatchTool(_Batch()))
    collector = Collector(
        registry, contract=_contract(tmp_path),
        cohorts=CohortRegistry(labels=[
            CohortLabel(key="nivolumab_plus_placebo",
                        display="Nivolumab (3 mg/kg) + placebo"),
            CohortLabel(key="ipilimumab_plus_placebo",
                        display="Ipilimumab (3 mg/kg) + placebo")]))
    results = asyncio.run(run_rows(
        _rows(), collector, ToleranceTable(), lambda _: REFERENCE))
    known = {reading.execution_id for reading in results.batch_readings}
    referenced = {r.batch_execution_id for r in results if r.batch_execution_id}
    assert referenced and referenced <= known


def test_a_row_that_was_never_batched_names_no_reading(tmp_path):
    """The single-target path leaves the field empty rather than borrowing one."""
    import asyncio

    from react_review.eval_accuracy import run_rows

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(ExtractSourceValueTool(_Single()))
    collector = Collector(registry, extraction_profile="targeted_v4")
    results = asyncio.run(run_rows(
        _rows(), collector, ToleranceTable(), lambda _: REFERENCE))
    assert all(r.batch_execution_id == "" for r in results)
    assert not getattr(results, "batch_readings", [])


# --- what the PRODUCTION entry point builds --------------------------------

def test_the_production_builder_is_what_the_cli_calls():
    """The assembly is one function, and this is that function.

    The previous version of this test read `cli.py` as text and asserted that
    the argument names appeared. It passed while the telemetry was created after
    the parsing it was supposed to measure, and it broke the moment the wiring
    moved into a builder — which is the drift it was supposed to catch and
    structurally could not.
    """
    import inspect

    from react_review import cli, production

    source = inspect.getsource(cli)
    assert "build_collector(" in source
    assert "ProductionBackends(" in source
    assert "ProductionSession(" in source
    # And the builder is not a second copy of the wiring.
    assert "Collector(" in inspect.getsource(production.build_collector)


def test_the_cli_does_not_let_the_pipeline_save_the_finished_package():
    """The authority to write package.json belongs to ONE object.

    While it was the Pipeline's, the file was written before the run had
    finished spending — so the telemetry inside it was a snapshot from before
    the last paper, and the fix was to save a second time over the same
    filename. Two writes, two different files' worth of bytes, one name.
    """
    import inspect

    from react_review import cli

    source = inspect.getsource(cli._run_audit)
    assert "owns_final_save=False" in source
    assert "store.save(pkg)" not in source
    assert "session.finalise_success(" in source


def test_the_production_entry_point_calls_the_audit_it_actually_has():
    """Nothing else in the suite runs `react-review run`.

    It needs a review PDF and a model, so the whole production entry point is
    covered by no test at all — a signature and its one call site drifted apart
    twice while 1210 tests stayed green, and only a linter noticed. This binds
    the call as written to the function as defined, which is the part that
    silently rots.
    """
    import ast
    import inspect
    import textwrap

    from react_review import cli

    tree = ast.parse(textwrap.dedent(inspect.getsource(cli._run_main)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "_run_audit"]
    assert len(calls) == 1, "one production entry point, one call"

    call = calls[0]
    inspect.signature(cli._run_audit).bind(
        *["x"] * len(call.args),
        **{kw.arg: "x" for kw in call.keywords})


def test_the_builder_gives_the_collector_everything_its_contract_needs(tmp_path):
    requires_frozen_evaluator()
    from react_review.production import build_collector
    from react_review.schemas.telemetry import RunTelemetry

    contract = _contract(tmp_path)
    telemetry = RunTelemetry()
    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(ExtractSourceValueTool(_Single()))
    registry.register(ExtractSourceBatchTool(_Batch()))
    collector = build_collector(registry, contract=contract,
                                knowledge_fingerprint="KB1", telemetry=telemetry)
    assert collector.route_for(_claims()[0]) == "targeted_v5_batch"
    assert collector._runtime is not None
    assert collector._knowledge_fingerprint == "KB1"
    assert collector._telemetry is telemetry


def test_parsing_gets_its_own_stage(tmp_path):
    """The parser and the resolver call the model before a paper is opened.

    Folding those into extraction would attribute the cost of reading the
    REVIEW to the cost of reading the papers.
    """
    from react_review.production import ProductionBackends, ProductionStages
    from react_review.schemas.telemetry import REVIEW_PARSING, RunTelemetry

    telemetry = RunTelemetry()
    stages = ProductionStages.of(_contract(tmp_path))
    assert stages.parsing == REVIEW_PARSING
    backends = ProductionBackends(_Single(), telemetry, stages)
    assert backends.parsing is not backends.single


def test_a_single_route_run_labels_no_stage_at_all(tmp_path):
    """Its global counters already say what it cost, and labelling it would add
    a section to every artifact ever replayed."""
    from react_review.production import ProductionStages

    body = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))         if (tmp_path / "contract.json").exists() else None
    _contract(tmp_path)
    plain = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    plain["extraction_routes"] = {"value": "targeted_v4",
                                  "arm_identity": "targeted_v4"}
    del plain["aggregation_policy_id"], plain["evaluator_version"]
    (tmp_path / "plain.json").write_text(json.dumps(plain), encoding="utf-8")
    stages = ProductionStages.of(load_run_contract(tmp_path / "plain.json"))
    assert (stages.parsing, stages.single, stages.batch, stages.semantic) ==         ("", "", "", "")


class _Cache:
    def __init__(self, hits, misses):
        self.hits, self.misses = hits, misses

    def save(self):
        return None


def test_the_global_cache_book_is_the_sum_of_the_caches():
    """The globals are what every existing artifact reads."""
    from react_review.production import snapshot_cache_totals
    from react_review.schemas.telemetry import RunTelemetry

    telemetry = RunTelemetry()
    snapshot_cache_totals(telemetry, extraction=_Cache(7, 2), semantic=_Cache(3, 1))
    assert telemetry.cache_hits == 10 and telemetry.cache_misses == 3


def test_a_shared_cache_is_never_attributed_to_one_stage():
    """Its hits belong to whichever tool did the looking up, and only that tool
    knows.

    One extraction cache serves both routes. Adding its whole total to the
    single-target bucket at the end of the run reported three hits in a stage
    where one had happened, and left the batch stage understating its own — the
    exact comparison the buckets exist to make.
    """
    from react_review.production import snapshot_cache_totals
    from react_review.schemas.telemetry import (
        BATCH_EXTRACTION,
        SINGLE_EXTRACTION,
        RunTelemetry,
    )

    telemetry = RunTelemetry()
    telemetry.record_stage_cache(SINGLE_EXTRACTION, hits=1, misses=0)
    telemetry.record_stage_cache(BATCH_EXTRACTION, hits=1, misses=0)
    snapshot_cache_totals(telemetry, extraction=_Cache(2, 0))

    assert telemetry.stages[SINGLE_EXTRACTION].cache_hits == 1
    assert telemetry.stages[BATCH_EXTRACTION].cache_hits == 1
    assert telemetry.cache_hits == 2


def test_the_cache_book_is_a_snapshot_so_closing_twice_changes_nothing():
    """Finalisation is reachable four ways, and totals that add count twice."""
    from react_review.production import snapshot_cache_totals
    from react_review.schemas.telemetry import RunTelemetry

    telemetry = RunTelemetry()
    for _ in range(3):
        snapshot_cache_totals(telemetry, extraction=_Cache(7, 2))
    assert (telemetry.cache_hits, telemetry.cache_misses) == (7, 2)


def test_a_run_with_no_cache_activity_creates_no_stage():
    """0/0 is not a measurement, and a bucket of zeroes is a section in every
    artifact for a fact nobody has."""
    from react_review.production import snapshot_cache_totals
    from react_review.schemas.telemetry import RunTelemetry

    telemetry = RunTelemetry()
    snapshot_cache_totals(telemetry, extraction=_Cache(0, 0))
    assert telemetry.stages is None
    assert "stages" not in telemetry.model_dump(mode="json")


def test_the_semantic_cache_records_its_own_lookups(tmp_path):
    """Where the lookup happens, not inferred from a total afterwards."""
    from react_review.audit.semantic_cache import SemanticCache
    from react_review.schemas.semantic import SemanticVerdict
    from react_review.schemas.telemetry import SEMANTIC, RunTelemetry

    telemetry = RunTelemetry()
    cache = SemanticCache(tmp_path / "semantic.json")
    cache.measure_into(telemetry, SEMANTIC)
    assert cache.get("k") is None                                   # a miss
    cache.put("k", SemanticVerdict(relation="same", equivalent=True,
                                   confidence=0.9))
    assert cache.get("k") is not None                               # a hit

    assert telemetry.stages[SEMANTIC].cache_hits == 1
    assert telemetry.stages[SEMANTIC].cache_misses == 1


def test_the_cli_resolves_a_runtime_only_for_a_batching_contract(tmp_path):
    from react_review.cli import _aggregation_runtime

    assert _aggregation_runtime(None) is None
    single = _contract(tmp_path)
    assert single.batching
    body = json.loads((tmp_path / "contract.json").read_text(encoding="utf-8"))
    body["extraction_routes"] = {"value": "targeted_v4",
                                 "arm_identity": "targeted_v4"}
    del body["aggregation_policy_id"], body["evaluator_version"]
    (tmp_path / "plain.json").write_text(json.dumps(body), encoding="utf-8")
    plain = load_run_contract(tmp_path / "plain.json")
    assert _aggregation_runtime(plain) is None


def test_the_manifest_records_the_runtime_that_ran(tmp_path):
    requires_frozen_evaluator()
    from react_review.schemas.run_manifest import RunManifest
    from react_review.tools.safe_aggregation import AggregationRuntime

    runtime = AggregationRuntime.resolve(policy_id="safe_sum_v5",
                                         evaluator_version="1.7.0")
    body = RunManifest.runtime_of(runtime)
    assert body["policy_id"] == "safe_sum_v5"
    assert body["evaluator_version"] == "1.7.0"
    assert len(body["policy_sha256"]) == 64
    assert "release_eligible" in body
    # And a run that never aggregated records nothing at all.
    assert RunManifest.runtime_of(None) == {}


def test_the_semantic_stage_is_measured_by_the_semantic_wrapper():
    """Both wrap the same backend; the label is the whole point of having two."""
    import inspect
    import pathlib

    source = pathlib.Path("eval/run_full_accuracy.py").read_text(encoding="utf-8")
    start = source.index("semantic=(SemanticCompareTool(")
    assert "semantic_backend" in source[start:start + 120]


def test_a_retried_batch_is_counted_as_a_repeated_attempt(tmp_path):
    """A batch that failed twice must not look as cheap as one that did not."""
    import asyncio

    from react_review.schemas.telemetry import RunTelemetry
    from react_review.tools.extract_batch import ExtractSourceBatchTool
    from react_review.schemas.batch import ARM, BatchQuestionId

    class _FlakyThenFine(_Batch):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("gateway")
            return await super().complete(prompt, seed=seed)

    telemetry = RunTelemetry()
    tool = ExtractSourceBatchTool(_FlakyThenFine(), telemetry=telemetry)
    record = asyncio.run(tool.read(
        question=BatchQuestionId(study_id="larkin", target_shape=ARM),
        prompt="PROMPT", document=PAPER))
    assert record.usable and len(record.attempts) == 2
    assert telemetry.repeated_attempts == 1


def test_a_batched_run_reports_what_batching_bought(tmp_path):
    import asyncio

    from react_review.schemas.telemetry import RunTelemetry

    telemetry = RunTelemetry()
    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(ExtractSourceValueTool(_Single()))
    registry.register(ExtractSourceBatchTool(_Batch()))
    collector = Collector(
        registry, contract=_contract(tmp_path), telemetry=telemetry,
        cohorts=CohortRegistry(labels=[
            CohortLabel(key="nivolumab_plus_placebo",
                        display="Nivolumab (3 mg/kg) + placebo"),
            CohortLabel(key="ipilimumab_plus_placebo",
                        display="Ipilimumab (3 mg/kg) + placebo")]))
    asyncio.run(collector.collect_study(_claims(), REFERENCE))
    stats = telemetry.batch
    assert stats.batches == 1 and stats.claims == 2
    assert stats.singleton_batches == 0
    assert stats.projections.get("ok") == 2


# --- the production construction chain, executed -------------------------

def test_the_production_pipeline_routes_records_and_measures(tmp_path):
    requires_frozen_evaluator()
    """Builds what the CLI builds, and RUNS it.

    The source-reading tests above prove the arguments are written down. This
    proves they do something: the contract is honoured per claim kind, the
    runtime reaches the manifest, the readings are persisted, and the run can
    say what it cost. Each of those was separately absent while the source
    still looked correct.
    """
    import asyncio

    from react_review.cli import _aggregation_runtime
    from react_review.llm.metered import MeteredBackend
    from react_review.schemas.run_manifest import RunManifest
    from react_review.schemas.telemetry import (
        BATCH_EXTRACTION,
        SINGLE_EXTRACTION,
        RunTelemetry,
    )

    contract = _contract(tmp_path)
    telemetry = RunTelemetry()
    batch_backend, single_backend = _Batch(), _Single()

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(ExtractSourceValueTool(
        MeteredBackend(single_backend, telemetry, SINGLE_EXTRACTION),
        telemetry=telemetry))
    registry.register(ExtractSourceBatchTool(
        MeteredBackend(batch_backend, telemetry, BATCH_EXTRACTION),
        telemetry=telemetry))
    registry.register(CompareValuesTool(ToleranceTable()))

    runtime = _aggregation_runtime(contract)
    assert runtime is not None, "a batching contract must resolve a runtime"

    collector = Collector(
        registry, contract=contract, aggregation_runtime=runtime,
        telemetry=telemetry,
        cohorts=CohortRegistry(labels=[
            CohortLabel(key="nivolumab_plus_placebo",
                        display="Nivolumab (3 mg/kg) + placebo"),
            CohortLabel(key="ipilimumab_plus_placebo",
                        display="Ipilimumab (3 mg/kg) + placebo")]))
    pipeline = AuditPipeline(collector, AuditOrchestrator(registry), Judge(),
                             telemetry=telemetry)

    claims = _claims() + [
        ReviewDataItem(review_data_id="r3", study_id="larkin",
                       group="nivolumab_plus_placebo", field_type="treatment_arm",
                       raw_field_name="Arm",
                       cohort_label="Nivolumab (3 mg/kg) + placebo",
                       value="Nivolumab (3 mg/kg) + placebo")]
    evidence = asyncio.run(pipeline.run(claims, lambda _: REFERENCE))

    # The routes were honoured: one batch for the two values, one single call
    # for the arm identity.
    assert batch_backend.calls == 1
    assert telemetry.stages[BATCH_EXTRACTION].requests == 1
    assert telemetry.stages[SINGLE_EXTRACTION].requests == 1

    # The readings are in the package and every reference resolves.
    known = {r.execution_id for r in evidence.batch_records}
    batched = [i for i in evidence.source_items if i.batch_provenance]
    assert len(batched) == 2
    assert all(i.batch_provenance.batch_execution_id in known for i in batched)

    # The arm identity went the other way and says so by carrying nothing.
    identity = [i for i in evidence.source_items if i.field_type == "treatment_arm"]
    assert identity and identity[0].batch_provenance is None

    # The run can say what it cost, and what batching bought.
    assert evidence.telemetry is not None
    assert evidence.telemetry.batch.batches == 1
    assert evidence.telemetry.batch.claims == 2
    assert evidence.telemetry.batch.claims_per_batch == 2.0

    # And what decided is recorded rather than inferred.
    body = RunManifest.runtime_of(runtime)
    assert body["policy_id"] == "safe_sum_v5"
    assert body["evaluator_version"] == "1.7.0"


def test_a_package_from_a_run_that_measured_nothing_has_no_telemetry_key(tmp_path):
    """Every package written before telemetry existed must stay unchanged."""
    import asyncio

    pipeline = _pipeline(tmp_path, _Batch())
    evidence = asyncio.run(pipeline.run(_claims(), lambda _: REFERENCE))
    assert "telemetry" not in evidence.model_dump(mode="json")
