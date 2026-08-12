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
            "evaluator_version": "1.6.1"}
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

def test_the_cli_hands_the_collector_the_whole_contract(tmp_path):
    """Not one field of it.

    A run may read values in batch and arm identities one at a time, and the
    Collector can only honour that if it holds the routes. Passing a single
    profile would let a v2 contract be loaded, recorded, and then ignored — and
    every test that builds a Collector directly would still pass.
    """
    import inspect

    from react_review import cli

    source = inspect.getsource(cli)
    start = source.index("pipeline = AuditPipeline(")
    block = source[start:start + 600]
    assert "contract=contract" in block
    assert "aggregation_runtime=runtime" in block
    assert "knowledge_fingerprint=" in block


def test_the_cli_loads_its_contract_before_anything_reads_it():
    """It used to be read thirty lines before it was assigned.

    Any run without an explicit --context raised UnboundLocalError before it
    reached a paper.
    """
    import inspect

    from react_review import cli

    source = inspect.getsource(cli)
    assert source.index("contract = load_run_contract(") < \
        source.index("contract.context_policy")


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
    from react_review.schemas.run_manifest import RunManifest
    from react_review.tools.safe_aggregation import AggregationRuntime

    runtime = AggregationRuntime.resolve(policy_id="safe_sum_v5",
                                         evaluator_version="1.6.1")
    body = RunManifest.runtime_of(runtime)
    assert body["policy_id"] == "safe_sum_v5"
    assert body["evaluator_version"] == "1.6.1"
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
