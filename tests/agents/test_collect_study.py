"""One paper, one pass, and each claim under the route its contract names.

A mixed contract is the normal case rather than an edge. Values are worth
batching because one reading answers several; arm identities are not, because
there is one per arm and "what is this arm called" needs a different prompt from
"what does it report". Both happen in one pass, and every claim records which
route actually read it — a run-level profile would describe half the run.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from react_review.agents.collector import Collector
from react_review.contracts import ContractError
from react_review.llm.base import LLMBackend
from react_review.run_profile import load_run_contract
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.base import Tool, ToolStage
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry

REFERENCE = ReferenceEntry(study_id="larkin", title="A trial", doi="10.1/x")
PAPER = ("A total of 945 patients underwent randomization: 316 patients were "
         "assigned to the nivolumab group, 314 to the nivolumab-plus-ipilimumab "
         "group, and 315 to the ipilimumab group.")


class _Fetch(Tool):
    name = "fetch_fulltext"
    stage = ToolStage.EXTRACT

    async def run(self, reference):
        class _Fetched:
            retrieved = True
            document = PaperDocument(paper_id="p1", reference=reference,
                                     full_text=PAPER)
        return _Fetched()


class _Backend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return json.dumps({
            "found": True, "value": "316", "unit": "",
            "quote": PAPER, "cohorts_seen": ["nivolumab group"],
            "group_label_in_paper": "nivolumab group",
            "arms_reported": [{"label": "nivolumab group", "value": "316",
                               "unit": "", "quote": PAPER}]})


def _registry(backend):
    registry = ToolRegistry()
    registry.register(_Fetch())
    registry.register(ExtractSourceValueTool(backend))
    return registry


def _contract(tmp_path, value_route, identity_route="targeted_v4", **extra):
    body = {"schema_version": 2, "profile_id": "routed",
            "semantic_prompt_profile": "semantic_v1",
            "context_policy": "cli_only", "scope_policy": "off",
            "extraction_routes": {"value": value_route,
                                  "arm_identity": identity_route}}
    if value_route == "targeted_v5_batch" or identity_route == "targeted_v5_batch":
        body["aggregation_policy_id"] = "safe_sum_v5"
        body["evaluator_version"] = "1.6.1"
    body.update(extra)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return load_run_contract(path)


def _claims():
    """Two values and one arm identity — a mixed contract's normal input."""
    return [
        ReviewDataItem(study_id="larkin", group="nivolumab_plus_placebo",
                       field_type="cohort_n", raw_field_name="Arm, n",
                       cohort_label="Nivolumab (3 mg/kg) + placebo", value="316"),
        ReviewDataItem(study_id="larkin", group="ipilimumab_plus_placebo",
                       field_type="cohort_n", raw_field_name="Arm, n",
                       cohort_label="Ipilimumab (3 mg/kg) + placebo", value="315"),
        ReviewDataItem(study_id="larkin", group="nivolumab_plus_placebo",
                       field_type="treatment_arm", raw_field_name="Arm",
                       cohort_label="Nivolumab (3 mg/kg) + placebo",
                       value="Nivolumab (3 mg/kg) + placebo"),
    ]


# --- the single-profile case keeps working --------------------------------

def test_a_v4_contract_reads_every_claim_one_at_a_time(tmp_path):
    backend = _Backend()
    collector = Collector(_registry(backend),
                          contract=_contract(tmp_path, "targeted_v4"))
    result = asyncio.run(collector.collect_study(_claims(), REFERENCE))
    assert len(result.claim_results) == 3
    assert len(backend.prompts) == 3
    assert result.batch_records == []


def test_v4_single_target_source_carries_the_review_claim_id(tmp_path):
    claims = [claim.model_copy(update={"review_data_id": f"A_0{i}"})
              for i, claim in enumerate(_claims(), start=1)]
    collector = Collector(_registry(_Backend()),
                          contract=_contract(tmp_path, "targeted_v4"))

    result = asyncio.run(collector.collect_study(claims, REFERENCE))

    assert [row.source_item.review_data_id for row in result.claim_results] == \
        ["A_01", "A_02", "A_03"]


def test_claims_come_back_in_the_order_they_arrived(tmp_path):
    """Grouping reorders them internally; nothing downstream should know."""
    claims = _claims()
    collector = Collector(_registry(_Backend()),
                          contract=_contract(tmp_path, "targeted_v4"))
    result = asyncio.run(collector.collect_study(claims, REFERENCE))
    assert [r.source_item.field_type for r in result.claim_results] == \
        [c.field_type for c in claims]
    assert [r.source_item.group for r in result.claim_results] == \
        [c.group for c in claims]


# --- the route is per claim, not per run ----------------------------------

def test_the_route_is_resolved_per_claim_kind(tmp_path):
    contract = _contract(tmp_path, "targeted_v5_batch")
    collector = Collector(_registry(_Backend()), contract=contract)
    values, identity = _claims()[0], _claims()[2]
    assert collector.route_for(values) == "targeted_v5_batch"
    assert collector.route_for(identity) == "targeted_v4"


def test_a_collector_with_no_contract_uses_its_single_profile(tmp_path):
    collector = Collector(_registry(_Backend()), extraction_profile="legacy_v3")
    assert collector.route_for(_claims()[0]) == "legacy_v3"


# --- fail closed ----------------------------------------------------------

def test_a_batched_route_with_no_batch_tool_stops_the_run(tmp_path):
    """Not a silent fall back to reading them one at a time.

    That would put half a run's answers under a profile the artifact does not
    name, and would make the cost of batching unmeasurable — every fallback
    adds back the calls the batch was supposed to save.
    """
    collector = Collector(_registry(_Backend()),
                          contract=_contract(tmp_path, "targeted_v5_batch"))
    with pytest.raises(ContractError, match="has no batch tool"):
        asyncio.run(collector.collect_study(_claims(), REFERENCE))


def test_the_arm_identity_claims_of_a_batched_contract_still_run(tmp_path):
    """The refusal is about the batched GROUP, not about the whole study."""
    collector = Collector(_registry(_Backend()),
                          contract=_contract(tmp_path, "targeted_v5_batch"))
    identity_only = [_claims()[2]]
    result = asyncio.run(collector.collect_study(identity_only, REFERENCE))
    assert len(result.claim_results) == 1


def test_a_contract_that_routes_a_kind_nowhere_never_loads(tmp_path):
    """Fail closed at the contract, before a Collector exists to guess."""
    body = {"schema_version": 2, "profile_id": "partial",
            "semantic_prompt_profile": "semantic_v1",
            "context_policy": "cli_only", "scope_policy": "off",
            "extraction_routes": {"value": "targeted_v4"}}
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="arm_identity"):
        load_run_contract(path)


# --- one paper, opened once -----------------------------------------------

def test_a_study_is_fetched_once_however_many_claims_it_has(tmp_path):
    class _CountingFetch(_Fetch):
        calls = 0

        async def run(self, reference):
            _CountingFetch.calls += 1
            return await super().run(reference)

    registry = ToolRegistry()
    registry.register(_CountingFetch())
    registry.register(ExtractSourceValueTool(_Backend()))
    collector = Collector(registry, contract=_contract(tmp_path, "targeted_v4"))
    asyncio.run(collector.collect_study(_claims(), REFERENCE))
    assert _CountingFetch.calls == 1


# --- the batch path, end to end -------------------------------------------

class _BatchBackend(LLMBackend):
    """Answers a v5 batch: every arm the paper reports, in one reply."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return json.dumps({"readings": [
            {"arm_label": "nivolumab group", "value": "316", "quote": PAPER,
             "population_phrase": "underwent randomization"},
            {"arm_label": "ipilimumab group", "value": "315", "quote": PAPER,
             "population_phrase": "underwent randomization"}]})


def _batched(tmp_path, backend=None):
    from react_review.tools.extract_batch import ExtractSourceBatchTool

    batch_backend = backend or _BatchBackend()
    registry = ToolRegistry()
    registry.register(_Fetch())
    registry.register(ExtractSourceValueTool(_Backend()))
    collector = Collector(
        registry, contract=_contract(tmp_path, "targeted_v5_batch"),
        batch_tool=ExtractSourceBatchTool(batch_backend),
        cohorts=_cohorts())
    return collector, batch_backend


def _cohorts():
    from react_review.normalize.cohorts import CohortLabel, CohortRegistry

    return CohortRegistry(labels=[
        CohortLabel(key="nivolumab_plus_placebo",
                    display="Nivolumab (3 mg/kg) + placebo"),
        CohortLabel(key="ipilimumab_plus_placebo",
                    display="Ipilimumab (3 mg/kg) + placebo")])


def test_two_value_claims_are_answered_by_one_reading(tmp_path):
    collector, backend = _batched(tmp_path)
    values = _claims()[:2]
    result = asyncio.run(collector.collect_study(values, REFERENCE))
    assert backend.calls == 1
    assert len(result.claim_results) == 2
    assert len(result.batch_records) == 1
    assert [r.source_item.source_value for r in result.claim_results] == \
        ["316", "315"]


def test_both_claims_point_at_the_same_reading(tmp_path):
    """The only way to SHOW two answers came from one act of reading."""
    collector, _ = _batched(tmp_path)
    result = asyncio.run(collector.collect_study(_claims()[:2], REFERENCE))
    executions = {r.source_item.batch_provenance.batch_execution_id
                  for r in result.claim_results}
    assert len(executions) == 1
    assert executions == {result.batch_records[0].execution_id}


def test_v5_batch_source_carries_one_consistent_claim_identity(tmp_path):
    collector, _ = _batched(tmp_path)
    claims = [claim.model_copy(update={"review_data_id": f"A_0{i}"})
              for i, claim in enumerate(_claims()[:2], start=1)]

    result = asyncio.run(collector.collect_study(claims, REFERENCE))

    for expected, row in zip(("A_01", "A_02"), result.claim_results):
        assert row.source_item.review_data_id == expected
        assert row.source_item.batch_provenance.claim_id == expected


def test_each_claim_records_the_route_that_actually_read_it(tmp_path):
    collector, _ = _batched(tmp_path)
    result = asyncio.run(collector.collect_study(_claims(), REFERENCE))
    routes = {r.source_item.field_type: r.source_item.batch_provenance
              for r in result.claim_results}
    assert routes["cohort_n"].route == "targeted_v5_batch"
    # The arm-identity claim went the other way, and says so by carrying no
    # batch provenance at all rather than a batch one it never had.
    assert routes["treatment_arm"] is None


def test_a_mixed_contract_issues_one_batch_and_one_single_call(tmp_path):
    collector, batch_backend = _batched(tmp_path)
    result = asyncio.run(collector.collect_study(_claims(), REFERENCE))
    assert batch_backend.calls == 1
    assert len(result.batch_records) == 1
    assert len(result.claim_results) == 3


def test_a_batch_that_never_arrives_is_not_a_silent_paper(tmp_path):
    """EXTRACTION_FAILED, not MISSING_SOURCE, which accuses the reviewer."""
    from react_review.core.enums import CollectionOutcome

    class _Broken(_BatchBackend):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.calls += 1
            raise TimeoutError("gateway")

    collector, _ = _batched(tmp_path, backend=_Broken())
    result = asyncio.run(collector.collect_study(_claims()[:2], REFERENCE))
    for claim in result.claim_results:
        assert claim.source_item.collection_outcome is \
            CollectionOutcome.EXTRACTION_FAILED


def test_a_claim_the_reading_never_covered_is_unresolved(tmp_path):
    from react_review.core.enums import CollectionOutcome

    class _OneArm(_BatchBackend):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.calls += 1
            return json.dumps({"readings": [
                {"arm_label": "nivolumab group", "value": "316", "quote": PAPER,
                 "population_phrase": "underwent randomization"}]})

    collector, _ = _batched(tmp_path, backend=_OneArm())
    result = asyncio.run(collector.collect_study(_claims()[:2], REFERENCE))
    outcomes = [r.source_item.collection_outcome for r in result.claim_results]
    assert outcomes[0] is CollectionOutcome.FOUND
    assert outcomes[1] is CollectionOutcome.EXTRACTION_UNRESOLVED


# --- two claims may not share an identity ---------------------------------

def test_two_claims_with_one_identity_are_refused(tmp_path):
    """Every reference in a batched artifact names a claim by its identity.

    Two claims with one identity make every one of those references ambiguous,
    in an artifact nobody can re-run to disambiguate. Refused here rather than
    resolved by a rule nobody agreed to.
    """
    twin = ReviewDataItem(study_id="larkin", group="a", field_type="cohort_n",
                          raw_field_name="Arm, n", value="316")
    collector = Collector(_registry(_Backend()),
                          contract=_contract(tmp_path, "targeted_v4"))
    with pytest.raises(ContractError, match="share the identity"):
        asyncio.run(collector.collect_study([twin, twin.model_copy()], REFERENCE))


def test_identical_claims_do_not_overwrite_each_others_positions(tmp_path):
    """`list.index` returns the first match for both, so one result replaced
    the other and a later read ran off the end."""
    from react_review.tools.batch_group import group_claims

    twin = ReviewDataItem(study_id="larkin", group="a", field_type="cohort_n",
                          raw_field_name="Arm, n", value="316")
    groups = group_claims([twin, twin.model_copy(), twin.model_copy()])
    assert groups[0].positions == [0, 1, 2]
