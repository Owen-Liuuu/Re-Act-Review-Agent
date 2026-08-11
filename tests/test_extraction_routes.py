"""A run may read values in batch and arm identities one at a time.

What it may not do is that while declaring a single profile. A recording whose
`extraction_profile` says `targeted_v5_batch` and whose arm-identity claims were
read under `targeted_v4` cannot be interpreted by anyone later: half of it was
produced by a contract the artifact never names.

So routing is a declared fact. v1 contracts name one profile and mean it for
everything — expanded, not reinterpreted, so every frozen benchmark keeps its
meaning and its bytes. v2 names each kind and must name all of them.
"""
from __future__ import annotations

import json

import pytest

from react_review.contracts import ContractError
from react_review.run_profile import CLAIM_KINDS, load_run_contract

BASE = {
    "schema_version": 2,
    "profile_id": "routed",
    "semantic_prompt_profile": "semantic_v1",
    "context_policy": "cli_only",
    "scope_policy": "off",
    "extraction_routes": {"value": "targeted_v5_batch",
                          "arm_identity": "targeted_v4"},
    "aggregation_policy_id": "safe_sum_v5",
    "evaluator_version": "1.6.0",
}


def _write(tmp_path, **changes):
    body = {**BASE, **changes}
    for key, value in changes.items():
        if value is None:
            body.pop(key, None)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# --- v1 keeps meaning exactly what it always meant -------------------------

def test_a_v1_contract_routes_every_kind_to_its_one_profile(tmp_path):
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({
        "schema_version": 1, "profile_id": "old",
        "extraction_profile": "targeted_v4",
        "semantic_prompt_profile": "semantic_v1",
        "context_policy": "cli_only", "scope_policy": "off"}), encoding="utf-8")
    contract = load_run_contract(path)
    assert contract.extraction_profile == "targeted_v4"
    assert all(contract.route_for(kind) == "targeted_v4" for kind in CLAIM_KINDS)
    assert not contract.batching


def test_a_v1_contract_records_no_routes_in_its_identity(tmp_path):
    """A manifest already written must not gain a key for a fact it never had."""
    path = tmp_path / "v1.json"
    path.write_text(json.dumps({
        "schema_version": 1, "profile_id": "old",
        "extraction_profile": "legacy_v3",
        "semantic_prompt_profile": "semantic_v1",
        "context_policy": "cli_only", "scope_policy": "off"}), encoding="utf-8")
    identity = load_run_contract(path).identity()
    assert identity["extraction_profile"] == "legacy_v3"
    assert "extraction_routes" not in identity
    assert "evaluator_version" not in identity


def test_a_v1_contract_may_not_also_declare_routes(tmp_path):
    path = tmp_path / "both.json"
    path.write_text(json.dumps({
        "schema_version": 1, "profile_id": "both",
        "extraction_profile": "legacy_v3",
        "extraction_routes": {"value": "targeted_v4"},
        "semantic_prompt_profile": "semantic_v1",
        "context_policy": "cli_only", "scope_policy": "off"}), encoding="utf-8")
    with pytest.raises(ContractError, match="two places saying which is in force"):
        load_run_contract(path)


# --- v2 routes, and must route everything ----------------------------------

def test_a_v2_contract_may_route_two_kinds_differently(tmp_path):
    contract = load_run_contract(_write(tmp_path))
    assert contract.route_for("value") == "targeted_v5_batch"
    assert contract.route_for("arm_identity") == "targeted_v4"
    assert contract.batching
    assert contract.extraction_profile == "targeted_v5_batch"


def test_the_identity_of_a_routed_contract_records_every_route(tmp_path):
    identity = load_run_contract(_write(tmp_path)).identity()
    assert identity["extraction_routes"] == {
        "arm_identity": "targeted_v4", "value": "targeted_v5_batch"}
    assert identity["aggregation_policy_id"] == "safe_sum_v5"
    assert identity["evaluator_version"] == "1.6.0"
    # The v1 key keeps its place and its meaning, so nothing that reads it moves.
    assert identity["extraction_profile"] == "targeted_v5_batch"


def test_a_kind_nobody_routed_is_refused(tmp_path):
    path = _write(tmp_path, extraction_routes={"value": "targeted_v5_batch"})
    with pytest.raises(ContractError, match="arm_identity"):
        load_run_contract(path)


def test_a_route_for_something_nothing_dispatches_on_is_refused(tmp_path):
    path = _write(tmp_path, extraction_routes={**BASE["extraction_routes"],
                                               "vibes": "targeted_v4"})
    with pytest.raises(ContractError, match="nothing dispatches on"):
        load_run_contract(path)


def test_a_v2_contract_may_not_keep_the_single_profile(tmp_path):
    path = _write(tmp_path, extraction_profile="targeted_v4")
    with pytest.raises(ContractError, match="second source of truth"):
        load_run_contract(path)


def test_asking_for_a_kind_the_contract_never_routed_is_an_error(tmp_path):
    contract = load_run_contract(_write(tmp_path))
    with pytest.raises(ContractError, match="declares no extraction route"):
        contract.route_for("something_else")


# --- batching costs an identity --------------------------------------------

def test_a_batching_contract_must_name_its_policy_and_evaluator(tmp_path):
    """A batched read can derive a total, and a derived total is worth exactly
    as much as the identity that cleared the rules it was derived under."""
    with pytest.raises(ContractError, match="aggregation_policy_id"):
        load_run_contract(_write(tmp_path, aggregation_policy_id=None))
    with pytest.raises(ContractError, match="aggregation_policy_id"):
        load_run_contract(_write(tmp_path, evaluator_version=None))


def test_naming_an_evaluator_nothing_uses_is_refused(tmp_path):
    """It would attribute the run to code that never ran."""
    path = _write(tmp_path,
                  extraction_routes={"value": "targeted_v4",
                                     "arm_identity": "targeted_v4"})
    with pytest.raises(ContractError, match="routes nothing to a batch"):
        load_run_contract(path)


# --- the second gate --------------------------------------------------------

def test_a_v5_claim_reaching_the_single_target_extractor_crashes():
    """Not a warning, and not a silent legacy read.

    Only `targeted_v4` turns the targeted prompt sections on, so a v5 request
    landing here would be built with the LEGACY body and cached under the v5
    prompt version — neither contract, written into the namespace of the one it
    is not. The startup gate should make this unreachable; this is what turns a
    hole in that gate into a crash instead of a poisoned recording.
    """
    import asyncio

    from react_review.tools.extract_source import (
        ExtractSourceValueInput,
        ExtractSourceValueTool,
    )

    from react_review.steps.data_extraction.schemas import PaperDocument
    from react_review.steps.paper_verification.schemas import ReferenceEntry

    tool = ExtractSourceValueTool(None, cache_mode="replay", cache=_NullCache())
    payload = ExtractSourceValueInput(
        document=PaperDocument(
            paper_id="p1", full_text="some paper text",
            reference=ReferenceEntry(study_id="s1", title="A trial")),
        field_type="cohort_n", group="a",
        extraction_profile="targeted_v5_batch")
    with pytest.raises(ContractError, match="neither contract"):
        asyncio.run(tool.run(payload))


class _NullCache:
    model_id = "test"

    def get(self, key):
        return None

    def put(self, key, data, *, model_id=""):
        raise AssertionError("nothing may be written for a refused route")
