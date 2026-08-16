"""The contract decides the answer; the execution mode decides only the cost.

These tests hold that line in both directions. A flag may not quietly move a
tolerance or a prompt version — a result nobody can attribute to a rule is not
evidence. And a contract may not fix the cache mode, because the same contract
has to be recorded once and replayed afterwards; binding the two together would
make reproducing a run impossible by construction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.contracts import ContractError, sha256_file
from react_review.run_profile import (
    ExecutionMode,
    RunManifest,
    guard_contract_overrides,
    legacy_contract,
    load_run_contract,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "configs" / "run_profiles"


# --- the shipped contracts ------------------------------------------------

def test_the_legacy_contract_describes_what_the_system_did_before():
    contract = load_run_contract(PROFILES / "legacy.json")
    assert contract.extraction_profile == "legacy_v3"
    assert contract.semantic_prompt_profile == "semantic_v1"
    assert contract.tolerances_path is None       # the comparator's own defaults
    assert contract.population_contract_path is None
    assert contract.context_policy == "cli_only"
    assert contract.scope_enabled is False


def test_the_phase8_contract_pins_its_tolerance_table():
    contract = load_run_contract(PROFILES / "phase8.json")
    assert contract.extraction_profile == "targeted_v4"
    assert contract.semantic_prompt_profile == "semantic_v2_specificity"
    assert contract.tolerances_path.name == "tolerances.phase8.yaml"
    assert contract.tolerances_sha256 == sha256_file(contract.tolerances_path)
    assert contract.scope_enabled is True
    assert contract.axes_for("cohort_n") == ["population_basis"]
    assert contract.axes_for("event_count") == ["population_basis", "analysis_set"]
    assert contract.axes_for("hazard_ratio") == []


def test_the_phase8_v8_contract_pins_the_evidence_gate_identity():
    contract = load_run_contract(PROFILES / "phase8_batch_v8.json")

    assert contract.schema_version == 4
    assert contract.adequacy_enabled is True
    assert contract.adequacy_policy_id == "evidence_adequacy_v1"
    assert contract.adequacy_policy_hash == (
        "9DA56A30430B6B2B78C4A051DA9DB620A559A6802E3DDBA606551C5ECCD42FC4")
    assert contract.adequacy_evaluator_id == "evidence_adequacy"
    assert contract.adequacy_evaluator_version == "1.0.0"
    assert contract.adequacy_evaluator_hash == (
        "sha256:4c04abf8b32e959b74d9dd6e0100c5c49c10ba09a31f727a1f39c3c1266e1931")
    assert contract.identity()["adequacy_evaluator_version"] == "1.0.0"


def test_the_v9_contract_moves_the_single_claim_route_and_nothing_else():
    """v9's whole claim is that only the arm-identity prompt changed.

    Asserted as a diff against v8 rather than by restating v9's fields: a test
    that listed them would pass while some other axis had silently moved too,
    and a run whose result differed would then have two candidate causes.
    """
    v8 = json.loads((PROFILES / "phase8_batch_v8.json").read_text(
        encoding="utf-8-sig"))
    v9 = json.loads((PROFILES / "phase8_batch_v9.json").read_text(
        encoding="utf-8-sig"))

    # Prose and identity are expected to differ; behaviour is not.
    narrative = {"profile_id", "written_on", "purpose", "why_routed",
                 "supersedes", "why_v8", "why_v9", "extraction_routes"}
    assert {k: v for k, v in v9.items() if k not in narrative} == \
        {k: v for k, v in v8.items() if k not in narrative}

    assert v9["extraction_routes"] == {"value": "targeted_v5_batch",
                                       "arm_identity": "targeted_v6"}
    assert v8["extraction_routes"]["value"] == v9["extraction_routes"]["value"]
    assert v9["supersedes"] == "phase8_batch_v8.json"


def test_v9_keeps_table_capture_on_the_baseline_the_ab_gate_chose():
    """The neutral TableCapture candidate regressed, so v9 must not adopt it.

    Neutral wording is not the principle being applied — "not shown to cost
    anything" is. Promoting table_capture_v2 here would contradict the B2
    diagnostic while looking like the same tidy-up.
    """
    contract = load_run_contract(PROFILES / "phase8_batch_v9.json")
    assert contract.table_capture_prompt_profile == "table_capture_v1"


def test_the_v9_contract_loads_and_routes_arm_identity_to_the_neutral_prompt():
    from react_review.tools.extraction_profile import uses_targeted_sections

    contract = load_run_contract(PROFILES / "phase8_batch_v9.json")
    assert contract.schema_version == 4
    assert contract.adequacy_enabled is True
    assert contract.extraction_routes["arm_identity"] == "targeted_v6"
    # A neutral prompt that lost the enumerate-then-assign sections would be a
    # different contract wearing a wording change's name.
    assert uses_targeted_sections(contract.extraction_routes["arm_identity"])


def test_only_the_phase8_table_makes_counts_exact():
    """A head count is a count; everything else keeps the bands it had.

    And the change lives in the Phase 8 file alone — the shipped default table
    and the comparator's own defaults are what every Phase 6/7 recording is
    still replayed against.
    """
    from react_review.audit import ToleranceTable

    contract = load_run_contract(PROFILES / "phase8.json")
    pinned = ToleranceTable.from_yaml(contract.tolerances_path)
    defaults = ToleranceTable()
    legacy_file = ToleranceTable.from_yaml(ROOT / "configs" / "tolerances.yaml")

    for field in ("sample_size", "cohort_n", "event_count"):
        assert pinned.rel_tolerance(field) == 0.0
        assert defaults.rel_tolerance(field) == 0.01      # unchanged
        assert legacy_file.rel_tolerance(field) == 0.01   # unchanged
    for field in ("hazard_ratio", "age", "p_value", "eat_thickness"):
        assert pinned.rel_tolerance(field) == defaults.rel_tolerance(field)
        assert pinned.sd_rel_tolerance(field) == defaults.sd_rel_tolerance(field)


# --- refusals -------------------------------------------------------------

def _contract_in(tmp_path: Path, **overrides) -> Path:
    """A copy of the shipped contract, with its tolerance table beside it.

    Paths inside a run profile resolve against the profile's own directory, so
    a profile copied anywhere keeps working — that is what this layout checks
    as much as the overrides do.
    """
    run_dir = tmp_path / "run_profiles"
    run_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "tolerances.phase8.yaml").write_bytes(
        (ROOT / "configs" / "tolerances.phase8.yaml").read_bytes())
    body = json.loads((PROFILES / "phase8.json").read_text(encoding="utf-8-sig"))
    body.update(overrides)
    path = run_dir / "contract.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_profile_resolves_its_files_relative_to_itself(tmp_path):
    contract = load_run_contract(_contract_in(tmp_path))
    assert contract.tolerances_path == (tmp_path / "tolerances.phase8.yaml").resolve()


def test_a_stale_tolerance_hash_is_refused(tmp_path):
    path = _contract_in(tmp_path, tolerances_sha256="0" * 64)
    with pytest.raises(ContractError, match="does not match the tolerances_sha256"):
        load_run_contract(path)


def test_unknown_contract_values_are_refused(tmp_path):
    for field, value, message in (
            ("extraction_profile", "targeted_v9", "extraction_profile"),
            ("semantic_prompt_profile", "semantic_v9", "semantic_prompt_profile"),
            ("context_policy", "always_parsed", "context_policy"),
            ("scope_policy", "maybe", "scope_policy"),
            ("schema_version", 7, "schema_version")):
        with pytest.raises(ContractError, match=message):
            load_run_contract(_contract_in(tmp_path, **{field: value}))


def test_an_undefined_scope_axis_is_refused(tmp_path):
    with pytest.raises(ContractError, match="scope axis"):
        load_run_contract(_contract_in(
            tmp_path, required_scope_axes={"cohort_n": ["trial_phase"]}))


def test_enabling_the_scope_check_without_saying_which_axes_is_refused(tmp_path):
    with pytest.raises(ContractError, match="which axes"):
        load_run_contract(_contract_in(tmp_path, required_scope_axes={}))


def test_old_contract_schema_may_not_silently_ignore_adequacy_fields(tmp_path):
    with pytest.raises(ContractError, match="schema_version 4"):
        load_run_contract(_contract_in(
            tmp_path, adequacy_policy_id="evidence_adequacy_v1"))


def test_v4_contract_refuses_an_unpinned_adequacy_evaluator(tmp_path):
    run_dir = tmp_path / "run_profiles"
    run_dir.mkdir(parents=True)
    (tmp_path / "tolerances.phase8.yaml").write_bytes(
        (ROOT / "configs" / "tolerances.phase8.yaml").read_bytes())
    body = json.loads(
        (PROFILES / "phase8_batch_v8.json").read_text(encoding="utf-8-sig"))
    body["adequacy_evaluator_hash"] = "sha256:" + "0" * 64
    path = run_dir / "bad-v4.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(ContractError, match="adequacy_evaluator_hash"):
        load_run_contract(path)


# --- the override boundary ------------------------------------------------

def test_execution_flags_are_not_contract_overrides():
    """--extraction and --semantic must stay free: record now, replay later."""
    contract = load_run_contract(PROFILES / "phase8.json")
    guard_contract_overrides(contract, {"--extraction": "record",
                                        "--semantic": "cache-only"})


@pytest.mark.parametrize("flag,value", [
    ("--tolerances", "configs/tolerances.yaml"),
    ("--extraction-profile", "legacy_v3"),
    ("--semantic-profile", "semantic_v1"),
    ("--scope", "off"),
])
def test_a_flag_that_would_change_the_answer_is_refused(flag, value):
    contract = load_run_contract(PROFILES / "phase8.json")
    with pytest.raises(ContractError, match="already fixes"):
        guard_contract_overrides(contract, {flag: value})


def test_a_flag_is_only_refused_where_the_contract_speaks():
    """legacy names no tolerance table, so --tolerances is not an override."""
    contract = load_run_contract(PROFILES / "legacy.json")
    guard_contract_overrides(contract, {"--tolerances": "configs/tolerances.yaml"})


# --- execution mode -------------------------------------------------------

def test_record_and_replay_need_a_cache(tmp_path):
    with pytest.raises(ContractError, match="needs an extraction cache"):
        ExecutionMode(extraction_mode="record").validate_modes()
    with pytest.raises(ContractError, match="needs a cache path"):
        ExecutionMode(semantic_mode="cache-only").validate_modes()
    ExecutionMode(extraction_mode="replay",
                  extraction_cache=tmp_path / "e.json",
                  semantic_mode="cache-only",
                  semantic_cache=tmp_path / "s.json").validate_modes()


def test_unknown_modes_are_refused():
    with pytest.raises(ContractError, match="extraction mode"):
        ExecutionMode(extraction_mode="dry-run").validate_modes()


# --- the manifest ---------------------------------------------------------

def test_a_record_and_its_replay_share_a_contract_but_not_a_mode(tmp_path):
    """Acceptance 5: identical question, different execution — provably so."""
    contract = load_run_contract(PROFILES / "phase8.json")
    cache = tmp_path / "extraction.json"
    cache.write_text('{"entries": {}}', encoding="utf-8")
    inputs = {"review_pdf": "ABC123"}

    record = RunManifest.of(
        contract, ExecutionMode(extraction_mode="record", extraction_cache=cache),
        inputs=inputs, context_source="parsed")
    replay = RunManifest.of(
        contract, ExecutionMode(extraction_mode="replay", extraction_cache=cache),
        inputs=inputs, context_source="parsed")

    assert record.same_contract_as(replay)
    assert record.execution["extraction_mode"] == "record"
    assert replay.execution["extraction_mode"] == "replay"
    assert record != replay


def test_a_partial_manifest_carries_no_cache_hash(tmp_path):
    """A file still being appended to has no meaningful content hash."""
    contract = load_run_contract(PROFILES / "legacy.json")
    cache = tmp_path / "extraction.json"
    cache.write_text('{"entries": {}}', encoding="utf-8")
    mode = ExecutionMode(extraction_mode="record", extraction_cache=cache)

    partial = RunManifest.of(contract, mode)
    assert partial.complete is False
    assert partial.extraction_cache_sha256 == ""
    assert partial.execution["extraction_cache"] == str(cache)

    finished = partial.finalise(mode)
    assert finished.complete is True
    assert finished.extraction_cache_sha256 == sha256_file(cache)


def test_manifest_records_the_resolved_evidence_gate_runtime():
    from react_review.production import evidence_adequacy_runtime

    contract = load_run_contract(PROFILES / "phase8_batch_v8.json")
    evaluator = evidence_adequacy_runtime(contract)
    manifest = RunManifest.of(contract, ExecutionMode())
    manifest.adequacy_runtime = RunManifest.adequacy_of(evaluator)

    runtime = manifest.model_dump(mode="json")["adequacy_runtime"]
    assert runtime["policy_id"] == contract.adequacy_policy_id
    assert runtime["policy_sha256"] == contract.adequacy_policy_hash
    assert runtime["evaluator_id"] == contract.adequacy_evaluator_id
    assert runtime["evaluator_version"] == contract.adequacy_evaluator_version
    assert runtime["evaluator_hash"] == contract.adequacy_evaluator_hash
    assert len(runtime["git_commit"]) == 40
    assert runtime["release_eligible"] is True


def test_a_different_context_source_is_a_different_question(tmp_path):
    contract = load_run_contract(PROFILES / "phase8.json")
    mode = ExecutionMode(extraction_mode="live")
    assert not RunManifest.of(contract, mode, context_source="cli").same_contract_as(
        RunManifest.of(contract, mode, context_source="parsed"))


# --- the legacy reconstruction -------------------------------------------

def test_the_legacy_reconstruction_never_borrows_a_new_axis():
    contract = legacy_contract(extraction_profile="targeted_v4",
                               semantic_prompt_profile="semantic_v2_specificity",
                               source="phase7")
    assert contract.derived_from_legacy is True
    assert contract.tolerances_path is None and contract.scope_enabled is False
    assert contract.context_policy == "cli_only"
    # …while keeping what the older file DID declare.
    assert contract.extraction_profile == "targeted_v4"
