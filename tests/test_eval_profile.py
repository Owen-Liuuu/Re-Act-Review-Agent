"""Phase 7 profile / overlay / target-contract loading — and its refusals.

The refusals are the point. A profile that quietly tolerated an unknown row, a
stale hash or a stray column would be a second answer key with no review, so
every one of those cases is asserted to raise rather than to warn.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from react_review.eval_profile import (
    ProfileError,
    load_profile,
    load_semantic_overlay,
    load_target_contract,
    sha256_file,
)

BENCHMARK = (Path(__file__).resolve().parents[1] / "eval" / "benchmarks"
             / "melanoma_checkpoint_2017")
ANSWER_KEY = BENCHMARK / "audit_template.csv"


def _answer_key_ids() -> list[str]:
    with open(ANSWER_KEY, encoding="utf-8-sig", newline="") as handle:
        return [row["audit_id"] for row in csv.DictReader(handle)
                if (row.get("audit_id") or "").strip()]


def _write(path: Path, rows: list[dict], columns: list[str]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


# --- the real, shipped contract ---

def test_shipped_profile_loads_and_pins_its_inputs():
    profile = load_profile(BENCHMARK, "phase7_profile.json",
                           answer_key_ids=_answer_key_ids())
    assert profile.extraction_profile == "targeted_v4"
    assert profile.semantic_prompt_profile == "semantic_v2_specificity"
    assert len(profile.targets) == 15
    assert set(profile.semantic) == {"MA001", "MA003", "MA005", "MA007"}
    assert profile.provenance()["benchmark_profile_sha256"] == profile.sha256


def test_target_contract_is_one_to_one_with_the_answer_key():
    profile = load_profile(BENCHMARK, "phase7_profile.json",
                           answer_key_ids=_answer_key_ids())
    assert sorted(profile.targets) == sorted(_answer_key_ids())


def test_target_contract_carries_the_review_side_question_only():
    """No expectation may reach the extractor through this file."""
    with open(BENCHMARK / "phase7_target_contract.csv",
              encoding="utf-8-sig", newline="") as handle:
        columns = set(csv.DictReader(handle).fieldnames or [])
    assert not [c for c in columns if c.startswith("expected_")]
    assert columns == {"audit_id", "review_data_source", "raw_field_name",
                       "cohort_label", "cohort_label_source", "timepoint",
                       "cell_ref"}


def test_target_rows_do_not_line_up_with_review_ids_by_number():
    """The mapping is content-derived; the numbering coincidence is false."""
    profile = load_profile(BENCHMARK, "phase7_profile.json",
                           answer_key_ids=_answer_key_ids())
    assert profile.targets["MA004"].review_data_source == "M006"
    assert profile.targets["MA005"].review_data_source == "M004"
    assert profile.targets["MA007"].review_data_source == "M005"


def test_semantic_overlay_states_one_consistent_direction():
    profile = load_profile(BENCHMARK, "phase7_profile.json",
                           answer_key_ids=_answer_key_ids())
    ma005 = profile.semantic_for("MA005")
    assert ma005.expected_semantic_relation == "source_broader"
    assert ma005.expected_more_specific_side == "review"
    assert ma005.expected_review_required is True


# --- refusals: profile ---

def _profile_body() -> dict:
    return json.loads((BENCHMARK / "phase7_profile.json").read_text(encoding="utf-8-sig"))


def _profile_in(tmp_path: Path, body: dict) -> Path:
    """A copy of the benchmark's contract files, so a mutated profile is testable."""
    for name in ("manifest.json", "audit_template.csv",
                 "phase7_semantic_overlay.csv", "phase7_target_contract.csv"):
        (tmp_path / name).write_bytes((BENCHMARK / name).read_bytes())
    (tmp_path / "phase7_profile.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return tmp_path / "phase7_profile.json"


def test_missing_profile_file_is_refused(tmp_path):
    with pytest.raises(ProfileError, match="does not exist"):
        load_profile(tmp_path, "phase7_profile.json", answer_key_ids=["MA001"])


def test_stale_base_hash_is_refused(tmp_path):
    body = _profile_body()
    body["base_audit_template_sha256"] = "0" * 64
    _profile_in(tmp_path, body)
    with pytest.raises(ProfileError, match="has changed since this profile"):
        load_profile(tmp_path, "phase7_profile.json",
                     answer_key_ids=_answer_key_ids())


def test_stale_overlay_hash_is_refused(tmp_path):
    body = _profile_body()
    body["semantic_expectation_overlay_sha256"] = "0" * 64
    _profile_in(tmp_path, body)
    with pytest.raises(ProfileError, match="does not match the"):
        load_profile(tmp_path, "phase7_profile.json",
                     answer_key_ids=_answer_key_ids())


def test_unknown_profile_names_are_refused(tmp_path):
    body = _profile_body()
    body["extraction_profile"] = "targeted_v5"
    _profile_in(tmp_path, body)
    with pytest.raises(ProfileError, match="unknown extraction_profile"):
        load_profile(tmp_path, "phase7_profile.json",
                     answer_key_ids=_answer_key_ids())

    body = _profile_body()
    body["semantic_prompt_profile"] = "semantic_v9"
    _profile_in(tmp_path, body)
    with pytest.raises(ProfileError, match="unknown semantic_prompt_profile"):
        load_profile(tmp_path, "phase7_profile.json",
                     answer_key_ids=_answer_key_ids())


def test_wrong_schema_version_is_refused(tmp_path):
    body = _profile_body()
    body["schema_version"] = 7
    _profile_in(tmp_path, body)
    with pytest.raises(ProfileError, match="is not one of"):
        load_profile(tmp_path, "phase7_profile.json",
                     answer_key_ids=_answer_key_ids())


def test_undeclared_hash_is_refused(tmp_path):
    body = _profile_body()
    del body["target_contract_sha256"]
    _profile_in(tmp_path, body)
    with pytest.raises(ProfileError, match="does not declare"):
        load_profile(tmp_path, "phase7_profile.json",
                     answer_key_ids=_answer_key_ids())


# --- refusals: semantic overlay ---

_OVERLAY_COLUMNS = ["audit_id", "expected_semantic_relation",
                    "expected_more_specific_side", "expected_review_required",
                    "reason"]


def _overlay_row(**kw) -> dict:
    row = {"audit_id": "MA001", "expected_semantic_relation": "same",
           "expected_more_specific_side": "neither",
           "expected_review_required": "false", "reason": "r"}
    row.update(kw)
    return row


def test_overlay_rejects_an_audit_id_the_key_does_not_have(tmp_path):
    path = _write(tmp_path / "o.csv", [_overlay_row(audit_id="MA099")],
                  _OVERLAY_COLUMNS)
    with pytest.raises(ProfileError, match="answer key does not contain"):
        load_semantic_overlay(path, ["MA001"])


def test_overlay_rejects_duplicates(tmp_path):
    path = _write(tmp_path / "o.csv", [_overlay_row(), _overlay_row()],
                  _OVERLAY_COLUMNS)
    with pytest.raises(ProfileError, match="repeats audit_id"):
        load_semantic_overlay(path, ["MA001"])


def test_overlay_cannot_carry_a_label(tmp_path):
    path = _write(tmp_path / "o.csv",
                  [{**_overlay_row(), "expected_label": "match"}],
                  [*_OVERLAY_COLUMNS, "expected_label"])
    with pytest.raises(ProfileError, match="not allowed to carry"):
        load_semantic_overlay(path, ["MA001"])


def test_overlay_cannot_carry_source_evidence(tmp_path):
    path = _write(tmp_path / "o.csv",
                  [{**_overlay_row(), "source_quote": "…"}],
                  [*_OVERLAY_COLUMNS, "source_quote"])
    with pytest.raises(ProfileError, match="not allowed to carry"):
        load_semantic_overlay(path, ["MA001"])


def test_overlay_rejects_an_undefined_relation_or_side(tmp_path):
    path = _write(tmp_path / "o.csv",
                  [_overlay_row(expected_semantic_relation="narrower")],
                  _OVERLAY_COLUMNS)
    with pytest.raises(ProfileError, match="expected_semantic_relation"):
        load_semantic_overlay(path, ["MA001"])

    path = _write(tmp_path / "o2.csv",
                  [_overlay_row(expected_more_specific_side="both")],
                  _OVERLAY_COLUMNS)
    with pytest.raises(ProfileError, match="expected_more_specific_side"):
        load_semantic_overlay(path, ["MA001"])


def test_overlay_rejects_a_non_boolean_flag(tmp_path):
    path = _write(tmp_path / "o.csv",
                  [_overlay_row(expected_review_required="maybe")],
                  _OVERLAY_COLUMNS)
    with pytest.raises(ProfileError, match="not a boolean"):
        load_semantic_overlay(path, ["MA001"])


# --- refusals: target contract ---

_TARGET_COLUMNS = ["audit_id", "review_data_source", "raw_field_name",
                   "cohort_label", "cohort_label_source", "timepoint", "cell_ref"]


def _target_row(audit_id: str = "MA001") -> dict:
    return {"audit_id": audit_id, "review_data_source": "M001",
            "raw_field_name": "Design", "cohort_label": "",
            "cohort_label_source": "", "timepoint": "baseline",
            "cell_ref": "table_1:r1c1"}


def test_target_contract_must_cover_every_answer_key_row(tmp_path):
    path = _write(tmp_path / "t.csv", [_target_row()], _TARGET_COLUMNS)
    with pytest.raises(ProfileError, match="missing rows for: MA002"):
        load_target_contract(path, ["MA001", "MA002"])


def test_target_contract_rejects_an_extra_row(tmp_path):
    path = _write(tmp_path / "t.csv",
                  [_target_row(), _target_row("MA404")], _TARGET_COLUMNS)
    with pytest.raises(ProfileError, match="answer key does not contain"):
        load_target_contract(path, ["MA001"])


def test_target_contract_rejects_duplicates(tmp_path):
    path = _write(tmp_path / "t.csv", [_target_row(), _target_row()],
                  _TARGET_COLUMNS)
    with pytest.raises(ProfileError, match="repeats audit_id"):
        load_target_contract(path, ["MA001"])


def test_target_contract_cannot_carry_an_expectation(tmp_path):
    for column, value in (("expected_label", "match"),
                          ("source_value", "0.42"),
                          ("source_quote", "…"),
                          ("expected_semantic_relation", "same")):
        path = _write(tmp_path / f"t_{column}.csv",
                      [{**_target_row(), column: value}],
                      [*_TARGET_COLUMNS, column])
        with pytest.raises(ProfileError, match="not allowed to carry"):
            load_target_contract(path, ["MA001"])


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == sha256_file(path)
    assert sha256_file(path).isupper()


# --- the runtime contract a benchmark runs under (P8-0 U2) ---

def _v2_profile(tmp_path: Path, **overrides) -> Path:
    """A schema-2 benchmark profile that names a run contract by path + hash."""
    from react_review.contracts import repo_root

    for name in ("manifest.json", "audit_template.csv",
                 "phase7_semantic_overlay.csv", "phase7_target_contract.csv"):
        (tmp_path / name).write_bytes((BENCHMARK / name).read_bytes())
    body = json.loads((BENCHMARK / "phase7_profile.json").read_text(encoding="utf-8-sig"))
    run_profile = repo_root() / "configs" / "run_profiles" / "phase8.json"
    body.update({
        "schema_version": 2,
        "run_profile": "configs/run_profiles/phase8.json",
        "run_profile_sha256": sha256_file(run_profile),
    })
    body.update(overrides)
    path = tmp_path / "profile_v2.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_a_v1_profile_gets_the_contract_it_actually_ran_under():
    """Phase 7 predates these axes: they are reconstructed, never inherited."""
    profile = load_profile(BENCHMARK, "phase7_profile.json",
                           answer_key_ids=_answer_key_ids())
    contract = profile.run_contract
    assert contract.derived_from_legacy is True
    # what the v1 file DID declare survives …
    assert contract.extraction_profile == "targeted_v4"
    assert contract.semantic_prompt_profile == "semantic_v2_specificity"
    # … and every axis it never declared takes the pre-Phase-8 value.
    assert contract.tolerances_path is None
    assert contract.population_contract_path is None
    assert contract.scope_enabled is False
    assert contract.context_policy == "cli_only"


def test_a_frozen_benchmark_does_not_inherit_the_production_contract():
    """Acceptance 6: switching the production default must not reach Phase 7.

    The reconstruction is compared against the shipped phase8 contract rather
    than against constants, so this test keeps failing if a future edit makes
    the two converge by accident.
    """
    from react_review.run_profile import load_run_contract
    from react_review.contracts import repo_root

    phase8 = load_run_contract(repo_root() / "configs" / "run_profiles" / "phase8.json")
    contract = load_profile(BENCHMARK, "phase7_profile.json",
                            answer_key_ids=_answer_key_ids()).run_contract
    assert phase8.scope_enabled and contract.scope_enabled is False
    assert phase8.tolerances_path is not None and contract.tolerances_path is None
    assert phase8.context_policy != contract.context_policy


def test_a_v2_profile_resolves_its_run_contract(tmp_path):
    profile = load_profile(tmp_path, _v2_profile(tmp_path),
                           answer_key_ids=_answer_key_ids())
    assert profile.schema_version == 2
    assert profile.run_contract.profile_id == "phase8"
    assert profile.run_contract.scope_enabled is True
    assert profile.provenance()["run_scope_policy"] == "on"


def test_v4_benchmark_profile_requires_and_exposes_the_evidence_gate():
    profile = load_profile(
        BENCHMARK, "phase8_batch_v8_profile.json",
        answer_key_ids=_answer_key_ids())

    assert profile.schema_version == 4
    assert profile.run_contract.adequacy_enabled is True
    provenance = profile.provenance()
    assert provenance["run_adequacy_policy_id"] == "evidence_adequacy_v1"
    assert provenance["run_adequacy_evaluator_version"] == "1.0.0"


def test_a_v2_profile_with_a_stale_run_contract_hash_is_refused(tmp_path):
    path = _v2_profile(tmp_path, run_profile_sha256="0" * 64)
    with pytest.raises(ProfileError, match="does not match the run_profile_sha256"):
        load_profile(tmp_path, path, answer_key_ids=_answer_key_ids())


def test_two_sources_of_truth_may_not_disagree(tmp_path):
    """The benchmark's own declaration must agree with the contract it names."""
    path = _v2_profile(tmp_path, extraction_profile="legacy_v3")
    with pytest.raises(ProfileError, match="but its run contract"):
        load_profile(tmp_path, path, answer_key_ids=_answer_key_ids())


def test_a_v2_profile_must_name_a_run_contract(tmp_path):
    path = _v2_profile(tmp_path)
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    del body["run_profile"]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ProfileError, match="must name a run_profile"):
        load_profile(tmp_path, path, answer_key_ids=_answer_key_ids())
