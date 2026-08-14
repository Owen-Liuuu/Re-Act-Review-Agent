"""The pinned hashes are checked, not merely written down.

Twice now a contract has been edited in place while the document above it said
that never happens, and once a published hash turned out never to have been
computed from the file at all — it shared a twelve-character prefix with the
real one and differed after, which is the shape a partly-copied value has and no
content change can produce.

Both survived because nothing compared the document to the files. A governance
rule that only exists in prose is a rule until the first time somebody is in a
hurry. This is the same rule, in a form that fails the build.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from react_review.contracts import repo_root, sha256_file

DOC = Path("docs/acceptance/gate_versions.md")

#: Every contract whose bytes are frozen, and the hash the document publishes.
#: Changing one of these values is a decision, not a fix: it means a file that
#: was supposed to be immutable has moved.
PINNED = {
    "configs/run_profiles/phase8_batch.json": "17C65B8C07A45898",
    "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_profile.json": "8F13C10443B8BEDE",
    "configs/gates/cross_domain_v1.json": "AE182D0097A67A18",
    "configs/gates/cross_domain_v2.json": "E29CF4F803BA0F8A",
    "configs/aggregation/safe_sum_v1.json": "1C99DCE79E4FDD3A",
    "configs/aggregation/safe_sum_v2.json": "5FED9271920DF0A4",
    "configs/aggregation/safe_sum_v3.json": "93E381F2ED633E06",
    "configs/aggregation/safe_sum_v4.json": "FE1B925C28FA7558",
    "configs/aggregation/registry.json": "0F16E3F1228BC4E9",
    "configs/aggregation/safe_sum_v5.json": "DAEB6715F812E88E",
    "configs/aggregation/registry_v2.json": "3F593BB1097A3CA0",
    "configs/aggregation/registry_v3.json": "7B439CBDA99D54D4",
    "configs/aggregation/evaluators/safe_aggregation_1.5.0.json": "0C5274C554AF1813",
    "configs/aggregation/evaluators/safe_aggregation_1.6.0.json": "E2514A771F246A1C",
    "configs/aggregation/registry_v4.json": "83AB58639A50D8EE",
    "configs/aggregation/evaluators/safe_aggregation_1.6.1.json": "A49941F1458D43B4",
    "configs/aggregation/evaluators/safe_aggregation_1.4.0.json": "3B4912D2A0596CEF",
    # What the model is ASKED under targeted_v5_batch. Pinned here so that
    # editing the contract to make an edited prompt pass is a visible decision
    # and not a hash quietly retyped.
    "configs/prompt_contracts/batch_v5.json": "1A02A44DDE55A797",
    # WHERE the evidence for each batched claim is. An answer key, so it is
    # frozen like one: a coverage number computed against a key that moved is
    # not comparable to the number before it moved.
    "eval/benchmarks/melanoma_checkpoint_2017/excerpt_gold_v1.json":
        "030A04D6332A5E2E",
    "configs/aggregation/registry_v5.json": "B4BFFCB393E581F2",
    "configs/aggregation/evaluators/safe_aggregation_1.6.2.json": "F01DF1DD72077BA7",
    "configs/run_profiles/phase8_batch_v2.json": "AD9F2F180F718BEB",
    "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v2_profile.json":
        "9275079632081DB6",
    "eval/benchmarks/melanoma_checkpoint_2017/excerpt_gold_v2.json":
        "9F67F418E245E656",
    "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v3_profile.json":
        "779E9F1C97D5B9A9",
    # What a D1-7 recording is expected to ask for, and what it has to achieve.
    # Both frozen before the recording exists, because a plan nothing compares
    # against is a description and a gate written afterwards is not a gate.
    "eval/benchmarks/melanoma_checkpoint_2017/d1_7_expected_plan.json":
        "72AD64CBE18447AD",
    "configs/gates/d1_batch_v1.json": "B624742F31536742",
    "configs/gates/d1_batch_v2.json": "1221EC40F789F04C",
    "configs/aggregation/registry_v6.json": "93D50BFC3F0B48CC",
    "configs/aggregation/evaluators/safe_aggregation_1.7.0.json": "43703B7DADE72943",
    "configs/run_profiles/phase8_batch_v3.json": "936985979F2A7726",
    "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v4_profile.json":
        "4F58FDB3672A1F46",
    "configs/aggregation/registry_v7.json":
        "CBAB63703AA30014",
    "configs/aggregation/evaluators/safe_aggregation_1.8.0.json":
        "D566BA50B65FCF7A",
    "configs/compare/evaluators/deterministic_compare_1.0.0.json":
        "44B1F8045D405BD7",
    "configs/run_profiles/phase8_batch_v4.json":
        "19A412345ECAD1EA",
    "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v5_profile.json":
        "B7F5EE9E1FC95D4F",
    "configs/aggregation/registry_v8.json":
        "27F66638AE845DAD",
    "configs/aggregation/evaluators/safe_aggregation_1.8.1.json":
        "7DCEB9C565CECF68",
    "configs/run_profiles/phase8_batch_v5.json":
        "C7F38629DA4EADBB",
    "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v6_profile.json":
        "5FB8CE66A4C05194",
    # Identical judging content to v2; it changes only what the gate says about
    # ITSELF, because v2 inherited v1's "fixed before the recording exists" and
    # that was false of v2.
    "configs/gates/d1_batch_v3.json": "BF991151137EE376",
    "configs/prompt_contracts/table_capture_v1.json": "654F6B9ABBEDFFE3",
    "configs/prompt_contracts/table_capture_v2.json": "285D676ED59098F0",
    "configs/run_profiles/phase8_batch_v6.json": "46888B89138AAFFC",
    "eval/table_capture_ab_v1.json": "68E35AE880AC87DD",
}


@pytest.mark.parametrize("path,expected", sorted(PINNED.items()))
def test_a_frozen_contract_still_has_the_bytes_it_was_published_with(path, expected):
    assert sha256_file(repo_root() / path)[:16].upper() == expected


@pytest.mark.parametrize("path,expected", sorted(PINNED.items()))
def test_the_governance_document_publishes_the_hash_the_file_actually_has(
        path, expected):
    """The failure that started this: a pin nobody ever compared to a file."""
    text = (repo_root() / DOC).read_text(encoding="utf-8")
    rows = dict(re.findall(r"`(\S+?\.json)`\s*\|\s*`([0-9A-F]{16})`", text))
    assert path in rows, f"{path} is frozen but {DOC} does not publish its hash"
    assert rows[path] == expected


def test_contract_files_are_pinned_to_lf_so_the_hashes_are_reproducible():
    """A Windows checkout renormalises to CRLF and changes every hash."""
    attributes = (repo_root() / ".gitattributes").read_text(encoding="utf-8")
    for path in PINNED:
        # Either the directory is covered, or the file is named. Naming one file
        # is not a lesser rule: a blanket pattern over `configs/run_profiles` or
        # `eval/benchmarks` would renormalise a file a frozen benchmark pins,
        # which is how one was made unrunnable already.
        directory = Path(path).parent.as_posix()
        assert (f"{directory}/*.json text eol=lf" in attributes
                or f"{path} text eol=lf" in attributes), path
    for path in PINNED:
        assert b"\r\n" not in (repo_root() / path).read_bytes(), path


def test_a_policy_may_not_declare_a_rule_that_nothing_reads(tmp_path):
    """A file that appears to control behaviour it does not control is worse
    than a file that says nothing."""
    import json

    from react_review.contracts import ContractError
    from react_review.tools.safe_aggregation import load_aggregation_policy

    body = json.loads((repo_root() / "configs/aggregation/safe_sum_v5.json"
                       ).read_text(encoding="utf-8"))
    body["requirements"]["require_something_nobody_implements"] = True
    path = tmp_path / "invented.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="nothing reads"):
        load_aggregation_policy(str(path))


def test_an_invariant_may_not_be_switched_off(tmp_path):
    import json

    from react_review.contracts import ContractError
    from react_review.tools.safe_aggregation import load_aggregation_policy

    body = json.loads((repo_root() / "configs/aggregation/safe_sum_v5.json"
                       ).read_text(encoding="utf-8"))
    body["invariants"]["match_on_both_population_axes"] = False
    path = tmp_path / "disabled.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="not a switch"):
        load_aggregation_policy(str(path))


def test_a_policy_may_not_omit_a_key_and_take_a_silent_default(tmp_path):
    """`"minimum_axes": []` used to load as ["population_basis"]."""
    import json

    from react_review.contracts import ContractError
    from react_review.tools.safe_aggregation import load_aggregation_policy

    body = json.loads((repo_root() / "configs/aggregation/safe_sum_v5.json"
                       ).read_text(encoding="utf-8"))
    del body["minimum_axes"]
    path = tmp_path / "no_axes.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="omits 'minimum_axes'"):
        load_aggregation_policy(str(path))


def test_a_policy_may_not_require_an_axis_nothing_compares(tmp_path):
    import json

    from react_review.contracts import ContractError
    from react_review.tools.safe_aggregation import load_aggregation_policy

    body = json.loads((repo_root() / "configs/aggregation/safe_sum_v5.json"
                       ).read_text(encoding="utf-8"))
    body["minimum_axes"] = ["population_basis", "phase_of_the_moon"]
    path = tmp_path / "invented_axis.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ContractError, match="phase_of_the_moon"):
        load_aggregation_policy(str(path))
