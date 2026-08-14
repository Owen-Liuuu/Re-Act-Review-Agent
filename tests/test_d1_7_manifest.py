"""The recording manifest is checked against the artifacts it describes.

The first one was hand-written, and every error in it was of a kind no test
could see: 32-character values that still looked like SHA-256, the global cache
total reported as the semantic one, and no record at all of the four JSON
artifacts the conclusions were computed from. 1327 tests were green throughout.

So the manifest is generated, and this is the thing that reads it back.
"""
from __future__ import annotations

import json
import sys

import pytest

from tests.conftest import requires_frozen_evaluator

from react_review.contracts import repo_root

MANIFEST = repo_root() / "docs/baselines/d1_7_batch_recording_manifest.json"
sys.path.insert(0, str(repo_root() / "eval"))


def _body() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8-sig"))


def test_the_manifest_agrees_with_the_artifacts_it_hashes():
    """The only failure permitted is the declared one: the recording is not yet
    retrievable by anyone but its author."""
    import d1_7_manifest

    problems = d1_7_manifest.verify(MANIFEST)
    unexpected = [p for p in problems if "artifact_storage" not in p]
    assert unexpected == [], unexpected


def test_the_unreachable_recording_is_declared_and_not_quietly_dropped():
    """A manifest that hashes files nobody else can obtain proves that a file
    existed, not that a result can be reproduced.

    The gap is closed by a bundle published AND verified elsewhere, not by
    filling in a string: `location: "foo"` used to satisfy the old check.
    """
    import d1_7_manifest

    problems = d1_7_manifest.verify(MANIFEST)
    assert any("blocked" in p for p in problems), (
        "either a bundle was published and independently verified — in which "
        "case the status says so — or the gap must keep announcing itself")
    storage = _body()["artifact_storage"]
    assert storage["status"] == "blocked"
    assert storage["bundle"]["uri"] is None


def test_every_hash_is_a_whole_sha256():
    """A truncated hash still looks like a hash, which is why it survived."""
    body = _body()
    for name, digest in body["recording"]["contracts_in_force_then"].items():
        assert len(digest) == 64, name
    for name, digest in body["reanalysis_contracts"].items():
        assert len(digest) == 64, name
    assert len(body["cache"]["sha256_after"]) == 64
    assert len(body["cache"]["seeded_from_sha256"]) == 64
    for entry in body["artifacts"].values():
        assert len(entry["sha256"]) == 64
    for row in body["prompts"]:
        assert len(row["prompt_sha256"]) == 64


def test_each_prompt_is_recorded_with_the_sha_of_the_prompt_itself():
    """The pre-registration asked for question_id, prompt_sha256, attempt and
    cache_key. The cache key is a hash OVER the prompt's sha and cannot stand in
    for it."""
    rows = _body()["prompts"]
    assert len(rows) == 7
    for row in rows:
        assert {"question_id", "prompt_sha256", "attempt", "cache_key",
                "claim_ids"} <= set(row)
        assert row["prompt_sha256"] != row["cache_key"]
    assert len({r["prompt_sha256"] for r in rows}) == 7


def test_the_cache_split_is_not_reported_as_the_semantic_total():
    """v1 wrote the global 16 as the semantic hit count. It is extraction plus
    semantic, and the two answer different questions."""
    results = _body()["results"]
    assert results["global_cache_hits"] == 16
    assert results["extraction_cache_hits"] == 13
    assert "global counter is extraction plus semantic" in results["cache_split_note"]


def test_retries_are_reported_at_the_level_they_happened():
    """Batch retries were 0. The run-level counter says 3, and those are
    cache-served arm-identity replays — a different fact with a similar name."""
    did = _body()["what_the_run_did"]
    assert did["batch_contract_retries"] == 0
    assert did["run_level_repeated_attempts"] == 3
    assert "not batch retries" in did["repeated_attempts_note"]
    assert "max_retries" in did["backend_requests_note"], (
        "7 logical complete() calls is not a claim about HTTP requests")


def test_both_gate_verdicts_are_computed_by_the_classifier():
    """v1 defines wrong_released as a wrong value released WITHOUT review, and
    MA015 was review_required. The FAIL reported on the day came from an ad-hoc
    classifier that contradicted the gate's own text."""
    gate = _body()["gate"]
    assert gate["classifier"] == "src/react_review/acceptance_transitions.py"
    assert gate["gate_v1_verdict"] == "NOT_EVALUABLE"
    assert "MA015" in gate["gate_v1_reason"]
    assert "review_required=True" in gate["gate_v1_note"]
    assert "capability floor" in gate["gate_v1_second_defect"]


def test_the_three_identities_are_kept_apart():
    """The recording, the manifest describing it and the reanalyses re-reading
    it happened at three commits under three sets of contracts. One `commit`
    field made them look like one event, and listed a gate published after the
    run among the run's own contracts."""
    body = _body()
    assert body["recording"]["code_commit"].startswith("aafa20a")
    # The generation identity is the GENERATOR's bytes and the commit that last
    # touched it — never the current HEAD. A manifest committed alongside its
    # generator would otherwise record the commit before its own, because a file
    # cannot contain the hash of a commit that contains the file.
    generation = body["manifest_generation"]
    assert generation["generator_sha256"] and generation["generator"]
    assert "code_commit" not in generation
    # The gates and evaluators that did not exist yet are not listed as the
    # recording's.
    named = set(body["recording"]["contracts_in_force_then"])
    assert "aggregation_evaluator" not in named
    assert body["recording"]["contracts_in_force_then"]["feature_gate"] !=         body["reanalysis_contracts"]["feature_gate"]


def test_each_reanalysis_carries_its_own_gate_and_release_status():
    """A development reading and a release-eligible one must not be one row."""
    entries = {r["id"]: r for r in _body()["reanalyses"]}
    early = entries["d1_7_3_component_verification"]
    assert early["aggregation_runtime"]["release_eligible"] is False
    assert early["gate_verdict"] == "FAIL"
    assert any("identity_wrong_released" in u
               for u in early["unmet_hard_conditions"])

    frozen = entries["d1_7_5_component_and_identity"]
    assert frozen["aggregation_runtime"]["release_eligible"] is True
    assert frozen["aggregation_runtime"]["git_commit_matches_evaluator"] is True
    assert frozen["compare_runtime"]["release_eligible"] is True
    assert frozen["unmet_hard_conditions"] == []
    assert frozen["gate"] == "d1_batch_v3"


def test_applying_v2_to_this_run_is_labelled_post_hoc():
    """The gate was changed after the run by someone who knew the verdict it
    would change. That has to be on the artifact, not only in a commit."""
    note = _body()["gate"]["gate_v2_note"]
    assert "POST-HOC REANALYSIS" in note
    assert "exactly 8" in note


def test_no_secret_reaches_the_manifest():
    assert "api_key" not in MANIFEST.read_text(encoding="utf-8").lower()


def test_the_manifest_makes_no_claim_the_recording_cannot_support():
    claims = " ".join(_body()["claims_not_made"])
    assert "no speedup" in claims
    assert "NOT ESTIMABLE" in claims


@pytest.mark.skipif(
    not (repo_root() / "output/baselines/melanoma_checkpoint_2017"
         / "phase8_batch_extraction_cache.json").is_file(),
    reason="the recording is local-only and not in this checkout")
def test_the_recorded_prompts_still_produce_the_keys_the_run_wrote():
    """The prompt shas are re-derived offline, so they are only trustworthy if
    the keys they compute are the ones actually in the cache."""
    requires_frozen_evaluator()
    import d1_7_manifest

    rows, _ = d1_7_manifest._prompt_rows()
    published = {(r["question_id"], r["attempt"]): r["prompt_sha256"]
                 for r in _body()["prompts"]}
    for row in rows:
        assert published[(row["question_id"], row["attempt"])] == \
            row["prompt_sha256"]


# --- the verifier really fails; proved by breaking things -------------------

def _probe(tmp_path, name, mutate):
    """One tampered copy, and what the verifier says about it."""
    import d1_7_manifest

    body = json.loads(json.dumps(_body()))
    mutate(body)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return [p for p in d1_7_manifest.verify(path)
            if "artifact_storage" not in p]


@pytest.mark.parametrize("name,mutate", [
    ("cache_hash", lambda b: b["cache"].__setitem__("sha256_after", "A" * 64)),
    ("seeded_hash",
     lambda b: b["cache"].__setitem__("seeded_from_sha256", "B" * 64)),
    ("reanalysis_artifact_hash",
     lambda b: b["reanalyses"][-1]["artifact"].__setitem__("sha256", "C" * 64)),
    ("accuracy_inflated",
     lambda b: b["reanalyses"][-1].__setitem__("label_accuracy", 0.9)),
    ("verdict_flipped",
     lambda b: b["reanalyses"][0].__setitem__("gate_verdict", "PASS")),
    ("version_in_prose",
     lambda b: b["reanalyses"][-1].__setitem__("what", "replayed under 9.9.9")),
    ("runtime_version",
     lambda b: b["reanalyses"][-1]["aggregation_runtime"].__setitem__(
         "evaluator_version", "1.7.0")),
    ("reanalysis_contract",
     lambda b: b["reanalysis_contracts"].__setitem__("feature_gate", "D" * 64)),
    ("recording_contract",
     lambda b: b["recording"]["contracts_in_force_then"].__setitem__(
         "feature_gate", "E" * 64)),
    ("prompt_set_hash",
     lambda b: b.__setitem__("prompt_set_sha256", "F" * 64)),
    ("one_prompt_sha",
     lambda b: b["prompts"][0].__setitem__("prompt_sha256", "0" * 64)),
])
def test_the_verifier_catches_a_retyped_value(tmp_path, name, mutate):
    """A verifier that only checks length passes any 64-character string, which
    is precisely the shape of the pin this repository once published without
    ever computing it. Each of these is a value that would look right."""
    assert _probe(tmp_path, name, mutate), f"{name} was not caught"


def test_a_manifest_made_by_different_code_is_not_trusted(tmp_path):
    assert _probe(tmp_path, "generator",
                  lambda b: b["manifest_generation"].__setitem__(
                      "generator_sha256", "9" * 64))
