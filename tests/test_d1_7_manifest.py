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
    unexpected = [p for p in problems if "artifact_storage.location" not in p]
    assert unexpected == [], unexpected


def test_the_unreachable_recording_is_declared_and_not_quietly_dropped():
    """A manifest that hashes files nobody else can obtain proves that a file
    existed, not that a result can be reproduced. Saying so is the minimum."""
    import d1_7_manifest

    problems = d1_7_manifest.verify(MANIFEST)
    assert any("artifact_storage.location" in p for p in problems), (
        "either a retrievable location was agreed — in which case fill it in — "
        "or the gap must keep announcing itself")
    assert not _body()["artifact_storage"]["location"]


def test_every_hash_is_a_whole_sha256():
    """A truncated hash still looks like a hash, which is why it survived."""
    body = _body()
    for name, digest in body["contracts"].items():
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


def test_the_v2_verdict_cannot_be_read_as_the_route_working():
    """Its floor is unset, so the strongest thing it can say is that nothing
    forbidden happened."""
    gate = _body()["gate"]
    assert gate["gate_v2_verdict"] == "PASS_PROHIBITIONS_ONLY"
    assert gate["gate_v2_capability_judged"] is False
    assert gate["gate_v2_states"]["wrong_released"] == 0
    assert gate["gate_v2_states"]["wrong_but_flagged"] == 1


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
