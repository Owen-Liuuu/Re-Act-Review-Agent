"""Can somebody else get this recording and reproduce it? — and can they fake it?

The manifest hashed files nobody else could obtain, which proves a file existed
and nothing more. The objection used to be closed by filling in a `location`
string, so any word would have closed it.

Three states now, and only the last one counts: a uri is not a verification, and
a verification script is not a verification either. These tests are mostly about
the ways the last state could be claimed without being earned.
"""
from __future__ import annotations

import json
import sys
import zipfile

import pytest

from react_review.contracts import repo_root

sys.path.insert(0, str(repo_root() / "eval"))

RUNS = repo_root() / "output/baselines/melanoma_checkpoint_2017"
SCORED = RUNS / "d1_7_5_scored.json"

needs_recording = pytest.mark.skipif(
    not SCORED.is_file(),
    reason="the recording is local-only and not in this checkout")


def _storage(**over):
    body = {
        "status": "independently_verified",
        "bundle": {"uri": "https://example.org/d1_7.zip", "sha256": "A" * 64,
                   "size_bytes": 14122, "media_type": "application/zip",
                   "access_mode": "controlled",
                   "copyright_restrictions": "reviewer access only"},
        "independent_verification": {
            "attestation_sha256": "B" * 64, "verified_at": "2026-08-13",
            "repo_commit": "c" * 40, "downloaded_sha256": "A" * 64,
            "replay_backend_requests": 0, "label_accuracy": 0.6,
            "correct_rows": 9, "total_rows": 15, "output_sha256": "D" * 64},
    }
    body.update(over)
    return body


# --- the state machine ------------------------------------------------------

def test_a_non_empty_location_is_no_longer_enough():
    """The old check was "is the string non-empty", so `foo` closed it."""
    import verify_d1_7_bundle as bundle

    storage = _storage(status="available_unverified",
                       independent_verification=None)
    storage["bundle"]["uri"] = "foo"
    problems = bundle.check_storage_block(storage)
    assert any("not an address anyone can act on" in p for p in problems)


def test_a_uri_alone_cannot_claim_verification():
    import verify_d1_7_bundle as bundle

    problems = bundle.check_storage_block(
        _storage(independent_verification=None))
    assert any("there is no attestation" in p for p in problems)


def test_an_attestation_without_the_state_is_refused_too():
    import verify_d1_7_bundle as bundle

    problems = bundle.check_storage_block(_storage(status="available_unverified"))
    assert any("does not claim independent verification" in p for p in problems)


def test_a_blocked_status_may_not_carry_a_uri():
    import verify_d1_7_bundle as bundle

    problems = bundle.check_storage_block(
        _storage(status="blocked", independent_verification=None))
    assert any("at least" in p and "available_unverified" in p for p in problems)


def test_an_attestation_for_a_different_file_is_caught():
    import verify_d1_7_bundle as bundle

    storage = _storage()
    storage["independent_verification"]["downloaded_sha256"] = "9" * 64
    assert any("verified a different file" in p
               for p in bundle.check_storage_block(storage))


def test_a_verification_that_called_a_model_replayed_nothing():
    import verify_d1_7_bundle as bundle

    storage = _storage()
    storage["independent_verification"]["replay_backend_requests"] = 4
    assert any("reached a model" in p for p in bundle.check_storage_block(storage))


def test_a_verification_that_scored_differently_is_not_this_result():
    import verify_d1_7_bundle as bundle

    storage = _storage()
    storage["independent_verification"]["correct_rows"] = 12
    assert any("scored 12 correct" in p for p in bundle.check_storage_block(storage))


def test_access_mode_and_copyright_must_be_stated():
    """Whether a reviewer can actually reach it is part of the claim, and the
    bundle quotes a copyrighted paper verbatim."""
    import verify_d1_7_bundle as bundle

    storage = _storage()
    storage["bundle"]["access_mode"] = None
    storage["bundle"]["copyright_restrictions"] = None
    problems = bundle.check_storage_block(storage)
    assert any("access_mode" in p for p in problems)
    assert any("copyright_restrictions" in p for p in problems)


# --- the bundle itself ------------------------------------------------------

@needs_recording
def test_a_freshly_built_bundle_verifies_against_the_real_replay(tmp_path):
    """The whole point, end to end, on the machine that has everything. It does
    NOT close the gap — that needs a machine that has nothing."""
    import verify_d1_7_bundle as bundle

    target = tmp_path / "d1_7.zip"
    made = bundle.build(target)
    assert made["size_bytes"] > 0

    problems, seen = bundle.verify_bundle(
        target, uri="https://example.org/d1_7.zip")
    problems += bundle.check_replay(SCORED, seen.get("descriptor") or {})
    assert problems == [], problems


@needs_recording
def test_a_bundle_missing_the_semantic_cache_cannot_reproduce_the_result(tmp_path):
    """The extraction cache alone is not enough: the scoring run replays
    semantic judgements too, and without them the numbers cannot be reached."""
    import verify_d1_7_bundle as bundle

    source = tmp_path / "full.zip"
    bundle.build(source)
    stripped = tmp_path / "stripped.zip"
    with zipfile.ZipFile(source) as original, \
            zipfile.ZipFile(stripped, "w") as reduced:
        for name in original.namelist():
            if name != "phase7_semantic_cache.json":
                reduced.writestr(name, original.read(name))

    problems, _ = bundle.verify_bundle(stripped)
    assert any("phase7_semantic_cache.json" in p for p in problems)


@needs_recording
def test_a_member_swapped_inside_the_bundle_is_caught(tmp_path):
    import verify_d1_7_bundle as bundle

    source = tmp_path / "full.zip"
    bundle.build(source)
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(source) as original, \
            zipfile.ZipFile(tampered, "w") as edited:
        for name in original.namelist():
            payload = (b'{"entries": {}}'
                       if name == "phase8_batch_extraction_cache.json"
                       else original.read(name))
            edited.writestr(name, payload)

    problems, _ = bundle.verify_bundle(tampered)
    assert any("is not what BUNDLE.json says" in p for p in problems)


def test_a_uri_that_resolves_to_nothing_is_not_an_available_artifact(tmp_path):
    import verify_d1_7_bundle as bundle

    problems, _ = bundle.verify_bundle(tmp_path / "absent.zip",
                                       uri="https://example.org/absent.zip")
    assert any("is not here" in p for p in problems)


@needs_recording
def test_the_wrong_hash_or_size_is_refused(tmp_path):
    import verify_d1_7_bundle as bundle

    target = tmp_path / "d1_7.zip"
    made = bundle.build(target)
    wrong_hash, _ = bundle.verify_bundle(target, expect_sha="0" * 64)
    assert any("not the one the manifest names" in p for p in wrong_hash)
    wrong_size, _ = bundle.verify_bundle(
        target, expect_sha=made["sha256"], expect_size=made["size_bytes"] + 1)
    assert any("bytes and the manifest says" in p for p in wrong_size)


# --- the replay the bundle promises ----------------------------------------

def test_a_replay_that_missed_the_cache_is_not_replaying_this_recording():
    import verify_d1_7_bundle as bundle

    scored = {"metrics": {}, "rows": [],
              "run": {"telemetry": {"cache_hits": 16, "cache_misses": 2,
                                    "backend_requests": 0}}}
    problems = _check_scored(bundle, scored)
    assert any("cache miss" in p for p in problems)


def test_a_replay_that_scored_differently_is_refused():
    import verify_d1_7_bundle as bundle

    scored = {"metrics": {"label_accuracy": 0.8}, "rows": [],
              "run": {"telemetry": {"cache_hits": 16, "cache_misses": 0,
                                    "backend_requests": 0}}}
    assert any("label_accuracy" in p for p in _check_scored(bundle, scored))


def _check_scored(bundle, scored):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "scored.json"
    path.write_text(json.dumps(scored), encoding="utf-8")
    return bundle.check_replay(path, {})


# --- the receipt ------------------------------------------------------------

@needs_recording
def test_an_attestation_edited_after_signing_is_not_one(tmp_path):
    import verify_d1_7_bundle as bundle

    target = tmp_path / "d1_7.zip"
    bundle.build(target)
    _, seen = bundle.verify_bundle(target)
    receipt = bundle.attest(target, SCORED, uri="https://example.org/d1_7.zip",
                            seen=seen)
    assert bundle.attestation_is_intact(receipt)

    receipt["correct_rows"] = 15
    assert not bundle.attestation_is_intact(receipt)


# --- and the manifest keeps failing until all of it is real -----------------

def test_the_manifest_still_reports_the_gap():
    """Not a warning. It is meant to keep failing until a bundle is published
    AND verified somewhere else."""
    import d1_7_manifest

    manifest = repo_root() / "docs/baselines/d1_7_batch_recording_manifest.json"
    problems = d1_7_manifest.verify(manifest)
    assert any("blocked" in p for p in problems)
    body = json.loads(manifest.read_text(encoding="utf-8-sig"))
    assert body["artifact_storage"]["status"] == "blocked"
    assert body["artifact_storage"]["bundle"]["uri"] is None
    assert body["artifact_storage"]["independent_verification"] is None
