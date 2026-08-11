"""What decided, and whether this run may publish what it decided.

A policy hash was never enough. Every wrong total in this phase came from rules
that read correctly and code that did not enforce them, so an artifact naming
only `safe_sum_v4` claims a reproducibility it does not have: two commits of the
evaluator, one policy, different answers, and nothing to say which ran.

These tests are mostly about the ways an identity can be untrue while looking
fine — a hash that no longer describes the files, a manifest nobody updated, a
commit that does not cover the code that ran.
"""
from __future__ import annotations

import json

import pytest

from react_review.contracts import ContractError, repo_root
from react_review.tools.aggregation_identity import (
    HASH_ALGORITHM,
    REGISTERED,
    UNREGISTERED,
    EvaluatorIdentity,
    evaluator_readiness,
    hash_sources,
    load_evaluator_manifest,
)

VERSION = "1.4.0"
POLICY = "safe_sum_v4"


def _policy_hash() -> str:
    from react_review.tools.safe_aggregation import load_aggregation_policy
    return load_aggregation_policy().sha256


# --- the hash means what it says ------------------------------------------

def test_the_published_hash_still_describes_the_files_on_disk():
    """The whole point. If this fails, the evaluator changed without a version."""
    manifest = load_evaluator_manifest(VERSION)
    digest, per_file = hash_sources(list(manifest.source_files))
    assert digest == manifest.evaluator_hash
    assert per_file == manifest.source_files


def test_one_changed_byte_changes_the_evaluator_hash(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    before, _ = hash_sources(["a.py", "b.py"], tmp_path)
    (tmp_path / "b.py").write_text("y = 3\n", encoding="utf-8")
    after, _ = hash_sources(["a.py", "b.py"], tmp_path)
    assert before != after


def test_the_order_the_files_are_listed_in_does_not_change_the_hash(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")
    assert hash_sources(["a.py", "b.py"], tmp_path)[0] == \
        hash_sources(["b.py", "a.py"], tmp_path)[0]


def test_line_endings_do_not_change_the_hash(tmp_path):
    """A Windows checkout must not be a different evaluator."""
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    lf, _ = hash_sources(["a.py"], tmp_path)
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    crlf, _ = hash_sources(["a.py"], tmp_path)
    assert lf == crlf


def test_moving_a_byte_across_a_file_boundary_changes_the_hash(tmp_path):
    """Why each file contributes its path and its LENGTH, not just its bytes."""
    (tmp_path / "a.py").write_text("xy", encoding="utf-8")
    (tmp_path / "b.py").write_text("z", encoding="utf-8")
    one, _ = hash_sources(["a.py", "b.py"], tmp_path)
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.py").write_text("yz", encoding="utf-8")
    two, _ = hash_sources(["a.py", "b.py"], tmp_path)
    assert one != two


def test_renaming_a_file_changes_the_hash(tmp_path):
    """Why the path is in there: the same code under a different name is a
    different evaluator, because a manifest names files."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    one, _ = hash_sources(["a.py"], tmp_path)
    (tmp_path / "c.py").write_text("x = 1\n", encoding="utf-8")
    two, _ = hash_sources(["c.py"], tmp_path)
    assert one != two


# --- the manifest is a claim that can be wrong ----------------------------

def test_a_manifest_hashed_by_another_scheme_is_refused(tmp_path, monkeypatch):
    body = json.loads((repo_root() / "configs/aggregation/evaluators"
                       / f"safe_aggregation_{VERSION}.json").read_text(encoding="utf-8"))
    body["hash_algorithm"] = "sha256-concat-v0"
    path = repo_root() / "configs/aggregation/evaluators/safe_aggregation_0.0.1.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    try:
        with pytest.raises(ContractError, match="do not mean the same thing"):
            load_evaluator_manifest("0.0.1")
    finally:
        path.unlink()


def test_a_manifest_nobody_updated_fails_readiness(tmp_path):
    """Change the code, forget the version: startup must stop, not warn."""
    body = json.loads((repo_root() / "configs/aggregation/evaluators"
                       / f"safe_aggregation_{VERSION}.json").read_text(encoding="utf-8"))
    body["evaluator_hash"] = "sha256:" + "0" * 64
    path = repo_root() / "configs/aggregation/evaluators/safe_aggregation_0.0.2.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    try:
        with pytest.raises(ContractError, match="without the version"):
            evaluator_readiness("0.0.2", policy_id=POLICY, policy_hash="x")
    finally:
        path.unlink()


# --- what a run may do with it --------------------------------------------

def test_a_clean_checkout_of_a_registered_pair_is_release_eligible():
    identity = evaluator_readiness(VERSION, policy_id=POLICY,
                                   policy_hash=_policy_hash())
    assert identity.status in (REGISTERED, UNREGISTERED)
    if identity.status == REGISTERED:
        assert identity.release_eligible
        assert len(identity.git_commit) == 40
        assert identity.git_commit_matches_evaluator
    else:
        # Working copy differs from HEAD — the development case, which must run
        # and must not be publishable.
        assert not identity.release_eligible
        assert "development only" in identity.reason or "registry" in identity.reason


def test_an_unrelated_dirty_file_does_not_make_the_evaluator_unregistered():
    """A slide deck in the working copy says nothing about which code decided."""
    scratch = repo_root() / "unrelated_scratch_file.txt"
    scratch.write_text("not evaluator source\n", encoding="utf-8")
    try:
        after = evaluator_readiness(VERSION, policy_id=POLICY,
                                    policy_hash=_policy_hash())
        assert "unrelated_scratch_file" not in after.reason
    finally:
        scratch.unlink()


def test_a_policy_the_registry_does_not_pair_with_this_evaluator_is_unregistered():
    identity = evaluator_readiness(VERSION, policy_id="safe_sum_v1",
                                   policy_hash="whatever")
    assert identity.status == UNREGISTERED
    assert not identity.release_eligible
    assert "meant to be used together" in identity.reason


@pytest.mark.parametrize("missing", ["evaluator_version", "evaluator_hash",
                                     "policy_hash", "git_commit"])
def test_an_identity_missing_any_part_is_not_release_eligible(missing):
    body = dict(evaluator_id="safe_aggregation", evaluator_version=VERSION,
                evaluator_hash="sha256:" + "a" * 64, policy_id=POLICY,
                policy_hash="sha256:" + "b" * 64, git_commit="c" * 40,
                git_commit_matches_evaluator=True, status=REGISTERED)
    body[missing] = ""
    assert not EvaluatorIdentity(**body).release_eligible


def test_a_short_commit_is_not_a_commit():
    """Twelve characters identify a commit to a human and not to a checker."""
    identity = EvaluatorIdentity(
        evaluator_id="safe_aggregation", evaluator_version=VERSION,
        evaluator_hash="sha256:" + "a" * 64, policy_id=POLICY,
        policy_hash="sha256:" + "b" * 64, git_commit="c" * 12,
        git_commit_matches_evaluator=True, status=REGISTERED)
    assert not identity.release_eligible


def test_the_boundary_excludes_the_prompt():
    """What is ASKED is versioned by the extraction profile, not by this.

    Folding the prompt in would make every wording change a new evaluator
    version without changing how any response is judged.
    """
    manifest = load_evaluator_manifest(VERSION)
    assert not any("batch_prompt" in path for path in manifest.source_files)
    assert any("safe_aggregation" in path for path in manifest.source_files)
    assert any("batch_parse" in path for path in manifest.source_files)


def test_the_manifest_declares_the_algorithm_it_used():
    assert load_evaluator_manifest(VERSION).hash_algorithm == HASH_ALGORITHM
