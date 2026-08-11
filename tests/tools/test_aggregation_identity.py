"""What decided, and whether this run may publish what it decided.

A policy hash was never enough. Every wrong total in this phase came from rules
that read correctly and code that did not enforce them, so an artifact naming
only `safe_sum_v5` claims a reproducibility it does not have: two commits of the
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
    CONTROL_PLANE,
    HASH_ALGORITHM,
    REGISTERED,
    UNREGISTERED,
    EvaluatorIdentity,
    evaluator_readiness,
    hash_sources,
    load_evaluator_manifest,
)

VERSION = "1.6.0"
POLICY = "safe_sum_v5"


# --- the hash means what it says ------------------------------------------

PENDING = "configs/aggregation/evaluators/PENDING.json"


def _pending() -> dict | None:
    path = repo_root() / PENDING
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _requires_frozen() -> None:
    """Skip what is only true of a checkout that IS a published evaluator.

    Mid-phase the boundary files are being changed, so readiness refuses — which
    is the gate working. Asserting a frozen checkout's properties here would
    either fail for the right reason or force the gate to be weakened, and the
    gate is the point.
    """
    if _pending() is not None:
        pytest.skip("the evaluator is declared unfrozen in PENDING.json")


def test_the_published_hash_describes_the_tree_or_the_tree_says_why_not():
    """Either this commit IS a published evaluator, or it declares that it is not.

    Mid-phase the tree legitimately matches no published manifest: the boundary
    files are being changed and the new version is not decided until the
    behaviour comparison runs. Regenerating a published manifest at every
    intermediate commit would be editing a published file, which is the failure
    this repository has already recorded three times. So the disagreement is
    declared in PENDING.json, and this test fails if the tree drifts WITHOUT
    that declaration — or if the declaration outlives the drift.
    """
    manifest = load_evaluator_manifest(VERSION)
    digest, per_file = hash_sources(list(manifest.source_files))
    pending = _pending()
    if digest == manifest.evaluator_hash:
        assert per_file == manifest.source_files
        assert pending is None, (
            f"{PENDING} says the evaluator is unfrozen, but the tree matches "
            f"{VERSION}. Freeze it or delete the marker")
        return
    assert pending is not None, (
        f"the tree no longer matches evaluator {VERSION} and nothing says so")
    assert pending["supersedes_version"] == VERSION
    assert pending["version_rule"] and pending["on_freeze"]


def test_nothing_is_release_eligible_while_the_evaluator_is_unfrozen():
    """The marker is not a way to keep publishing."""
    if _pending() is None:
        pytest.skip("the evaluator is frozen; there is nothing to be lenient about")
    with pytest.raises(ContractError, match="without the version"):
        evaluator_readiness(VERSION, policy_id=POLICY)


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
    body["evaluator_version"] = "0.0.1"
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
    body["evaluator_version"] = "0.0.2"
    path = repo_root() / "configs/aggregation/evaluators/safe_aggregation_0.0.2.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    try:
        with pytest.raises(ContractError, match="without the version"):
            evaluator_readiness("0.0.2", policy_id=POLICY)
    finally:
        path.unlink()


# --- what a run may do with it --------------------------------------------

def test_a_clean_checkout_of_a_registered_pair_is_release_eligible():
    _requires_frozen()
    identity = evaluator_readiness(VERSION, policy_id=POLICY)
    assert identity.status in (REGISTERED, UNREGISTERED)
    if identity.status == REGISTERED:
        assert identity.release_eligible
        assert len(identity.git_commit) == 40
        assert identity.git_commit_matches_evaluator
    else:
        # Working copy differs from HEAD — the development case, which must run
        # and must not be publishable.
        assert not identity.release_eligible
        # The development states: files edited since HEAD, or not yet tracked
        # at all. Both must run and neither may publish.
        assert any(x in identity.reason for x in
                   ("development only", "registry", "not tracked"))


def test_an_unrelated_dirty_file_does_not_make_the_evaluator_unregistered():
    """A slide deck in the working copy says nothing about which code decided."""
    _requires_frozen()
    scratch = repo_root() / "unrelated_scratch_file.txt"
    scratch.write_text("not evaluator source\n", encoding="utf-8")
    try:
        after = evaluator_readiness(VERSION, policy_id=POLICY)
        assert "unrelated_scratch_file" not in after.reason
    finally:
        scratch.unlink()


def test_a_policy_the_registry_does_not_pair_with_this_evaluator_is_unregistered():
    _requires_frozen()
    identity = evaluator_readiness(VERSION, policy_id="safe_sum_v1")
    assert identity.status == UNREGISTERED
    assert not identity.release_eligible
    assert "may not produce a publishable result" in identity.reason


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


# --- the checker does its own checking ------------------------------------

def test_readiness_computes_the_policy_hash_instead_of_believing_a_caller():
    """It used to take one as a parameter, and "not-a-hash" passed.

    The one function whose job is to establish identity cannot take the most
    important part of that identity on trust.
    """
    _requires_frozen()
    import inspect

    from react_review.tools.safe_aggregation import load_aggregation_policy

    assert "policy_hash" not in inspect.signature(evaluator_readiness).parameters
    identity = evaluator_readiness(VERSION, policy_id=POLICY)
    assert identity.policy_hash == load_aggregation_policy().sha256
    assert len(identity.policy_hash) == 64


def test_a_policy_the_registry_marks_unpublishable_cannot_publish():
    _requires_frozen()
    identity = evaluator_readiness(VERSION, policy_id="safe_sum_v4")
    assert identity.status == UNREGISTERED and not identity.release_eligible
    assert "may not produce a publishable result" in identity.reason


def test_a_policy_whose_bytes_no_longer_match_the_registry_stops_the_run(tmp_path):
    """A frozen policy that moved is not a warning."""
    _requires_frozen()
    import json
    import shutil

    from react_review.tools.aggregation_identity import REGISTRY, _registry

    scratch = tmp_path / "repo"
    shutil.copytree(repo_root() / "configs", scratch / "configs")
    shutil.copytree(repo_root() / "src", scratch / "src")
    policy = scratch / "configs/aggregation/safe_sum_v5.json"
    body = json.loads(policy.read_text(encoding="utf-8"))
    body["written_on"] = "1999-01-01"
    policy.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8",
                      newline="\n")
    _registry.cache_clear()
    try:
        with pytest.raises(ContractError, match="edited"):
            evaluator_readiness(VERSION, policy_id=POLICY, root=scratch)
    finally:
        _registry.cache_clear()


def test_the_files_that_decide_publishability_are_covered_somewhere():
    """The registry by the clean check; this module by the HASH.

    Being only in the clean check was not enough for code: an uncommitted edit
    was caught, and the same edit committed went back to clean with the
    evaluator hash unchanged. Code that decides publishability belongs inside
    the boundary that is hashed.
    """
    from react_review.tools.aggregation_identity import REGISTRY

    assert REGISTRY in CONTROL_PLANE
    manifest = load_evaluator_manifest(VERSION)
    assert "src/react_review/tools/aggregation_identity.py" in manifest.source_files


def test_a_manifest_named_for_one_version_may_not_declare_another():
    import json

    body = json.loads((repo_root() / "configs/aggregation/evaluators"
                       / f"safe_aggregation_{VERSION}.json").read_text(encoding="utf-8"))
    body["evaluator_version"] = "9.9.9"
    path = repo_root() / "configs/aggregation/evaluators/safe_aggregation_0.0.3.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    try:
        with pytest.raises(ContractError, match="is named for evaluator"):
            load_evaluator_manifest("0.0.3")
    finally:
        path.unlink()


def test_a_dirty_control_plane_file_makes_the_run_unpublishable():
    """A TRANSITION, not a state.

    This test named registry_v2 after the constant had moved to v3, so it dirtied
    a file nothing was watching. It stayed green anyway, because the working copy
    it ran in had uncommitted evaluator sources and the run was already
    unregistered for another reason entirely — the assertion held and measured
    nothing. It could only ever have failed once something was committed, which
    is precisely when it did.

    So it reads the constant rather than a name, and asserts the change: eligible
    before, not eligible after. If the baseline is not eligible there is no
    transition to observe and nothing to claim.
    """
    _requires_frozen()
    for name in CONTROL_PLANE:
        watched = repo_root() / name
        before = evaluator_readiness(VERSION, policy_id=POLICY)
        if not before.release_eligible:
            pytest.skip(f"this checkout is already {before.status}, so dirtying "
                        f"{name} would prove nothing")
        original = watched.read_bytes()
        try:
            watched.write_bytes(original + b"\n")
            after = evaluator_readiness(VERSION, policy_id=POLICY)
            assert not after.release_eligible, name
            assert after.status == UNREGISTERED
            assert "differ from HEAD" in after.reason
        finally:
            watched.write_bytes(original)
        assert evaluator_readiness(VERSION, policy_id=POLICY).release_eligible


# --- D1-4C: the policy that runs is the policy that was cleared -----------

def test_a_runtime_binds_the_policy_to_the_identity_that_cleared_it():
    _requires_frozen()
    from react_review.tools.safe_aggregation import AggregationRuntime

    runtime = AggregationRuntime.resolve(policy_id=POLICY,
                                         evaluator_version=VERSION)
    assert runtime.policy.policy_id == runtime.evaluator.policy_id
    assert runtime.policy.sha256 == runtime.evaluator.policy_hash
    assert runtime.release_eligible == runtime.evaluator.release_eligible


def test_a_runtime_whose_policy_is_not_the_cleared_one_is_not_publishable():
    """The P0 this object exists for.

    A rogue policy with a real identity beside it used to report
    release_eligible=True while the result named the rogue.
    """
    _requires_frozen()
    from dataclasses import replace

    from react_review.tools.safe_aggregation import AggregationRuntime

    good = AggregationRuntime.resolve(policy_id=POLICY, evaluator_version=VERSION)
    # A perfect identity, so that the ONLY thing under test is whether the
    # policy beside it is the one that identity vouches for. Asserting on the
    # real identity would make this test depend on whether the working copy
    # happens to be committed.
    cleared = replace(good, evaluator=replace(
        good.evaluator, status=REGISTERED, git_commit="a" * 40,
        git_commit_matches_evaluator=True, reason=""))
    assert cleared.release_eligible

    rogue = replace(cleared, policy=replace(cleared.policy,
                                            policy_id="rogue_policy",
                                            sha256="0" * 64))
    assert rogue.evaluator.release_eligible      # the identity is still perfect
    assert not rogue.release_eligible            # and it does not vouch for THIS


def test_an_unregistered_runtime_says_so_in_every_field_a_reader_checks():
    from react_review.tools.safe_aggregation import AggregationRuntime

    runtime = AggregationRuntime.unregistered()
    assert not runtime.release_eligible
    assert runtime.evaluator.status == "unavailable"
    assert runtime.evaluator.git_commit == ""
    assert "attributes its verdicts" in runtime.evaluator.reason


def test_readiness_reads_the_manifest_of_the_checkout_it_was_pointed_at(tmp_path):
    """Otherwise it verifies one tree's code against another tree's claim."""
    _requires_frozen()
    import shutil

    scratch = tmp_path / "repo"
    (scratch / "configs").mkdir(parents=True)
    shutil.copytree(repo_root() / "configs/aggregation",
                    scratch / "configs/aggregation")
    shutil.copytree(repo_root() / "src", scratch / "src")
    identity = evaluator_readiness(VERSION, policy_id=POLICY, root=scratch)
    # No .git there, so it cannot be attributed — but it got far enough to hash
    # the copy's own files against the copy's own manifest.
    assert identity.evaluator_hash
    assert not identity.release_eligible
