"""Which CODE decided, not merely which rules it was decided under.

A policy file says what the conditions are. It says nothing about the program
that applies them, and every wrong total found in this phase was a policy that
read correctly and code that did not enforce it. A result that records only
`safe_sum_v3` claims a reproducibility it does not have: the same policy under
two commits of the evaluator produced different answers, and nothing in the
artifact would show which one ran.

So the evaluator gets an identity of its own — a version, the set of files whose
behaviour it is, and a hash over their bytes — and a result records both. The
two move independently and are versioned independently: a policy can tighten
without the code changing, and the code can be corrected without the rules
changing, but neither can change invisibly.

The hash is over PATHS as well as contents, length-delimited and sorted, so that
renaming a file, reordering the list, or shifting a byte from the end of one
file to the start of the next all change it. Bytes are canonicalised to LF
first: a Windows checkout must not produce a different evaluator.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from react_review.contracts import ContractError, read_json_object, repo_root

#: Named so a future scheme cannot be mistaken for this one in an old artifact.
HASH_ALGORITHM = "sha256-path-lf-v1"

EVALUATOR_DIR = "configs/aggregation/evaluators"

#: What a run may do with the evaluator it is running.
REGISTERED = "registered"          # matches a published manifest at a clean HEAD
UNREGISTERED = "unregistered"      # runnable for development; not publishable
UNAVAILABLE = "unavailable"        # no git; offline tests only


@dataclass(frozen=True)
class EvaluatorIdentity:
    """What decided, recorded beside what it decided under."""

    evaluator_id: str = ""
    evaluator_version: str = ""
    evaluator_hash: str = ""
    policy_id: str = ""
    policy_hash: str = ""
    git_commit: str = ""
    git_commit_matches_evaluator: bool = False
    status: str = UNAVAILABLE
    #: Why the identity is not REGISTERED, in words a reader can act on.
    reason: str = ""

    @property
    def release_eligible(self) -> bool:
        """Whether a result carrying this identity may be published.

        Everything must be present AND attributable. A hash with no commit says
        what ran but not where it came from; a commit with uncommitted changes
        to the evaluator says where it came from but not what ran.
        """
        return (self.status == REGISTERED
                and self.git_commit_matches_evaluator
                and len(self.git_commit) == 40
                and bool(self.evaluator_hash) and bool(self.evaluator_version)
                and bool(self.policy_hash))

    def describe(self) -> str:
        return (f"{self.evaluator_id} {self.evaluator_version} "
                f"({self.evaluator_hash[7:19]}…) under {self.policy_id} "
                f"({self.policy_hash[7:19]}…) at {self.git_commit[:12] or '?'}"
                + ("" if self.release_eligible else f" — {self.status}"))


@dataclass
class EvaluatorManifest:
    """The published account of one evaluator version."""

    evaluator_id: str
    evaluator_version: str
    hash_algorithm: str
    source_files: dict[str, str] = field(default_factory=dict)
    evaluator_hash: str = ""
    path: Path | None = None


def hash_sources(paths: list[str], root: Path | None = None) -> tuple[str, dict[str, str]]:
    """Hash a set of files by path and content, order-independently.

    Each file contributes ``path NUL length NUL bytes``. The path is in there so
    that moving code between files changes the hash even when the bytes are the
    same set; the length is in there so that no shifting of bytes across a
    boundary can produce the same stream from a different pair of files.
    """
    base = root or repo_root()
    digest = hashlib.sha256()
    per_file: dict[str, str] = {}
    for relative in sorted(paths):
        body = _canonical_bytes(base / relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(body)).encode("ascii"))
        digest.update(b"\0")
        digest.update(body)
        per_file[relative] = "sha256:" + hashlib.sha256(body).hexdigest()
    return "sha256:" + digest.hexdigest(), per_file


def _canonical_bytes(path: Path) -> bytes:
    """LF, always. A checkout's line endings are not part of what code does."""
    return path.read_bytes().replace(b"\r\n", b"\n")


@lru_cache(maxsize=8)
def load_evaluator_manifest(version: str) -> EvaluatorManifest:
    path = repo_root() / EVALUATOR_DIR / f"safe_aggregation_{version}.json"
    body = read_json_object(path, kind="evaluator manifest")
    algorithm = str(body.get("hash_algorithm") or "")
    if algorithm != HASH_ALGORITHM:
        raise ContractError(
            f"{path} was hashed with {algorithm!r}; this build computes "
            f"{HASH_ALGORITHM!r}, and comparing the two would be comparing "
            "numbers that do not mean the same thing")
    sources = body.get("source_files") or {}
    if not isinstance(sources, dict) or not sources:
        raise ContractError(f"{path} lists no source files, so it identifies nothing")
    return EvaluatorManifest(
        evaluator_id=str(body.get("evaluator_id") or ""),
        evaluator_version=str(body.get("evaluator_version") or ""),
        hash_algorithm=algorithm, source_files=dict(sources),
        evaluator_hash=str(body.get("evaluator_hash") or ""), path=path)


def evaluator_readiness(version: str, *, policy_id: str, policy_hash: str,
                        root: Path | None = None) -> EvaluatorIdentity:
    """Decide once, at startup, what this run is allowed to produce.

    Deliberately not per claim. It shells out to git, and a check that runs
    thousands of times is a check somebody eventually removes for being slow.
    """
    base = root or repo_root()
    manifest = load_evaluator_manifest(version)
    computed, per_file = hash_sources(list(manifest.source_files), base)

    def refuse(status: str, reason: str) -> EvaluatorIdentity:
        return EvaluatorIdentity(
            evaluator_id=manifest.evaluator_id, evaluator_version=version,
            evaluator_hash=computed, policy_id=policy_id, policy_hash=policy_hash,
            status=status, reason=reason)

    if computed != manifest.evaluator_hash:
        # Not a warning. The manifest is the published claim about what this
        # version of the evaluator is, and the code on disk is not it.
        raise ContractError(
            f"evaluator {version} hashes to {computed[:19]}… and its manifest "
            f"publishes {manifest.evaluator_hash[:19]}…. Either the code changed "
            "without the version, or the manifest was written by hand")
    for relative, published in manifest.source_files.items():
        if per_file.get(relative) != published:
            raise ContractError(
                f"{relative} does not match the hash published for evaluator "
                f"{version}")

    if not _registry().get(policy_id, {}).get("evaluators", []).__contains__(version):
        return refuse(UNREGISTERED,
                      f"the registry does not record {policy_id} being applied by "
                      f"evaluator {version}, so nothing says the two were meant "
                      "to be used together")

    paths = sorted(manifest.source_files)
    tracked, error = _git(["ls-files", "--error-unmatch", *paths], base)
    if tracked is None:
        return refuse(UNAVAILABLE,
                      f"git is not available here, so what ran cannot be tied to a "
                      f"commit ({error})")
    if error:
        return refuse(UNREGISTERED,
                      "some evaluator source files are not tracked by git, so no "
                      "commit describes them")

    commit, error = _git(["rev-parse", "HEAD"], base)
    if commit is None or len(commit.strip()) != 40:
        return refuse(UNAVAILABLE, f"no commit could be resolved ({error})")
    commit = commit.strip()

    # ONLY the evaluator's own files. An untracked slide deck in the working
    # copy says nothing about which code decided.
    _, dirty = _git(["diff", "--quiet", "HEAD", "--", *paths], base,
                    expect_status=True)
    if dirty:
        return EvaluatorIdentity(
            evaluator_id=manifest.evaluator_id, evaluator_version=version,
            evaluator_hash=computed, policy_id=policy_id, policy_hash=policy_hash,
            git_commit=commit, git_commit_matches_evaluator=False,
            status=UNREGISTERED,
            reason=("evaluator source files differ from HEAD; this run is "
                    "development only and its results are not publishable"))

    return EvaluatorIdentity(
        evaluator_id=manifest.evaluator_id, evaluator_version=version,
        evaluator_hash=computed, policy_id=policy_id, policy_hash=policy_hash,
        git_commit=commit, git_commit_matches_evaluator=True, status=REGISTERED)


@lru_cache(maxsize=1)
def _registry() -> dict:
    path = repo_root() / "configs/aggregation/registry.json"
    body = read_json_object(path, kind="aggregation registry")
    return {str(k): v for k, v in (body.get("policies") or {}).items()}


def _git(args: list[str], root: Path, *, expect_status: bool = False):
    try:
        done = subprocess.run(["git", *args], cwd=root, capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)[:120]
    if expect_status:
        return done.stdout, done.returncode != 0
    return done.stdout, (done.stderr.strip() if done.returncode else "")
