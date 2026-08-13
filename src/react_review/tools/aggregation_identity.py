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

from react_review.contracts import (
    ContractError,
    read_json_object,
    repo_root,
    sha256_file,
)

#: Named so a future scheme cannot be mistaken for this one in an old artifact.
HASH_ALGORITHM = "sha256-path-lf-v1"

EVALUATOR_ID = "safe_aggregation"
EVALUATOR_DIR = "configs/aggregation/evaluators"
#: The registry is versioned like everything else it governs. It is pinned by
#: hash, so it could never gain a policy or an evaluator pair without breaking
#: its own immutability rule — and changing which pairs may publish is exactly
#: the kind of change that should require a version anyway.
REGISTRY = "configs/aggregation/registry_v6.json"

#: Files that decide WHETHER a result may be published, as opposed to what it
#: says. `aggregation_identity.py` is no longer listed here: it is inside the
#: hashed evaluator boundary now, which is strictly stronger. Being only in the
#: clean check meant an uncommitted change was caught and a COMMITTED one was
#: not — the evaluator hash stayed put and the run went back to clean.
CONTROL_PLANE = ("configs/aggregation/registry_v6.json",)

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
def load_evaluator_manifest(version: str,
                            root: Path | None = None) -> EvaluatorManifest:
    """The manifest for one version, read from the checkout being examined.

    ``root`` is honoured rather than ignored: a readiness check pointed at
    another working copy has to read THAT copy's manifest, or it verifies one
    tree's code against another tree's published claim about it.
    """
    base = root or repo_root()
    path = base / EVALUATOR_DIR / f"safe_aggregation_{version}.json"
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
    # A manifest that does not agree with its own filename identifies the wrong
    # thing: 1.6.0's file saying `evaluator_version: 1.5.0` would publish one
    # version's hash under another's name.
    if str(body.get("evaluator_version") or "") != version:
        raise ContractError(
            f"{path} is named for evaluator {version} and declares "
            f"{body.get('evaluator_version')!r}")
    if str(body.get("evaluator_id") or "") != EVALUATOR_ID:
        raise ContractError(
            f"{path} declares evaluator_id {body.get('evaluator_id')!r}; this "
            f"build is {EVALUATOR_ID!r}")
    return EvaluatorManifest(
        evaluator_id=str(body.get("evaluator_id") or ""),
        evaluator_version=str(body.get("evaluator_version") or ""),
        hash_algorithm=algorithm, source_files=dict(sources),
        evaluator_hash=str(body.get("evaluator_hash") or ""), path=path)


def evaluator_readiness(version: str, *, policy_id: str,
                        root: Path | None = None) -> EvaluatorIdentity:
    """Decide once, at startup, what this run is allowed to produce.

    The policy hash is COMPUTED here, not accepted from the caller. It used to be
    a parameter, which meant the one function whose job is to establish identity
    took the most important part of that identity on trust: handed the string
    "not-a-hash" it reported a registered, release-eligible run. A checker that
    believes what it is told is not a checker.

    Deliberately not per claim. It shells out to git, and a check that runs
    thousands of times is a check somebody eventually removes for being slow.
    """
    base = root or repo_root()
    manifest = load_evaluator_manifest(version, base)
    computed, per_file = hash_sources(list(manifest.source_files), base)
    entry = _registry(base).get(policy_id) or {}
    policy_file = str(entry.get("file") or "")
    policy_hash = (sha256_file(base / policy_file) if policy_file
                   and (base / policy_file).exists() else "")

    def refuse(status: str, reason: str) -> EvaluatorIdentity:
        return EvaluatorIdentity(
            evaluator_id=manifest.evaluator_id, evaluator_version=version,
            evaluator_hash=computed, policy_id=policy_id, policy_hash=policy_hash,
            status=status, reason=reason)

    if not entry:
        return refuse(UNREGISTERED,
                      f"the registry does not list {policy_id!r} at all, so nothing "
                      "says what it is or whether it may be applied")
    if not policy_hash:
        return refuse(UNREGISTERED,
                      f"the registry points {policy_id!r} at {policy_file!r}, which "
                      "is not there")
    if policy_hash != str(entry.get("sha256") or ""):
        raise ContractError(
            f"{policy_file} hashes to {policy_hash[:16]}… and the registry records "
            f"{str(entry.get('sha256') or '')[:16]}…. A frozen policy has been "
            "edited, or the registry was written by hand")
    if entry.get("status") != "active" or entry.get("formal_results") is not True:
        return refuse(UNREGISTERED,
                      f"the registry records {policy_id} as "
                      f"{entry.get('status')!r} with formal_results="
                      f"{entry.get('formal_results')!r}, so it may be replayed but "
                      "may not produce a publishable result")

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

    if version not in (entry.get("evaluators") or []):
        return refuse(UNREGISTERED,
                      f"the registry does not record {policy_id} being applied by "
                      f"evaluator {version}, so nothing says the two were meant "
                      "to be used together")

    # The evaluator's own sources AND everything that decides whether its
    # verdicts may be published. Without the second group, changing which pairs
    # are authorised — or changing this function — left the run looking clean.
    paths = sorted({*manifest.source_files, *CONTROL_PLANE, policy_file,
                    manifest.path.relative_to(base).as_posix()})
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


def registry_entry(policy_id: str, root: Path | None = None) -> dict:
    """What the registry says about one policy — its file, bytes and standing."""
    return dict(_registry(root).get(policy_id) or {})


@lru_cache(maxsize=4)
def _registry(root: Path | None = None) -> dict:
    path = (root or repo_root()) / REGISTRY
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
