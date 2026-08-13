"""Which code decided MATCH — an identity of its own, beside the aggregation one.

`safe_aggregation` versions the code that turns readings into totals. It never
covered the code that decides whether two values agree, and that code can change
a verdict: D1-7.3 added a refusal in `audit/compare.py` under which the same
recording, the same policy and the same evaluator hash produce NOT_COMPARABLE
where they previously produced MATCH.

Folding the comparator into the aggregation evaluator would have closed the hole
and mislabelled it: the comparator is not part of a safe-summation executor, and
every edit to a file that changes often would have forced a new aggregation
version. So it gets its own identity, on the same terms — a manifest listing the
files whose behaviour it is, a hash over them, and a version that only moves
when somebody decides it should.

There is no registry here and no policy to pair with. A registry exists on the
aggregation side to say which POLICY may be applied by which evaluator; the
comparator applies no policy, so its identity is the manifest and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from react_review.contracts import ContractError, read_json_object, repo_root
from react_review.tools.aggregation_identity import hash_sources

COMPARE_ID = "deterministic_compare"
COMPARE_DIR = "configs/compare/evaluators"

REGISTERED = "registered"
UNREGISTERED = "unregistered"


@dataclass(frozen=True)
class CompareIdentity:
    """The comparator that ran, and whether its answers may be published."""

    version: str
    compare_hash: str
    status: str
    source_files: dict
    reason: str = ""

    @property
    def release_eligible(self) -> bool:
        return self.status == REGISTERED

    def describe(self) -> str:
        return (f"{COMPARE_ID} {self.version} ({self.compare_hash[7:19]}…)"
                + ("" if self.release_eligible else f" — {self.status}"))

    def as_dict(self) -> dict:
        return {"compare_id": COMPARE_ID, "compare_version": self.version,
                "compare_hash": self.compare_hash, "status": self.status,
                "release_eligible": self.release_eligible,
                **({"reason": self.reason} if self.reason else {})}


def load_manifest(version: str, root: Path | None = None) -> dict:
    base = root or repo_root()
    path = base / COMPARE_DIR / f"{COMPARE_ID}_{version}.json"
    body = read_json_object(path, kind="compare evaluator manifest")
    if body.get("compare_version") != version:
        raise ContractError(
            f"{path} declares version {body.get('compare_version')!r} and is "
            f"filed as {version!r}")
    return body


def compare_readiness(version: str, root: Path | None = None) -> CompareIdentity:
    """Whether this checkout IS the comparator its manifest describes.

    Unlike the aggregation evaluator this does not raise on a mismatch. The
    comparator runs in every audit, including ones that never aggregate, and a
    development checkout must stay runnable — it simply may not publish. The
    refusal to publish is carried on the result instead of thrown at startup.
    """
    manifest = load_manifest(version, root)
    files = list(manifest.get("source_files") or {})
    digest, per_file = hash_sources(files, root)
    if digest == manifest.get("compare_hash"):
        return CompareIdentity(version, digest, REGISTERED, per_file)
    return CompareIdentity(
        version, digest, UNREGISTERED, per_file,
        reason=(f"the comparator hashes to {digest[7:19]}… and manifest "
                f"{version} publishes {str(manifest.get('compare_hash'))[7:19]}…, "
                "so what decided MATCH here is not the published code"))
