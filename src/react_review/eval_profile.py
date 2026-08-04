"""Phase 7 benchmark profiles — an opt-in overlay on a frozen benchmark.

A frozen benchmark has to keep replaying exactly as it was recorded, so nothing
in here is discovered automatically: a profile applies only when a run names its
file. Without that file every path stays on the Phase 6 contract, which is what
keeps an old cache valid and an old replay honest.

Three artifacts, with different authority:

``phase7_profile.json``
    Names the extraction / semantic prompt profiles and pins the SHA-256 of the
    frozen inputs it was written against, plus of the two files below. A hash
    that no longer matches is a hard failure, not a warning: it means the
    profile is describing a benchmark that has since changed.

``phase7_semantic_overlay.csv``
    The Phase 7 semantic expectation contract. It may restate a semantic
    expectation, and NOTHING else — an overlay that could move
    ``expected_label`` would be a rewritten answer key wearing a different name.

``phase7_target_contract.csv``
    Evaluation INPUT: the review-side facts a runner needs to ask the source
    paper a well-formed question (the review's own column label, its own word
    for the cohort, the timepoint, the cell). It carries no expectation of any
    kind, so it cannot leak an answer into the extractor.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

EXTRACTION_PROFILES = ("legacy_v3", "targeted_v4")
SEMANTIC_PROFILES = ("semantic_v1", "semantic_v2_specificity")

# Exactly the columns each contract file may carry. An unexpected column is a
# failure rather than something to ignore: silently dropping a column is how a
# file grows an expectation nobody agreed to.
_OVERLAY_COLUMNS = {
    "audit_id", "expected_semantic_relation", "expected_more_specific_side",
    "expected_review_required", "reason",
}
_TARGET_COLUMNS = {
    "audit_id", "review_data_source", "raw_field_name", "cohort_label",
    "cohort_label_source", "timepoint", "cell_ref",
}

_RELATIONS = {"same", "review_broader", "source_broader", "different", "unknown"}
_SIDES = {"review", "source", "neither", "unknown"}
_TRUE = {"1", "true", "yes", "y"}
_FALSE = {"0", "false", "no", "n", ""}


class ProfileError(ValueError):
    """A profile, overlay or target contract that must not be used as given."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


class SemanticExpectation(BaseModel):
    """What Phase 7 expects of ONE semantic row. Never a label."""

    audit_id: str
    expected_semantic_relation: str
    expected_more_specific_side: str
    expected_review_required: bool = False
    reason: str = ""


class TargetContractRow(BaseModel):
    """Review-side, observable, expectation-free. The question, not the answer."""

    audit_id: str
    review_data_source: str = ""      # the review_ground_truth id this came from
    raw_field_name: str = ""          # the review's OWN column label
    cohort_label: str = ""            # the review's OWN word for the cohort
    cohort_label_source: str = ""     # which review row supplied that word
    timepoint: str = ""
    cell_ref: str = ""


class BenchmarkProfile(BaseModel):
    """A named, hash-pinned Phase 7 contract over a frozen benchmark."""

    path: Path
    sha256: str
    schema_version: int = SCHEMA_VERSION
    extraction_profile: str = "legacy_v3"
    semantic_prompt_profile: str = "semantic_v1"
    semantic_overlay_path: Path | None = None
    semantic_overlay_sha256: str = ""
    target_contract_path: Path | None = None
    target_contract_sha256: str = ""
    semantic: dict[str, SemanticExpectation] = Field(default_factory=dict)
    targets: dict[str, TargetContractRow] = Field(default_factory=dict)

    def target_for(self, audit_id: str) -> TargetContractRow | None:
        return self.targets.get(audit_id)

    def semantic_for(self, audit_id: str) -> SemanticExpectation | None:
        return self.semantic.get(audit_id)

    def provenance(self) -> dict[str, str]:
        """What a run must record so its profile can be reconstructed later."""
        return {
            "benchmark_profile": str(self.path),
            "benchmark_profile_sha256": self.sha256,
            "extraction_profile": self.extraction_profile,
            "semantic_prompt_profile": self.semantic_prompt_profile,
            "semantic_overlay": str(self.semantic_overlay_path or ""),
            "semantic_overlay_sha256": self.semantic_overlay_sha256,
            "target_contract": str(self.target_contract_path or ""),
            "target_contract_sha256": self.target_contract_sha256,
        }


def _read_rows(path: Path, allowed: set[str], kind: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        if not header:
            raise ProfileError(f"{kind} {path.name} has no header row")
        unknown = [h for h in header if h not in allowed]
        if unknown:
            raise ProfileError(
                f"{kind} {path.name} carries columns it is not allowed to "
                f"carry: {', '.join(sorted(unknown))}")
        if "audit_id" not in header:
            raise ProfileError(f"{kind} {path.name} has no audit_id column")
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k}
                for row in reader]


def _checked_ids(rows: list[dict[str, str]], known: set[str], kind: str,
                 path: Path) -> list[str]:
    ids: list[str] = []
    for row in rows:
        audit_id = row.get("audit_id", "")
        if not audit_id:
            raise ProfileError(f"{kind} {path.name} has a row with no audit_id")
        if audit_id in ids:
            raise ProfileError(f"{kind} {path.name} repeats audit_id {audit_id}")
        if audit_id not in known:
            raise ProfileError(
                f"{kind} {path.name} names audit_id {audit_id}, which the "
                "benchmark answer key does not contain")
        ids.append(audit_id)
    return ids


def _as_bool(value: str, *, field: str, audit_id: str) -> bool:
    text = (value or "").strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ProfileError(f"{audit_id}: {field} is not a boolean: {value!r}")


def load_semantic_overlay(
    path: Path, answer_key_ids: Iterable[str],
) -> dict[str, SemanticExpectation]:
    """Load the Phase 7 semantic expectations, refusing anything out of scope."""
    path = Path(path)
    if not path.is_file():
        raise ProfileError(f"semantic overlay does not exist: {path}")
    known = set(answer_key_ids)
    rows = _read_rows(path, _OVERLAY_COLUMNS, "semantic overlay")
    _checked_ids(rows, known, "semantic overlay", path)

    overlay: dict[str, SemanticExpectation] = {}
    for row in rows:
        audit_id = row["audit_id"]
        relation = row.get("expected_semantic_relation", "")
        side = row.get("expected_more_specific_side", "")
        if relation not in _RELATIONS:
            raise ProfileError(
                f"{audit_id}: expected_semantic_relation {relation!r} is not one "
                f"of {sorted(_RELATIONS)}")
        if side not in _SIDES:
            raise ProfileError(
                f"{audit_id}: expected_more_specific_side {side!r} is not one of "
                f"{sorted(_SIDES)}")
        overlay[audit_id] = SemanticExpectation(
            audit_id=audit_id, expected_semantic_relation=relation,
            expected_more_specific_side=side,
            expected_review_required=_as_bool(
                row.get("expected_review_required", ""),
                field="expected_review_required", audit_id=audit_id),
            reason=row.get("reason", ""))
    return overlay


def load_target_contract(
    path: Path, answer_key_ids: Sequence[str],
) -> dict[str, TargetContractRow]:
    """Load the evaluation input contract — one row per answer-key row, exactly.

    One-to-one is enforced in BOTH directions. A missing row would silently send
    the extractor an under-specified question (which is the Phase 6B failure),
    and an extra row would mean the contract is describing something the
    benchmark does not audit.
    """
    path = Path(path)
    if not path.is_file():
        raise ProfileError(f"target contract does not exist: {path}")
    known = list(answer_key_ids)
    rows = _read_rows(path, _TARGET_COLUMNS, "target contract")
    ids = _checked_ids(rows, set(known), "target contract", path)

    missing = [audit_id for audit_id in known if audit_id not in set(ids)]
    if missing:
        raise ProfileError(
            f"target contract {path.name} is missing rows for: "
            f"{', '.join(missing)}")
    return {row["audit_id"]: TargetContractRow(**row) for row in rows}


def load_profile(
    benchmark: Path, spec: str | Path, *, answer_key_ids: Sequence[str],
) -> BenchmarkProfile:
    """Load and verify a profile. Every failure here is explicit and fatal.

    ``spec`` is a file — never a bare profile name. A run selects a whole,
    hash-pinned contract; it cannot assemble one out of loose flags that might
    drift away from the benchmark they were written for.
    """
    benchmark = Path(benchmark)
    path = Path(spec)
    if not path.is_absolute():
        path = benchmark / path
    if not path.is_file():
        raise ProfileError(f"benchmark profile does not exist: {path}")

    try:
        body = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ProfileError(f"benchmark profile is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ProfileError("benchmark profile must be a JSON object")

    version = body.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ProfileError(
            f"benchmark profile schema_version {version!r} is not {SCHEMA_VERSION}")

    extraction = str(body.get("extraction_profile") or "")
    semantic_profile = str(body.get("semantic_prompt_profile") or "")
    if extraction not in EXTRACTION_PROFILES:
        raise ProfileError(
            f"unknown extraction_profile {extraction!r} (known: "
            f"{', '.join(EXTRACTION_PROFILES)})")
    if semantic_profile not in SEMANTIC_PROFILES:
        raise ProfileError(
            f"unknown semantic_prompt_profile {semantic_profile!r} (known: "
            f"{', '.join(SEMANTIC_PROFILES)})")

    # The profile is written against a frozen benchmark. If those inputs have
    # moved, the profile's claims about them are no longer true.
    for key, relative in (("base_manifest_sha256", "manifest.json"),
                          ("base_audit_template_sha256", "audit_template.csv")):
        declared = str(body.get(key) or "").upper()
        target = benchmark / relative
        if not declared:
            raise ProfileError(f"benchmark profile does not declare {key}")
        if not target.is_file():
            raise ProfileError(f"benchmark is missing {relative}")
        actual = sha256_file(target)
        if declared != actual:
            raise ProfileError(
                f"{relative} has changed since this profile was written "
                f"({key}: profile {declared[:16]}…, file {actual[:16]}…)")

    overlay_path = overlay_sha = ""
    semantic: dict[str, SemanticExpectation] = {}
    if body.get("semantic_expectation_overlay"):
        overlay_path = benchmark / str(body["semantic_expectation_overlay"])
        overlay_sha = _verify_declared(
            body, "semantic_expectation_overlay_sha256", overlay_path)
        semantic = load_semantic_overlay(overlay_path, answer_key_ids)

    target_path = target_sha = ""
    targets: dict[str, TargetContractRow] = {}
    if body.get("target_contract"):
        target_path = benchmark / str(body["target_contract"])
        target_sha = _verify_declared(
            body, "target_contract_sha256", target_path)
        targets = load_target_contract(target_path, answer_key_ids)

    return BenchmarkProfile(
        path=path, sha256=sha256_file(path), schema_version=SCHEMA_VERSION,
        extraction_profile=extraction, semantic_prompt_profile=semantic_profile,
        semantic_overlay_path=(overlay_path or None),
        semantic_overlay_sha256=overlay_sha,
        target_contract_path=(target_path or None),
        target_contract_sha256=target_sha,
        semantic=semantic, targets=targets)


def _verify_declared(body: dict, key: str, path: Path) -> str:
    """A contract file the profile points at must also be the one it was hashed
    against — otherwise the profile pins nothing."""
    if not path.is_file():
        raise ProfileError(f"benchmark profile points at a missing file: {path}")
    declared = str(body.get(key) or "").upper()
    if not declared:
        raise ProfileError(f"benchmark profile does not declare {key}")
    actual = sha256_file(path)
    if declared != actual:
        raise ProfileError(
            f"{path.name} does not match the {key} recorded in the profile "
            f"(profile {declared[:16]}…, file {actual[:16]}…)")
    return actual
