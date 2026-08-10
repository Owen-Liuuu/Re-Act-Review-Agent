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
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

from react_review.contracts import (
    ContractError,
    read_json_object,
    repo_root,
    sha256_file,
    verify_declared_hash,
)
from react_review.run_profile import (
    RunContractProfile,
    legacy_contract,
    load_run_contract,
)

SCHEMA_VERSION = 1
SCHEMA_VERSIONS = (1, 2)

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
    # v2: what population the REVIEW's own column says this cell counts
    # ("allocated", "analysed/itt"). Still a review-side observable and still
    # no expectation: it says what the review claims to report, never what the
    # source ought to say.
    "population_scope", "population_scope_source",
}

# The GOLD file. Separate from the target contract on purpose: the contract is
# an input the extractor is allowed to see, and this is what the extractor is
# graded against. One file that could be both is one file that could leak.
_GOLD_COLUMNS = {
    "audit_id", "expected_source_target_id", "expected_population_scope", "reason",
}

_RELATIONS = {"same", "review_broader", "source_broader", "different", "unknown"}
_SIDES = {"review", "source", "neither", "unknown"}
_TRUE = {"1", "true", "yes", "y"}
_FALSE = {"0", "false", "no", "n", ""}


#: Kept under its original name; the rules now live with the other contract
#: loaders so a benchmark and a run cannot verify a hash by different standards.
ProfileError = ContractError


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
    population_scope: str = ""          # "" | basis | basis/analysis_set
    population_scope_source: str = ""   # which review words say so

    def scope(self):
        """The declared population, refused if it names an undefined value."""
        from react_review.normalize.population import PopulationScope

        return PopulationScope.parse(self.population_scope, source="contract")


class TargetGoldRow(BaseModel):
    """What a human says the source evidence should have been about.

    Never handed to the extractor — the audit's own ``target_check`` says what
    its guard decided, which is a different question from whether the guard was
    right. Grading a system with its own verdict is not grading.
    """

    audit_id: str
    expected_source_target_id: str = ""     # the paper's own label, or "A vs B"
    expected_population_scope: str = ""     # basis | basis/analysis_set
    reason: str = ""

    @property
    def identity_assessable(self) -> bool:
        return bool(self.expected_source_target_id.strip())

    @property
    def scope_assessable(self) -> bool:
        return bool(self.expected_population_scope.strip())


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
    # The RUNTIME contract this benchmark runs under. A v2 profile names one and
    # pins its hash; a v1 profile predates the idea, so one is reconstructed
    # from what it did declare — never inherited from whatever the current
    # production default happens to be, which is how a frozen benchmark would
    # quietly start replaying under new rules.
    run_contract: RunContractProfile | None = None
    target_gold_path: Path | None = None
    target_gold_sha256: str = ""
    gold: dict[str, TargetGoldRow] = Field(default_factory=dict)

    def gold_for(self, audit_id: str) -> TargetGoldRow | None:
        return self.gold.get(audit_id)

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
            **{f"run_{k}": str(v) for k, v in
               (self.run_contract.identity() if self.run_contract else {}).items()},
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


def load_target_gold(
    path: Path, answer_key_ids: Iterable[str],
) -> dict[str, TargetGoldRow]:
    """Load the human gold for target identity and population scope."""
    path = Path(path)
    if not path.is_file():
        raise ProfileError(f"target gold does not exist: {path}")
    rows = _read_rows(path, _GOLD_COLUMNS, "target gold")
    _checked_ids(rows, set(answer_key_ids), "target gold", path)
    from react_review.normalize.population import PopulationScope

    gold: dict[str, TargetGoldRow] = {}
    for row in rows:
        # Validated here so an undefined population can never sit in a gold file
        # waiting to be compared against something.
        PopulationScope.parse(row.get("expected_population_scope", ""))
        gold[row["audit_id"]] = TargetGoldRow(**row)
    return gold


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
    body = read_json_object(path, kind="benchmark profile")

    version = body.get("schema_version")
    if version not in SCHEMA_VERSIONS:
        raise ProfileError(
            f"benchmark profile schema_version {version!r} is not one of "
            f"{', '.join(str(v) for v in SCHEMA_VERSIONS)}")

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

    run_contract = _run_contract_for(body, path, version, extraction, semantic_profile)

    gold_path = gold_sha = ""
    gold: dict[str, TargetGoldRow] = {}
    if body.get("target_gold"):
        gold_path = benchmark / str(body["target_gold"])
        gold_sha = _verify_declared(body, "target_gold_sha256", gold_path)
        gold = load_target_gold(gold_path, answer_key_ids)

    return BenchmarkProfile(
        path=path, sha256=sha256_file(path), schema_version=version,
        extraction_profile=extraction, semantic_prompt_profile=semantic_profile,
        semantic_overlay_path=(overlay_path or None),
        semantic_overlay_sha256=overlay_sha,
        target_contract_path=(target_path or None),
        target_contract_sha256=target_sha,
        semantic=semantic, targets=targets, run_contract=run_contract,
        target_gold_path=(gold_path or None), target_gold_sha256=gold_sha, gold=gold)


def _run_contract_for(body: dict, path: Path, version: int,
                      extraction: str, semantic_profile: str) -> RunContractProfile:
    """The runtime contract this benchmark profile runs under.

    A v2 profile names a run-profile FILE and pins its hash, and its own
    extraction/semantic declarations must agree with it — two sources of truth
    that could disagree are worse than one. A v1 profile predates all of this,
    so the axes it never declared are filled with what it actually used:
    the comparator's own tolerances, no population contract, no scope check, and
    a context that came only from the command line. Nothing is inherited from
    the current production default.
    """
    if not body.get("run_profile"):
        if version != 1:
            raise ProfileError(
                "a schema_version 2 benchmark profile must name a run_profile")
        return legacy_contract(extraction_profile=extraction,
                               semantic_prompt_profile=semantic_profile,
                               source=f"{path.stem}:v1-compatibility")

    run_path = Path(str(body["run_profile"]))
    if not run_path.is_absolute():
        # Resolved against the repository root, never against the benchmark
        # directory: run profiles live with the code, and a frozen benchmark
        # must not be able to redefine production rules by shipping its own.
        run_path = repo_root() / run_path
    verify_declared_hash(body, "run_profile_sha256", run_path,
                         kind="run contract profile")
    contract = load_run_contract(run_path)
    for field, declared, actual in (
            ("extraction_profile", extraction, contract.extraction_profile),
            ("semantic_prompt_profile", semantic_profile,
             contract.semantic_prompt_profile)):
        if declared != actual:
            raise ProfileError(
                f"the benchmark profile declares {field}={declared!r} but its "
                f"run contract {contract.profile_id!r} says {actual!r}")
    return contract


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
