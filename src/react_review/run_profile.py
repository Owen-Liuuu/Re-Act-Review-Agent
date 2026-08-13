"""What a run was decided by, and how it was executed — kept apart.

Two things were tangled together and had to be separated before either could be
relied on:

*The contract* is what determines the ANSWER — which prompt versions were used,
which tolerances applied, how a population is classified, whether a research
context may fall back to the parsed one. Two runs under the same contract are
asking the same question. It lives in a file, it is hash-pinned, and a command
line may not quietly override any of it: a flag that silently changed a
tolerance would make every recorded result unattributable.

*The execution mode* is how the model was reached — live, recorded, or replayed,
and from which cache. It changes cost and reproducibility, never the answer, and
it MUST be free to vary: the same contract has to be recorded once and replayed
afterwards, so binding a cache mode into the contract would make that
impossible.

*The manifest* is what actually happened: the resolved values of both, written
next to the results so a reader never has to guess which rules produced them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from react_review.contracts import (
    ContractError,
    one_of,
    read_json_object,
    sha256_file,
    verify_declared_hash,
)
from react_review.schemas.run_manifest import (
    EXTRACTION_MODES,
    SEMANTIC_MODES,
    ExecutionMode,
    RunManifest,
)
from react_review.tools.extraction_profile import PROMPT_VERSIONS as EXTRACTION_PROFILES
from react_review.tools.semantic_compare import PROMPT_VERSIONS as SEMANTIC_PROFILES

#: v1 names ONE extraction profile. v2 names a route per claim kind, because a
#: run can legitimately read values in batch and arm identities one at a time —
#: and a run that does both while declaring one profile is a recording nobody
#: can interpret. Both versions load; only v2 can express a mixed contract.
CONTRACT_SCHEMA_VERSION = 2
SUPPORTED_CONTRACT_VERSIONS = (1, 2)

#: The kinds of claim a route can be declared for. Closed, so a contract cannot
#: name a route for something nothing dispatches on.
CLAIM_KINDS = ("value", "arm_identity")

CONTEXT_POLICIES = ("cli_only", "cli_then_parsed")
SCOPE_POLICIES = ("off", "on")
# The axes a scope check can require. Deliberately small and closed: a contract
# may say which axes matter for a field, not invent a new dimension.
SCOPE_AXES = ("population_basis", "analysis_set")


class RunContractProfile(BaseModel):
    """The rules that decide the answer. Hash-pinned, never overridden by flags."""

    profile_id: str
    path: Path
    sha256: str
    schema_version: int = CONTRACT_SCHEMA_VERSION
    #: Which extraction contract each kind of claim runs under. A v1 file gives
    #: every kind the same one; a v2 file may differ, and then the difference is
    #: a declared fact rather than a branch hidden in the Collector.
    extraction_routes: dict[str, str] = Field(default_factory=dict)
    #: Which aggregation policy and evaluator this run must resolve. Required by
    #: v2 whenever a route batches, because a batched read can derive a total and
    #: a derived total is only as good as the identity that cleared it.
    aggregation_policy_id: str = ""
    evaluator_version: str = ""
    #: Which comparator decided MATCH. Separate from `evaluator_version`: the
    #: aggregation executor and the comparison code are different identities,
    #: and one field for both would let a change in either ride on the other.
    compare_version: str = ""
    semantic_prompt_profile: str = "semantic_v1"
    # None means "the comparator's own defaults" — the Phase 6/7 behaviour. A
    # contract that names a file pins its hash; nothing loads a tolerance table
    # that no contract points at.
    tolerances_path: Path | None = None
    tolerances_sha256: str = ""
    population_contract_path: Path | None = None
    population_contract_sha256: str = ""
    context_policy: str = "cli_only"
    scope_policy: str = "off"
    required_scope_axes: dict[str, list[str]] = Field(default_factory=dict)
    # True when this profile was synthesised for a contract file written before
    # these axes existed, rather than read from one that declares them.
    derived_from_legacy: bool = False

    @property
    def extraction_profile(self) -> str:
        """The value route — what a single-profile caller has always meant.

        Kept as a read-only view so v1 callers keep working, and deliberately
        NOT a stored field: two places to say which profile is in force is how a
        run comes to be recorded under one and executed under another.
        """
        return self.route_for("value")

    def route_for(self, claim_kind: str) -> str:
        try:
            return self.extraction_routes[claim_kind or "value"]
        except KeyError:
            raise ContractError(
                f"{self.profile_id} declares no extraction route for "
                f"{claim_kind!r}. A claim nobody routed is a claim that would be "
                "read under whatever profile happened to be lying around") from None

    @property
    def batching(self) -> bool:
        return any(r == "targeted_v5_batch" for r in self.extraction_routes.values())

    @property
    def scope_enabled(self) -> bool:
        return self.scope_policy == "on"

    def axes_for(self, field_type: str) -> list[str]:
        return list(self.required_scope_axes.get((field_type or "").lower(), []))

    def identity(self) -> dict[str, Any]:
        """Everything about the contract that a result must be attributable to."""
        body = {
            "profile_id": self.profile_id,
            "profile_path": str(self.path),
            "profile_sha256": self.sha256,
            "extraction_profile": self.extraction_profile,
            "semantic_prompt_profile": self.semantic_prompt_profile,
            "tolerances": str(self.tolerances_path or ""),
            "tolerances_sha256": self.tolerances_sha256,
            "population_contract": str(self.population_contract_path or ""),
            "population_contract_sha256": self.population_contract_sha256,
            "context_policy": self.context_policy,
            "scope_policy": self.scope_policy,
            "derived_from_legacy": self.derived_from_legacy,
        }
        if self.schema_version >= 2:
            # Only a contract that HAS routes records them. A v1 run gaining an
            # `extraction_routes` key would rewrite every manifest already
            # written, for a fact it did not have.
            body["extraction_routes"] = dict(sorted(self.extraction_routes.items()))
            body["aggregation_policy_id"] = self.aggregation_policy_id
            body["evaluator_version"] = self.evaluator_version
            if self.compare_version:
                body["compare_version"] = self.compare_version
        return body


def load_run_contract(path: Path | str) -> RunContractProfile:
    """Load and verify a runtime contract file."""
    path = Path(path)
    body = read_json_object(path, kind="run contract profile")

    version = body.get("schema_version")
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        raise ContractError(
            f"run contract schema_version {version!r} is not one of "
            f"{SUPPORTED_CONTRACT_VERSIONS}")
    routes, policy_id, evaluator_version = _routes(body, version, path)
    semantic = one_of(body.get("semantic_prompt_profile"),
                      tuple(sorted(SEMANTIC_PROFILES)), field="semantic_prompt_profile")
    context_policy = one_of(body.get("context_policy"), CONTEXT_POLICIES,
                            field="context_policy")
    scope_policy = one_of(body.get("scope_policy"), SCOPE_POLICIES, field="scope_policy")

    tolerances_path = tolerances_sha = ""
    if body.get("tolerances"):
        tolerances_path = _resolve(path.parent, str(body["tolerances"]))
        tolerances_sha = verify_declared_hash(body, "tolerances_sha256",
                                              tolerances_path, kind="tolerance table")
    population_path = population_sha = ""
    if body.get("population_contract"):
        population_path = _resolve(path.parent, str(body["population_contract"]))
        population_sha = verify_declared_hash(body, "population_contract_sha256",
                                              population_path,
                                              kind="population contract")

    axes = _scope_axes(body.get("required_scope_axes"))
    if scope_policy == "on" and not axes:
        raise ContractError(
            "a contract that enables the scope check must say which axes each "
            "field requires (required_scope_axes)")

    return RunContractProfile(
        profile_id=str(body.get("profile_id") or path.stem),
        path=path, sha256=sha256_file(path),
        schema_version=version,
        extraction_routes=routes, aggregation_policy_id=policy_id,
        evaluator_version=evaluator_version,
        compare_version=str(body.get("compare_version") or ""),
        semantic_prompt_profile=semantic,
        tolerances_path=(tolerances_path or None), tolerances_sha256=tolerances_sha,
        population_contract_path=(population_path or None),
        population_contract_sha256=population_sha,
        context_policy=context_policy, scope_policy=scope_policy,
        required_scope_axes=axes)


def _routes(body: dict, version: int, path: Path) -> tuple[dict[str, str], str, str]:
    """Which profile reads which kind of claim, for either schema version.

    A v1 file names one profile and means it for everything — that is what it
    always meant, so it is expanded rather than reinterpreted. A v2 file names
    each kind, and must name every kind: a claim nobody routed would be read
    under whatever profile happened to be lying around, which is the mixed
    recording this schema exists to make impossible.
    """
    known = tuple(sorted(EXTRACTION_PROFILES))
    if version == 1:
        if body.get("extraction_routes"):
            raise ContractError(
                f"{path} declares schema_version 1 and extraction_routes. v1 has "
                "one profile; declaring both leaves two places saying which is in "
                "force")
        one = one_of(body.get("extraction_profile"), known, field="extraction_profile")
        return {kind: one for kind in CLAIM_KINDS}, "", ""

    if "extraction_profile" in body:
        raise ContractError(
            f"{path} declares schema_version 2 and extraction_profile. v2 routes "
            "by claim kind, and a leftover single profile is a second source of "
            "truth about what ran")
    declared = body.get("extraction_routes")
    if not isinstance(declared, dict):
        raise ContractError(f"{path} must declare extraction_routes")
    missing = [k for k in CLAIM_KINDS if k not in declared]
    if missing:
        raise ContractError(
            f"{path} declares no extraction route for {missing}. Every kind of "
            "claim must be routed, or one of them is read under a profile the "
            "contract never named")
    unknown = [k for k in declared if k not in CLAIM_KINDS]
    if unknown:
        raise ContractError(
            f"{path} routes {unknown}, which nothing dispatches on "
            f"(known: {', '.join(CLAIM_KINDS)})")
    routes = {kind: one_of(declared[kind], known, field=f"extraction_routes.{kind}")
              for kind in CLAIM_KINDS}

    batching = any(r == "targeted_v5_batch" for r in routes.values())
    policy_id = str(body.get("aggregation_policy_id") or "")
    evaluator_version = str(body.get("evaluator_version") or "")
    if batching and not (policy_id and evaluator_version):
        raise ContractError(
            f"{path} routes a claim kind to targeted_v5_batch without naming an "
            "aggregation_policy_id and evaluator_version. A batched read can "
            "derive a total, and a derived total is worth exactly as much as the "
            "identity that cleared the rules it was derived under")
    if not batching and (policy_id or evaluator_version):
        raise ContractError(
            f"{path} names an aggregation policy or evaluator but routes nothing "
            "to a batch. Recording an identity nothing used would attribute a run "
            "to code that never ran")
    return routes, policy_id, evaluator_version


def legacy_contract(*, extraction_profile: str = "legacy_v3",
                    semantic_prompt_profile: str = "semantic_v1",
                    source: str = "legacy") -> RunContractProfile:
    """The contract a file written before these axes existed was really using.

    Phase 6 and Phase 7 recordings were made with the comparator's own
    tolerances, no population contract and no scope check. Anything that
    reconstructs them must say so explicitly rather than inherit whatever the
    current default happens to be — that inheritance is precisely how a frozen
    benchmark would silently start replaying under new rules.
    """
    return RunContractProfile(
        profile_id=source, path=Path(source), sha256="",
        schema_version=1,
        extraction_routes={kind: extraction_profile for kind in CLAIM_KINDS},
        semantic_prompt_profile=semantic_prompt_profile,
        tolerances_path=None, population_contract_path=None,
        context_policy="cli_only", scope_policy="off",
        required_scope_axes={}, derived_from_legacy=True)


#: Flags that would change the ANSWER. Passing one alongside a contract that
#: speaks to it is refused rather than resolved by precedence: a silent winner
#: makes results unattributable in a way no log line repairs.
CONTRACT_FLAGS = {
    "--tolerances": "tolerances",
    "--extraction-profile": "extraction_profile",
    "--semantic-profile": "semantic_prompt_profile",
    "--population-contract": "population_contract",
    "--scope": "scope_policy",
}


def guard_contract_overrides(contract: RunContractProfile,
                             supplied: dict[str, Any]) -> None:
    """Refuse a flag that would override what the contract already decides.

    ``--extraction`` and ``--semantic`` are NOT here: they select how the model
    is reached, not what the answer is, and the same contract must be recordable
    and then replayable.
    """
    declared = {
        "tolerances": contract.tolerances_path is not None,
        "extraction_profile": True,
        "semantic_prompt_profile": True,
        "population_contract": contract.population_contract_path is not None,
        "scope_policy": True,
    }
    for flag, field in CONTRACT_FLAGS.items():
        if supplied.get(flag) in (None, "", False):
            continue
        if declared.get(field):
            raise ContractError(
                f"{flag} would override {field}, which the run contract "
                f"{contract.profile_id!r} already fixes. Write a new contract "
                "profile instead of overriding this one on the command line.")


def _resolve(base: Path, relative: str) -> Path:
    """Relative to the profile's own directory, so a profile travels intact."""
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _scope_axes(raw: object) -> dict[str, list[str]]:
    if raw in (None, "", {}):
        return {}
    if not isinstance(raw, dict):
        raise ContractError("required_scope_axes must be a mapping")
    axes: dict[str, list[str]] = {}
    for field, value in raw.items():
        if not isinstance(value, list) or not value:
            raise ContractError(
                f"required_scope_axes[{field}] must be a non-empty list")
        for axis in value:
            one_of(axis, SCOPE_AXES, field=f"scope axis for {field}")
        axes[str(field).strip().lower()] = [str(a) for a in value]
    return axes
