"""Adding the arms up — only when the paper has shown that is the same number.

A total computed from per-arm counts is a different kind of fact from a total the
paper prints, and it is only equal to it under conditions the paper has to
establish: that these are all the groups, that nobody is in two of them, and that
they all count the same people. Any one of those failing turns a sum into a
number describing nobody.

The existing legacy path in ``tools/extract_source`` grants that permission too
easily. It sums when the partition quote is present but not anchored, recording
the gap in a reason nobody gates on — and, more quietly, it sums when there is no
partition quote at all, reporting that as "all explicit, mutually-exclusive arms
were deterministically summed". It also never reads a population per component,
so allocated and analysed counts are indistinguishable to it and can be added
together. That path is frozen for replay and is not touched here; this is the
strict implementation the v5 batch uses, and nothing is shared between them until
the old recordings have proved the two agree.

The division of labour is the point. The model reports what the paper says — the
arm's name, its count, the words for which people, and whether the paper states
these arms partition that population. This module decides whether that is enough,
and only this module does the arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from react_review.contracts import ContractError, read_json_object, repo_root, sha256_file
from react_review.normalize.cohorts import distinguishing_tokens
from react_review.normalize.population import PopulationContract, PopulationScope
from react_review.schemas.batch import BatchAggregationEvidence, BatchCohortCount

#: The four states a sum can end in. They mean what the policy file says they
#: mean, and they are the same four the legacy result schema already uses.
DERIVED = "derived"
REJECTED = "rejected"
PROTOCOL_ERROR = "protocol_error"
NOT_APPLICABLE = "not_applicable"

DEFAULT_POLICY = "configs/aggregation/safe_sum_v1.json"


@dataclass(frozen=True)
class AggregationPolicy:
    """When arithmetic is permitted, as a hashable contract rather than as code."""

    policy_id: str
    sha256: str
    field_types: frozenset[str]
    target_shapes: frozenset[str]
    min_components: int
    require_partition_anchor: bool
    require_complete: bool
    require_mutually_exclusive: bool
    same_population_required: bool
    population_must_match_claim: bool

    def applies_to(self, target_shape: str, field_type: str) -> bool:
        return target_shape in self.target_shapes and field_type in self.field_types


@lru_cache(maxsize=8)
def load_aggregation_policy(path: str | Path = DEFAULT_POLICY) -> AggregationPolicy:
    """Read the frozen policy, recording the bytes it was read from."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repo_root() / resolved
    body = read_json_object(resolved, kind="aggregation policy")
    applies = body.get("applies_to") or {}
    required = body.get("requirements") or {}
    if not applies.get("field_types") or not applies.get("target_shapes"):
        raise ContractError(
            f"{resolved} must say which field types and target shapes it applies "
            "to; a policy that applies to everything is not a policy")
    return AggregationPolicy(
        policy_id=str(body.get("policy_id") or resolved.stem),
        sha256=sha256_file(resolved),
        field_types=frozenset(applies["field_types"]),
        target_shapes=frozenset(applies["target_shapes"]),
        min_components=int(required.get("min_components", 2)),
        require_partition_anchor=bool(required.get("require_partition_anchor", True)),
        require_complete=bool(required.get("require_complete", True)),
        require_mutually_exclusive=bool(
            required.get("require_mutually_exclusive", True)),
        same_population_required=bool(required.get("same_population_required", True)),
        population_must_match_claim=bool(
            required.get("population_must_match_claim", True)),
    )


@dataclass
class AggregationOutcome:
    """What the arithmetic did, or precisely which condition stopped it."""

    status: str = NOT_APPLICABLE
    total: int | None = None
    reason: str = ""
    derivation: str = ""
    components: list[BatchCohortCount] = field(default_factory=list)
    verified_scope: PopulationScope | None = None
    #: The passage that established the partition. This is the only quote a
    #: derived total is entitled to, and it supports the ARITHMETIC, not the
    #: number: no passage of the paper prints a total it never computed.
    partition_quote: str = ""
    policy_id: str = ""
    policy_sha256: str = ""

    @property
    def derived(self) -> bool:
        return self.status == DERIVED and self.total is not None


def derive_partitioned_total(
    evidence: BatchAggregationEvidence | None,
    requested_scope: PopulationScope | None,
    *,
    required_axes: list[str] | None = None,
    policy: AggregationPolicy | None = None,
    population_contract: PopulationContract | None = None,
) -> AggregationOutcome:
    """Sum the components, or say which of the policy's conditions failed.

    Every refusal names the condition rather than the outcome, because "could not
    be derived" is not something a reader can act on and "the paper does not state
    that these three arms are all of them" is.
    """
    rules = policy or load_aggregation_policy()
    out = AggregationOutcome(policy_id=rules.policy_id, policy_sha256=rules.sha256)

    if evidence is None or not evidence.cohort_counts:
        out.status = NOT_APPLICABLE
        out.reason = "the response offered no per-arm counts to add up"
        return out

    components = evidence.cohort_counts
    out.components = list(components)

    arms = {tuple(sorted(distinguishing_tokens(c.arm_label))) for c in components}
    if len(arms) < len(components):
        out.status = PROTOCOL_ERROR
        out.reason = ("the same arm was offered more than once as a component, so "
                      "adding them would count some people twice")
        return out
    if len(components) < rules.min_components:
        out.status = REJECTED
        out.reason = (f"only {len(components)} arm count was offered; a total needs "
                      f"at least {rules.min_components} to be a partition of anything")
        return out
    if any(c.count <= 0 for c in components):
        out.status = PROTOCOL_ERROR
        out.reason = "a component is not a positive whole number of people"
        return out

    # --- one population, and the one the claim asked for --------------------
    scopes = {(c.population or PopulationScope()).basis:
              (c.population or PopulationScope()) for c in components}
    if rules.same_population_required and len(scopes) > 1:
        named = ", ".join(sorted(scopes))
        out.status = REJECTED
        out.reason = (f"the components count different populations ({named}); their "
                      "sum would describe a group of people that never existed")
        return out
    verified = next(iter(scopes.values()))
    out.verified_scope = verified

    if rules.population_must_match_claim and requested_scope is not None:
        from react_review.audit.scope import scope_verdict

        outcome = scope_verdict(requested_scope, verified,
                                required_axes=required_axes or ["population_basis"],
                                contract=population_contract)
        if outcome.status != "ok":
            out.status = REJECTED
            out.reason = (f"the components count the {verified.describe()} "
                          f"population, and the review reports the "
                          f"{requested_scope.describe()} one")
            return out

    # --- the partition itself, which is what makes the sum meaningful -------
    partition = evidence.partition
    if partition is None:
        out.status = REJECTED
        out.reason = ("the response did not say whether these arms are all of them "
                      "or whether they overlap, so nothing licenses adding them")
        return out
    if rules.require_complete and not partition.complete:
        out.status = REJECTED
        out.reason = _with(("the paper is not shown to state that these arms cover "
                            "the whole population"), partition.reason)
        return out
    if rules.require_mutually_exclusive and not partition.mutually_exclusive:
        out.status = REJECTED
        out.reason = _with(("the paper is not shown to state that these arms do not "
                            "overlap"), partition.reason)
        return out
    if rules.require_partition_anchor and not partition.anchored:
        # The condition this policy exists for. A boolean is a claim about the
        # paper; only a locatable passage makes it a reading of one.
        out.status = REJECTED
        out.reason = _with(
            ("the arms were asserted to be complete and non-overlapping, but no "
             "passage of the paper saying so could be located, and an unanchored "
             "assertion is not evidence"), partition.reason)
        return out

    total = sum(c.count for c in components)
    out.partition_quote = partition.quote
    out.status = DERIVED
    out.total = total
    out.derivation = " + ".join(str(c.count) for c in components) + f" = {total}"
    out.reason = (f"{len(components)} arms the paper states are all of the "
                  f"{verified.describe()} population and do not overlap, added by "
                  "deterministic code")
    return out


def _with(head: str, detail: str) -> str:
    return f"{head}: {detail}" if detail else head
