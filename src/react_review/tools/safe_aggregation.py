"""Adding the arms up — only when the paper has shown that is the same number.

A total computed from per-arm counts is a different kind of fact from a total the
paper prints, and it is only equal to it under conditions the paper has to
establish: that these are all the groups, that nobody is in two of them, and that
they all count the same people at the same moment. Any one of those failing turns
a sum into a number describing nobody.

The existing legacy path in ``tools/extract_source`` grants that permission too
easily. It sums when the partition quote is present but not anchored, recording
the gap in a reason nobody gates on — and, more quietly, it sums when there is no
partition quote at all, reporting that as "all explicit, mutually-exclusive arms
were deterministically summed". It also never reads a population per component,
so allocated and analysed counts are indistinguishable to it and can be added
together. That path is frozen for replay and is not touched here; this is the
strict implementation the v5 batch uses, and nothing is shared between them until
the old recordings have proved the two agree.

The division of labour is the point, and it is not "the model reads, the code
checks". It is narrower than that: the model may report only what the paper SAYS
— each arm's name and count, the words for which people and when, and what the
paper states about how those arms divide that population. It never chooses which
population answers the claim, and it never adds anything up. Selecting the set
and doing the arithmetic are both here, in code, against a frozen policy.

What makes ``complete`` checkable at all is the arm census. A boolean about a
list the code cannot see is unfalsifiable; "the population was divided into three
groups" is not. So a partition is only honoured when the paper's own count or
names of the arms can be matched against the components actually in hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from react_review.contracts import ContractError, read_json_object, repo_root, sha256_file
from react_review.normalize.cohorts import distinguishing_tokens
from react_review.normalize.population import PopulationContract, PopulationScope
from react_review.schemas.batch import STUDY, AggregationSet, BatchCohortCount

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
    require_arm_census: bool
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
        require_arm_census=bool(required.get("require_arm_census", True)),
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
    chosen_set: AggregationSet | None = None
    verified_scope: PopulationScope | None = None
    #: The passage that established the partition. This is the only quote a
    #: derived total is entitled to, and it supports the ARITHMETIC, not the
    #: number: no passage of the paper prints a total it never computed.
    partition_quote: str = ""
    population_quote: str = ""
    timepoint_quote: str = ""
    policy_id: str = ""
    policy_sha256: str = ""
    #: True when the requested scope selected exactly one set on both population
    #: axes. The projector needs this to know whether a scope that went
    #: unresolved among the printed totals was actually resolved here.
    scope_matched_exactly: bool = False
    timepoint_verified: bool = False

    @property
    def derived(self) -> bool:
        return self.status == DERIVED and self.total is not None


def derive_partitioned_total(
    sets: list[AggregationSet],
    requested_scope: PopulationScope | None,
    *,
    target_shape: str = STUDY,
    field_type: str = "",
    timepoint_label: str = "",
    parse_errors: list[str] | None = None,
    policy: AggregationPolicy | None = None,
    population_contract: PopulationContract | None = None,
) -> AggregationOutcome:
    """Choose the one set that answers this claim, then sum it — or refuse.

    Every refusal names the condition rather than the outcome, because "could not
    be derived" is not something a reader can act on and "the paper does not state
    that these three arms are all of them" is.
    """
    rules = policy or load_aggregation_policy()
    out = AggregationOutcome(policy_id=rules.policy_id, policy_sha256=rules.sha256)

    # The whitelist is enforced HERE, not only in the prompt. A prompt shapes
    # what is asked; a policy decides what is permitted, and a response that
    # offers components for a field nobody may sum must not be summable because
    # it happened to arrive in the right shape.
    if not rules.applies_to(target_shape, field_type):
        out.status = NOT_APPLICABLE
        out.reason = (f"{rules.policy_id} permits deriving a total only for "
                      f"{'/'.join(sorted(rules.field_types))} at "
                      f"{'/'.join(sorted(rules.target_shapes))} level, and this is "
                      f"{field_type or 'an unnamed field'} at {target_shape} level")
        return out

    if parse_errors:
        # A malformed aggregation is a broken answer, not a missing one. Calling
        # it "not applicable" would let a response that tried and failed to
        # describe a partition read exactly like one that never mentioned arms.
        out.status = PROTOCOL_ERROR
        out.reason = "; ".join(parse_errors)
        return out
    if not sets:
        out.status = NOT_APPLICABLE
        out.reason = "the response offered no per-arm counts to add up"
        return out

    chosen, reason, exact = _select_set(sets, requested_scope, timepoint_label,
                                        rules, population_contract)
    if chosen is None:
        out.status = REJECTED
        out.reason = reason
        return out
    out.chosen_set = chosen
    out.components = list(chosen.cohort_counts)
    out.verified_scope = chosen.population
    out.population_quote = chosen.population_quote
    out.timepoint_quote = chosen.timepoint_quote
    out.scope_matched_exactly = exact
    out.timepoint_verified = bool(chosen.timepoint_phrase)

    reason = _partition_holds(chosen, rules)
    if reason:
        out.status = REJECTED
        out.reason = reason
        return out

    total = sum(c.count for c in chosen.cohort_counts)
    out.partition_quote = (chosen.partition.quote if chosen.partition else "")
    out.status = DERIVED
    out.total = total
    out.derivation = " + ".join(str(c.count) for c in chosen.cohort_counts) + \
        f" = {total}"
    out.reason = (f"{len(chosen.cohort_counts)} arms the paper states are all of "
                  f"the {chosen.describe()} population and do not overlap, added "
                  "by deterministic code")
    return out


def _select_set(sets: list[AggregationSet], requested: PopulationScope | None,
                timepoint_label: str, rules: AggregationPolicy,
                contract: PopulationContract | None
                ) -> tuple[AggregationSet | None, str, bool]:
    """Which set answers THIS claim — decided here, never by the model.

    Matching is on BOTH population axes. An analysed/ITT set and an
    analysed/per-protocol set are different people, and comparing only the basis
    would let either answer a claim about the other.
    """
    if requested is None or not rules.population_must_match_claim:
        if len(sets) == 1:
            return sets[0], "", False
        return None, ("the response offers several populations and the claim does "
                      "not say which one it reports, so none of them is its "
                      "answer"), False

    from react_review.audit.scope import scope_verdict

    # Both axes are compared; only one is always REQUIRED. A stated conflict on
    # either — analysed/ITT against analysed/per-protocol — is a mismatch on its
    # own. But demanding that a set name an analysis set when the review never
    # named one would refuse every honest count in the corpus, which is refusing
    # the question rather than answering it safely.
    axes = ["population_basis"]
    if requested.axis_stated("analysis_set"):
        axes.append("analysis_set")
    matches = [s for s in sets
               if scope_verdict(requested, s.population or PopulationScope(),
                                required_axes=axes,
                                contract=contract).status == "ok"]
    if timepoint_label and len(matches) > 1:
        narrowed = [s for s in matches if s.timepoint_phrase
                    and _same_timepoint(s.timepoint_phrase, timepoint_label)]
        if narrowed:
            matches = narrowed
    if not matches:
        offered = ", ".join(sorted(s.describe() for s in sets)) or "nothing"
        return None, (f"the review reports the {requested.describe()} population "
                      f"and the response offers {offered}; an unknown or "
                      "differently-defined analysis set is not the same people"), False
    if len(matches) > 1:
        offered = ", ".join(sorted(s.describe() for s in matches))
        return None, (f"{len(matches)} of the response's sets ({offered}) answer to "
                      f"the {requested.describe()} population equally well, so "
                      "which one the review means is not settled here"), False
    return matches[0], "", True


def _same_timepoint(phrase: str, label: str) -> bool:
    """Words, not spelling — and only to separate sets, never to reject a lone one."""
    wanted, seen = distinguishing_tokens(label), distinguishing_tokens(phrase)
    return bool(wanted and seen and wanted & seen)


def _partition_holds(chosen: AggregationSet, rules: AggregationPolicy) -> str:
    """Whether this set's arms are shown to be all of them, once each."""
    components = chosen.cohort_counts
    if len(components) < rules.min_components:
        return (f"only {len(components)} arm count was offered for the "
                f"{chosen.describe()} population; a total needs at least "
                f"{rules.min_components} to be a partition of anything")

    partition = chosen.partition
    if partition is None:
        return ("the response did not say whether these arms are all of them or "
                "whether they overlap, so nothing licenses adding them")
    if rules.require_complete and not partition.complete:
        return _with("the paper is not shown to state that these arms cover the "
                     "whole population", partition.reason)
    if rules.require_mutually_exclusive and not partition.mutually_exclusive:
        return _with("the paper is not shown to state that these arms do not "
                     "overlap", partition.reason)
    if rules.require_partition_anchor and not partition.anchored:
        return _with("the arms were asserted to be complete and non-overlapping, "
                     "but no passage of the paper saying so could be located, and "
                     "an unanchored assertion is not evidence", partition.reason)

    if not rules.require_arm_census:
        return ""
    # `complete` is the model's opinion about a list this code cannot see. The
    # census is what makes it falsifiable, so without one there is nothing to
    # check and the claim is refused rather than believed.
    if partition.declared_arm_count is None and not partition.declared_arm_labels:
        return _with("the paper's own passage was not shown to say how many groups "
                     "the population was divided into, or which ones, so that "
                     "these are all of them cannot be checked — only taken on "
                     "trust", partition.reason)
    if (partition.declared_arm_count is not None
            and partition.declared_arm_count != len(components)):
        return (f"the paper describes {partition.declared_arm_count} groups and "
                f"{len(components)} arm count(s) were offered, so at least one arm "
                "of this population is missing from the sum")
    if partition.declared_arm_labels:
        have = [tuple(sorted(distinguishing_tokens(c.arm_label)))
                for c in components]
        missing = [label for label in partition.declared_arm_labels
                   if tuple(sorted(distinguishing_tokens(label))) not in have]
        if missing:
            return (f"the paper names {', '.join(partition.declared_arm_labels)} as "
                    f"the groups, and no count was offered for "
                    f"{', '.join(missing)}")
    return ""


def _with(head: str, detail: str) -> str:
    return f"{head}: {detail}" if detail else head
