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

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from react_review.contracts import (
    ContractError,
    one_of,
    read_json_object,
    repo_root,
    sha256_file,
)
from react_review.normalize.cohorts import distinguishing_tokens
from react_review.tools.aggregation_identity import EvaluatorIdentity
from react_review.normalize.population import PopulationContract, PopulationScope
from react_review.schemas.batch import (
    STUDY,
    AggregationSet,
    BatchCohortCount,
    RejectedAggregationSet,
)

#: The four states a sum can end in. They mean what the policy file says they
#: mean, and they are the same four the legacy result schema already uses.
DERIVED = "derived"
REJECTED = "rejected"
PROTOCOL_ERROR = "protocol_error"
NOT_APPLICABLE = "not_applicable"

DEFAULT_POLICY = "configs/aggregation/safe_sum_v4.json"


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
    require_distinct_arms: bool
    honour_run_contract_axes: bool
    require_axes_the_claim_states: bool
    population_must_match_claim: bool
    timepoint_must_match_claim: bool
    #: The weakest set of axes this policy will ever accept. A run profile may
    #: add to it; nothing may take from it, so a derived total is never held to
    #: a looser standard than a printed one in the same run.
    minimum_axes: tuple[str, ...]
    on_unknown: str
    timepoint_matching: str

    def applies_to(self, target_shape: str, field_type: str) -> bool:
        return target_shape in self.target_shapes and field_type in self.field_types

    def axes_for(self, required_axes: list[str] | None,
                 claim: PopulationScope | None = None) -> list[str]:
        """This policy's floor, raised by the run contract AND by the claim.

        The claim is the third source and the easily forgotten one. A review
        reporting the ITT population has named an axis; a set that never says
        which analysis set it counts is unknown on that axis, and unknown is
        refused — whether or not the profile happened to list it.
        """
        axes = set(self.minimum_axes)
        if self.honour_run_contract_axes:
            axes |= set(required_axes or [])
        if self.require_axes_the_claim_states and claim is not None:
            axes |= {name for name in ("population_basis", "analysis_set")
                     if claim.axis_stated(name)}
        return sorted(axes)


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
    # An invariant is not a switch. These describe what the code always does, so
    # a policy that claims to turn one off is refused rather than quietly
    # ignored — a file that appears to control behaviour it does not control is
    # worse than a file that says nothing.
    declared = {k: v for k, v in (body.get("invariants") or {}).items()
                if not k.startswith("_")}
    for name, value in declared.items():
        if value is not True:
            raise ContractError(
                f"{resolved} sets the invariant {name!r} to {value!r}. It is not a "
                "switch: the code enforces it unconditionally, and a policy that "
                "reads as though it could disable it would be describing behaviour "
                "nobody implements")
    if set(declared) != _INVARIANTS:
        missing = sorted(_INVARIANTS - set(declared))
        invented = sorted(set(declared) - _INVARIANTS)
        raise ContractError(
            f"{resolved} does not describe this evaluator: "
            + (f"missing {missing}" if missing else "")
            + ("; " if missing and invented else "")
            + (f"invented {invented}" if invented else "")
            + ". The invariant list is the file's account of what the code always "
              "does, so a name nobody implements is a promise nobody keeps, and an "
              "omission hides one that is being kept")
    for name, value in required.items():
        if name != "min_components" and type(value) is not bool:
            raise ContractError(
                f"{resolved} sets the requirement {name!r} to {value!r}. Only a "
                "JSON true or false says anything here: 0, 1 and the strings "
                '"true"/"false" all read as true under coercion, which would turn '
                "a rule its author switched off into one they switched on")
    unread = set(required) - _READ_REQUIREMENTS
    if unread:
        raise ContractError(
            f"{resolved} declares requirement(s) {sorted(unread)} that nothing "
            "reads. A rule the loader ignores is documentation pretending to be a "
            "contract: setting it to false would load cleanly and change nothing. "
            "Move it to `invariants` if the code always enforces it, or read it")
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
        require_distinct_arms=bool(required.get("require_distinct_arms", True)),
        honour_run_contract_axes=bool(
            required.get("honour_run_contract_axes", True)),
        require_axes_the_claim_states=bool(
            required.get("require_axes_the_claim_states", True)),
        population_must_match_claim=bool(
            required.get("population_must_match_claim", True)),
        timepoint_must_match_claim=bool(
            required.get("timepoint_must_match_claim", True)),
        minimum_axes=tuple(body.get("minimum_axes") or ["population_basis"]),
        on_unknown=one_of(str(body.get("on_unknown", "reject")),
                          ("reject",), field="on_unknown"),
        timepoint_matching=one_of(str(body.get("timepoint_matching",
                                               "normalised_exact")),
                                  ("normalised_exact",), field="timepoint_matching"),
    )


#: Every key `load_aggregation_policy` actually consumes. Kept beside the loader
#: so adding a rule to the JSON without reading it fails loudly.
_READ_REQUIREMENTS = {
    "min_components", "require_partition_anchor", "require_complete",
    "require_mutually_exclusive", "require_arm_census", "require_distinct_arms",
    "population_must_match_claim", "timepoint_must_match_claim",
    "require_partition_axis_binding", "require_explicit_census_grammar",
    "honour_run_contract_axes", "require_axes_the_claim_states",
}

#: Exactly what this evaluator enforces unconditionally. A policy must list all
#: of these and nothing else: an omission hides a rule that is in force, and an
#: invention describes one that is not.
_INVARIANTS = {
    "component_value_anchored", "component_label_anchored",
    "set_population_anchored", "set_timepoint_anchored",
    "match_on_both_population_axes", "never_derive_over_explicit_contradiction",
    "model_never_computes", "census_read_from_partition_passage",
    "census_matches_components_exactly", "component_bound_to_set_population",
    "component_bound_to_set_timepoint", "partition_describes_the_sets_population",
    "census_number_must_qualify_a_group_noun", "partition_bound_per_axis",
    "census_matches_explicit_grammar", "aggregation_uses_run_contract_axes",
    "rejected_set_cleared_only_by_definite_mismatch",
}


@dataclass
class AggregationOutcome:
    """What the arithmetic did, or precisely which condition stopped it."""

    status: str = NOT_APPLICABLE
    total: int | None = None
    reason: str = ""
    derivation: str = ""
    components: list[BatchCohortCount] = field(default_factory=list)
    #: The axes actually applied, so a refusal can be read against what was asked.
    required_axes: list[str] = field(default_factory=list)
    #: What decided, not only what it decided under. Filled by the caller, which
    #: is the only layer that knows whether this run is registered.
    evaluator: EvaluatorIdentity | None = None
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
    #: Broken sets that described OTHER people. They cost this claim nothing, but
    #: they happened, and a reader deciding whether to trust the run should see
    #: that part of the response could not be read.
    unrelated_rejections: list[str] = field(default_factory=list)

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
    required_axes: list[str] | None = None,
    rejected_sets: list[RejectedAggregationSet] | None = None,
    evaluator: EvaluatorIdentity | None = None,
    policy: AggregationPolicy | None = None,
    population_contract: PopulationContract | None = None,
) -> AggregationOutcome:
    """Choose the one set that answers this claim, then sum it — or refuse.

    Every refusal names the condition rather than the outcome, because "could not
    be derived" is not something a reader can act on and "the paper does not state
    that these three arms are all of them" is.
    """
    rules = policy or load_aggregation_policy()
    axes = rules.axes_for(required_axes, requested_scope)
    out = AggregationOutcome(policy_id=rules.policy_id, policy_sha256=rules.sha256,
                             required_axes=list(axes), evaluator=evaluator)

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

    # A broken set costs this claim only what it was about. One that described
    # another population never had this claim's answer in it; one that described
    # THIS population, or whose population could not be read at all, might have —
    # and then no surviving set can be shown to be the only candidate, which is
    # the whole basis on which a set is allowed to answer.
    blocking, aside = _split_rejected(rejected_sets or [], requested_scope,
                                      rules, population_contract, timepoint_label,
                                      axes)
    out.unrelated_rejections = [r.describe() for r in aside]
    if blocking:
        # A malformed aggregation is a broken answer, not a missing one. Calling
        # it "not applicable" would let a response that tried and failed to
        # describe a partition read exactly like one that never mentioned arms.
        out.status = PROTOCOL_ERROR
        out.reason = "; ".join(e for r in blocking for e in r.errors)
        return out
    if not sets:
        out.status = NOT_APPLICABLE
        out.reason = "the response offered no per-arm counts to add up"
        return out

    # `on_unknown: reject`, applied to the claim's own side. Being the only set
    # on offer is not a statement that it counts the people the review reports —
    # and a derived total is exactly the kind of answer nobody can re-check by
    # looking at one printed number, so the population it belongs to has to have
    # been asked for rather than inferred from a lack of alternatives.
    if rules.population_must_match_claim and (
            requested_scope is None or not requested_scope.stated):
        out.status = REJECTED
        out.reason = ("the claim does not say which population it reports, and "
                      f"this policy ({rules.on_unknown} on unknown) will not let "
                      "a computed total stand in for a population nobody named")
        return out

    chosen, reason, exact = _select_set(sets, requested_scope, timepoint_label,
                                        rules, population_contract, axes)
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
                contract: PopulationContract | None, axes: list[str]
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

    # The axes come from the caller: this policy's floor, raised by whatever the
    # run contract requires. A printed total and a computed one are held to the
    # same standard in the same run, which they were not while this function
    # decided for itself that population basis was enough.
    matches = [s for s in sets
               if scope_verdict(requested, s.population or PopulationScope(),
                                required_axes=axes,
                                contract=contract).status == "ok"]
    # A declared timepoint is checked whether or not there is a choice. Being
    # the only candidate is not evidence of being the right one: a set anchored
    # at baseline answering a claim about week 12 is wrong in exactly the way it
    # would be wrong if a second set existed.
    if timepoint_label and rules.timepoint_must_match_claim:
        unanchored = [s for s in matches if not s.timepoint_phrase]
        if unanchored:
            return None, (f"the review reports this at {timepoint_label!r} and "
                          f"{len(unanchored)} of the response's sets carry no "
                          "timepoint of their own, so nothing shows they are "
                          "counted at that moment"), False
        matches = [s for s in matches
                   if _same_timepoint(s.timepoint_phrase, timepoint_label)]
        if not matches:
            return None, (f"the review reports this at {timepoint_label!r} and no "
                          "set of counts is stated at that timepoint"), False
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
    """Normalised exact, because there is no timepoint contract to be lenient with.

    A shared word is not a shared moment: "median progression-free survival" and
    "overall survival at 5 years" have "survival" in common and nothing else.
    Until a frozen timepoint vocabulary exists — the same thing the abbreviation
    problem in ``batch_project`` is waiting on — the only defensible test is
    whether the two say the same thing, and anything looser refuses nothing.
    """
    return _normalise_timepoint(phrase) == _normalise_timepoint(label)


def _normalise_timepoint(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text or "").lower().split())


def _split_rejected(rejected: list[RejectedAggregationSet],
                    requested: PopulationScope | None, rules: AggregationPolicy,
                    contract: PopulationContract | None, timepoint_label: str = "",
                    axes: list[str] | None = None
                    ) -> tuple[list[RejectedAggregationSet],
                               list[RejectedAggregationSet]]:
    """Which broken sets could have held this claim's answer, and which could not.

    ANY axis that is definitely different clears the set: a set about other
    people cannot have held this answer whatever its timepoint, and one at
    another moment cannot whatever its population. But every axis has to be
    definite to be useful — an axis nobody could resolve leaves the set possibly
    relevant, and a possibly-relevant broken set means no surviving set can be
    shown to be the only candidate.
    """
    if requested is None or not rules.population_must_match_claim:
        return list(rejected), []       # nothing distinguishes them; all block

    from react_review.audit.scope import MISMATCH, scope_verdict

    blocking, aside = [], []
    for bad in rejected:
        cleared = False
        if bad.population_known:
            verdict = scope_verdict(requested, bad.population,
                                    required_axes=axes or ["population_basis"],
                                    contract=contract)
            cleared = verdict.status == MISMATCH
        if not cleared and timepoint_label and bad.timepoint_phrase:
            cleared = not _same_timepoint(bad.timepoint_phrase, timepoint_label)
        (aside if cleared else blocking).append(bad)
    return blocking, aside


def _partition_holds(chosen: AggregationSet, rules: AggregationPolicy) -> str:
    """Whether this set's arms are shown to be all of them, once each."""
    components = chosen.cohort_counts
    if rules.require_distinct_arms:
        keys = [tuple(sorted(distinguishing_tokens(c.arm_label)))
                for c in components]
        if len(set(keys)) != len(keys) or not all(keys):
            return ("the same arm appears more than once among the components, so "
                    "adding them would count some people twice")
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
