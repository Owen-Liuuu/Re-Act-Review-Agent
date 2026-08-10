"""Finding ONE claim's answer inside a batched reading — in two stages.

The order matters, and getting it wrong is what would make the whole batch
pointless.

**Stage one asks which ARM.** All the readings of one arm are folded together
first, so the paper's arms can be matched against the review's by the same
global one-to-one assignment as before. A trial that reports 314 allocated to
the combination arm and 313 analysed in it has ONE combination arm; feeding both
readings into the assignment as if they were two arms would leave three review
arms chasing four paper arms and nothing would resolve.

**Stage two asks which READING.** Only inside the arm that won does the claim
choose: by population, by timepoint, by effect definition — the axes its
contract says matter. This is where MA004 is decided, and it can only be decided
here, because "which arm" and "which reading of that arm" are different
questions with different evidence.

Everything that does not resolve uniquely is an explicit outcome with a reason.
A batch never falls back to the nearest reading, and a claim whose arm the batch
never reported is not answered by another arm that happens to be present.

Arithmetic stays out. A whole-study total the paper does not print is not a
total it reported: this projects the total the paper states, or nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from react_review.normalize.cohorts import ComparisonTarget, label_affinity
from react_review.normalize.population import PopulationContract, PopulationScope
from react_review.schemas.batch import (
    ARM,
    COMPARISON,
    STUDY,
    BatchCohortCount,
    BatchEntry,
)
from react_review.tools.batch_parse import BatchReading
from react_review.tools.safe_aggregation import (
    NOT_APPLICABLE,
    PROTOCOL_ERROR,
    REJECTED,
    AggregationPolicy,
    derive_partitioned_total,
)
from react_review.tools.target_assignment import MIN_PAIR_SIDE, assign_arms, resolve_sides

OK = "ok"
#: The total was not read but computed, under the frozen aggregation policy.
#: Deliberately not ``OK``: a derived number and a printed one are different
#: kinds of fact, and a caller that wants only what the paper states can say so.
DERIVED = "derived"
NOT_REPORTED = "not_reported"
AMBIGUOUS = "ambiguous"
SCOPE_UNRESOLVED = "scope_unresolved"
TIMEPOINT_UNRESOLVED = "timepoint_unresolved"
CONTRADICTORY = "contradictory"
UNSUPPORTED = "unsupported"
BATCH_FAILED = "batch_failed"


@dataclass
class Projection:
    """What one claim got out of the batch, and how."""

    status: str = UNSUPPORTED
    entry: BatchEntry | None = None
    reason: str = ""
    #: Every reading that survived stage one, so a refusal can be inspected.
    candidates: list[BatchEntry] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    #: Set only when the total was computed rather than read. Kept beside the
    #: entry rather than dressed up as one: a derived number has no quote that
    #: prints it, and manufacturing a BatchEntry for it would let it be handled
    #: everywhere as though the paper had stated it.
    derived_value: int | None = None
    derivation: str = ""
    aggregation_status: str = NOT_APPLICABLE
    aggregation_reason: str = ""
    cohort_counts: list[BatchCohortCount] = field(default_factory=list)
    verified_scope: PopulationScope | None = None
    partition_quote: str = ""
    population_quote: str = ""
    timepoint_quote: str = ""
    aggregation_set: str = ""
    unrelated_rejections: list[str] = field(default_factory=list)
    #: Why a set could not be read at all. Survives every path: a released
    #: printed total does not make a malformed aggregation block stop existing.
    aggregation_errors: list[str] = field(default_factory=list)
    policy_id: str = ""
    policy_sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def released(self) -> bool:
        """Whether this claim got an answer at all, read or computed."""
        return self.status in (OK, DERIVED)

    @property
    def value(self) -> str | None:
        if self.entry is not None:
            return self.entry.value
        return str(self.derived_value) if self.derived_value is not None else None

    @property
    def evidence_anchors(self) -> list[str]:
        """EVERY passage this answer rests on.

        For a derived total that is four kinds of passage, not one: what was
        added, why adding them is the whole, whom they count and when. Returning
        only the component quotes would present the arithmetic as though the
        conditions that licensed it were not evidence.
        """
        if self.entry is not None:
            return [self.entry.quote] if self.entry.quote else []
        return [q for q in (self.population_quote, self.timepoint_quote,
                            self.partition_quote,
                            *(c.quote for c in self.cohort_counts)) if q]


def project_claim(
    reading: BatchReading, *,
    target_shape: str = ARM,
    review_labels: dict[str, str] | None = None,
    cohort_key: str = "",
    comparison: ComparisonTarget | None = None,
    requested_scope: PopulationScope | None = None,
    required_axes: list[str] | None = None,
    timepoint_label: str = "",
    population_contract: PopulationContract | None = None,
    aggregation_policy: AggregationPolicy | None = None,
    field_type: str = "",
) -> Projection:
    """Answer ONE claim from a batched reading, or say why it cannot be."""
    if reading.batch_error:
        return Projection(status=BATCH_FAILED, reason=reading.batch_error)
    entries = reading.usable

    if target_shape == STUDY:
        return _project_study(
            reading, entries, requested_scope=requested_scope,
            required_axes=required_axes or [], timepoint_label=timepoint_label,
            population_contract=population_contract, policy=aggregation_policy,
            field_type=field_type)

    if not entries:
        return Projection(
            status=NOT_REPORTED,
            reason=(reading.nothing_reported_reason
                    or "the batch reported no usable reading for this field"))

    # --- stage one: which target ------------------------------------------
    groups = _by_target(entries)
    if target_shape == COMPARISON:
        candidates, reason, provenance = _comparison_candidates(
            groups, comparison, review_labels or {})
        if candidates is None:
            return Projection(status=reason[0], reason=reason[1], candidates=entries,
                              provenance=provenance)
    else:
        candidates, reason, provenance = _arm_candidates(
            groups, review_labels or {}, cohort_key)
        if candidates is None:
            return Projection(status=reason[0], reason=reason[1], candidates=entries,
                              provenance=provenance)

    # --- stage two: which reading of it -----------------------------------
    return _select(candidates, requested_scope=requested_scope,
                   required_axes=required_axes or [], timepoint_label=timepoint_label,
                   population_contract=population_contract, provenance=provenance)


#: What a printed-total outcome permits the arithmetic to do afterwards.
#:
#: A sum may fill a silence. It may never settle an argument: where the paper's
#: own printed totals disagree, or where two of them answer the claim equally
#: well, offering a computed third number does not resolve that — it decides
#: which of the paper's statements to ignore, and hides the discrepancy the audit
#: exists to surface. Those two outcomes are final.
_NEVER_DERIVE = (CONTRADICTORY, AMBIGUOUS)


def _project_study(reading: BatchReading, entries: list[BatchEntry], *,
                   requested_scope, required_axes, timepoint_label: str,
                   population_contract, policy, field_type: str) -> Projection:
    """A whole-study total: read it if the paper prints it, compute it if not.

    The order is not a preference, it is the difference between a fact and an
    inference. A printed total is what the paper says; a sum is what the paper
    implies under conditions that have to hold. So the printed one is looked for
    first, the computed one is only reached when there is no printed one AT THE
    POPULATION THE REVIEW REPORTS, and when both exist they must agree — a
    disagreement is the paper's own accounting failing, and picking whichever
    number matches the review would hide exactly the discrepancy being audited.
    """
    study_entries = [e for e in entries if e.identity.target_shape == STUDY]
    provenance: dict[str, Any] = {"stage_one": "study-level, no arm to match"}
    if reading.aggregation_errors:
        provenance["aggregation_errors"] = list(reading.aggregation_errors)

    explicit = Projection(
        status=NOT_REPORTED, candidates=study_entries,
        reason=(reading.nothing_reported_reason
                or "the batch reports no whole-study value"))
    if study_entries:
        explicit = _select(study_entries, requested_scope=requested_scope,
                           required_axes=required_axes,
                           timepoint_label=timepoint_label,
                           population_contract=population_contract,
                           provenance=provenance)

    summed = derive_partitioned_total(
        reading.aggregation_sets, requested_scope, target_shape=STUDY,
        field_type=field_type, timepoint_label=timepoint_label,
        rejected_sets=reading.rejected_sets, policy=policy,
        population_contract=population_contract)
    provenance["policy"] = f"{summed.policy_id} ({summed.policy_sha256[:12]}…)"
    if summed.chosen_set is not None:
        provenance["aggregation_set"] = summed.chosen_set.describe()

    # Whatever happens next, what the aggregation DID is recorded. A malformed
    # set that is never used still has to be visible: it is a fact about the
    # response, and a released explicit total does not make it go away.
    def _carry(projection: Projection) -> Projection:
        projection.aggregation_status = summed.status
        projection.aggregation_reason = summed.reason
        projection.cohort_counts = summed.components
        projection.aggregation_errors = list(reading.aggregation_errors)
        projection.policy_id = summed.policy_id
        projection.policy_sha256 = summed.policy_sha256
        projection.population_quote = summed.population_quote
        projection.timepoint_quote = summed.timepoint_quote
        projection.partition_quote = (projection.partition_quote
                                      or summed.partition_quote)
        projection.unrelated_rejections = list(summed.unrelated_rejections)
        projection.aggregation_set = (summed.chosen_set.describe()
                                      if summed.chosen_set else "")
        projection.provenance = {**projection.provenance, **provenance}
        return projection

    if explicit.ok:
        explicit = _carry(explicit)
        if summed.derived:
            printed = _as_int(explicit.value)
            explicit.provenance["cross_check"] = (
                f"{summed.derivation} vs printed {explicit.value}")
            if printed is None:
                # A total that is not one whole number is not something the arms
                # could corroborate; the printed value stands on its own.
                explicit.aggregation_status = NOT_APPLICABLE
                explicit.aggregation_reason = (
                    "the printed total is not a single whole number, so the arm "
                    "counts cannot confirm or contradict it")
            elif printed != summed.total:
                return _carry(Projection(
                    status=CONTRADICTORY, candidates=study_entries,
                    entry=explicit.entry, derivation=summed.derivation,
                    verified_scope=summed.verified_scope,
                    partition_quote=summed.partition_quote,
                    provenance=explicit.provenance,
                    reason=(f"the paper prints {printed} for this population and its "
                            f"own arms add to {summed.total} ({summed.derivation}); "
                            "the paper does not agree with itself here, and choosing "
                            "one of them would hide that")))
            else:
                explicit.aggregation_reason = (
                    f"the arms independently add to the same total "
                    f"({summed.derivation})")
        return explicit

    # --- may the sum be reached at all? -----------------------------------
    if explicit.status in _NEVER_DERIVE:
        explicit = _carry(explicit)
        explicit.reason = (
            f"{explicit.reason}. Adding the arms up would not settle this: a "
            "computed total cannot decide which of the paper's own printed "
            "statements to believe")
        return explicit
    if explicit.status == SCOPE_UNRESOLVED and not summed.scope_matched_exactly:
        # The printed totals were for other people. A set that also fails to
        # match this claim's population exactly is for other people too.
        explicit = _carry(explicit)
        return explicit
    if explicit.status == TIMEPOINT_UNRESOLVED and not summed.timepoint_verified:
        explicit = _carry(explicit)
        explicit.reason = (
            f"{explicit.reason}; and the arm counts carry no timepoint of their "
            "own, so they cannot answer at a timepoint the printed totals could "
            "not")
        return explicit

    if summed.derived:
        return _carry(Projection(
            status=DERIVED, candidates=study_entries, derived_value=summed.total,
            derivation=summed.derivation, verified_scope=summed.verified_scope,
            partition_quote=summed.partition_quote, reason=summed.reason,
            provenance={**provenance, "explicit_total": explicit.status,
                        "explicit_reason": explicit.reason}))

    # Neither route worked. The refusal that explains the most is the one that
    # got furthest: if components were offered, say which condition stopped the
    # sum; otherwise say that the paper simply does not print the total.
    explicit = _carry(explicit)
    if summed.status in (REJECTED, PROTOCOL_ERROR):
        explicit.reason = (f"{explicit.reason}; and the arm counts could not be "
                           f"added up because {summed.reason}")
    elif explicit.status == NOT_REPORTED:
        explicit.reason = (f"{explicit.reason}. A total the paper does not print is "
                           "not a total it reported, and no per-arm counts were "
                           "offered that could be shown to partition anything")
    return explicit


def _as_int(text: str | None) -> int | None:
    """A printed total, when it is one whole number and nothing else."""
    digits = "".join(ch for ch in str(text or "") if ch.isdigit() or ch == ",")
    digits = digits.replace(",", "")
    return int(digits) if digits and digits == str(text or "").strip().replace(",", "") else None


def _by_target(entries: list[BatchEntry]) -> dict[tuple, list[BatchEntry]]:
    """Fold every reading of one target together — the ONLY level stage one sees."""
    groups: dict[tuple, list[BatchEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.identity.target(), []).append(entry)
    return groups


def _arm_candidates(groups, review_labels: dict[str, str], cohort_key: str):
    """Map the paper's arms onto the review's, then hand back the one asked for."""
    if not cohort_key or not review_labels:
        return None, (UNSUPPORTED, "no cohort was requested for an arm batch"), {}
    representatives = {target: group[0].identity.arm_label
                       for target, group in groups.items()}
    mapping, reason, margin = assign_arms(review_labels, list(representatives.values()))
    provenance = {"stage_one": "global one-to-one arm assignment",
                  "arm_mapping": mapping, "margin": round(margin, 4),
                  "papers_arms": list(representatives.values())}
    if not mapping:
        return None, (AMBIGUOUS, f"the arms could not be matched: {reason}"), provenance
    label = mapping.get(cohort_key)
    if not label:
        return None, (NOT_REPORTED,
                      f"the batch reports no arm matching {cohort_key!r}"), provenance
    for target, name in representatives.items():
        if name == label:
            return groups[target], ("", ""), provenance
    return None, (NOT_REPORTED, f"no reading was kept for {label!r}"), provenance


def _comparison_candidates(groups, comparison: ComparisonTarget | None,
                           review_labels: dict[str, str]):
    """Match the requested pair against the pairs the paper reports."""
    if comparison is None:
        return None, (UNSUPPORTED, "no comparison was requested"), {}
    sides, reason = resolve_sides(comparison, review_labels)
    if sides is None:
        return None, (UNSUPPORTED, reason), {}
    scored = []
    for target, group in groups.items():
        pair = group[0].identity.comparison_pair
        if not pair:
            continue
        forward = min(label_affinity(sides[0], pair[0]), label_affinity(sides[1], pair[1]))
        inverted = min(label_affinity(sides[0], pair[1]), label_affinity(sides[1], pair[0]))
        scored.append((forward, inverted, target, pair))
    if not scored:
        return None, (NOT_REPORTED, "the batch reports no comparison"), {}
    scored.sort(key=lambda s: -s[0])
    best_forward, best_inverted, target, pair = scored[0]
    provenance = {"stage_one": "comparison pair matching",
                  "paper_pair": list(pair), "forward": round(best_forward, 3),
                  "inverted": round(best_inverted, 3)}
    if best_inverted > best_forward and best_inverted >= MIN_PAIR_SIDE:
        return None, (UNSUPPORTED,
                      "the batch reports this comparison the other way round, "
                      "which is a different number"), provenance
    if best_forward < MIN_PAIR_SIDE:
        return None, (NOT_REPORTED,
                      "no comparison in the batch matches the one requested"), provenance
    if len(scored) > 1 and scored[1][0] >= best_forward:
        return None, (AMBIGUOUS,
                      "two comparisons in the batch match the request equally "
                      "well"), provenance
    return groups[target], ("", ""), provenance


def _select(candidates: list[BatchEntry], *, requested_scope, required_axes,
            timepoint_label: str, population_contract, provenance) -> Projection:
    """Choose among the readings OF ONE TARGET. This is where MA004 is decided."""
    remaining = list(candidates)
    trace: list[str] = []

    if required_axes and requested_scope is not None:
        matched = [e for e in remaining
                   if _scope_fits(e.identity.population, requested_scope,
                                  required_axes, population_contract)]
        trace.append(f"{len(matched)}/{len(remaining)} match the requested "
                     f"population {requested_scope.describe()}")
        if not matched:
            return Projection(
                status=SCOPE_UNRESOLVED, candidates=candidates,
                provenance={**provenance, "stage_two": trace},
                reason=(f"the batch reports this target, but none of its "
                        f"{len(remaining)} reading(s) counts the "
                        f"{requested_scope.describe()} population the review "
                        "reports"))
        remaining = matched

    # The timepoint DISCRIMINATES; it does not reject on its own.
    #
    # Confirming that "median PFS" and "median progression-free survival" name
    # one timepoint needs an abbreviation table — domain data, not a string
    # rule — and any word matcher loose enough to accept that pair also accepts
    # "overall survival", because both share "survival". So where the batch
    # offers a choice, the timepoint decides it; where it offers one reading,
    # the claim is answered and the fact that nobody verified the timepoint is
    # recorded rather than hidden. A real guard waits on that table.
    phrases = sorted({e.identity.timepoint_phrase for e in remaining
                      if e.identity.timepoint_phrase})
    if timepoint_label and len(remaining) > 1 and len(phrases) > 1:
        scored = sorted(
            ((label_affinity(e.identity.timepoint_phrase, timepoint_label), i, e)
             for i, e in enumerate(remaining)), key=lambda s: (-s[0], s[1]))
        best = scored[0][0]
        if best <= 0.0:
            return Projection(
                status=TIMEPOINT_UNRESOLVED, candidates=candidates,
                provenance={**provenance, "stage_two": trace},
                reason=(f"the batch reports this target at {', '.join(phrases)}, "
                        f"none of which is the {timepoint_label!r} the review "
                        "reports"))
        if len(scored) > 1 and scored[1][0] >= best:
            return Projection(
                status=TIMEPOINT_UNRESOLVED, candidates=candidates,
                provenance={**provenance, "stage_two": trace},
                reason=(f"two readings answer to {timepoint_label!r} equally "
                        "well, so which one the review means is not settled here"))
        remaining = [scored[0][2]]
        trace.append(f"timepoint {timepoint_label!r} chose "
                     f"{remaining[0].identity.timepoint_phrase!r}")
    elif timepoint_label and phrases:
        trace.append(f"timepoint not verified: the paper says {phrases}, the "
                     f"review says {timepoint_label!r}, and confirming they are "
                     "the same needs an abbreviation table")

    if len(remaining) == 1:
        return Projection(status=OK, entry=remaining[0], candidates=candidates,
                          provenance={**provenance, "stage_two": trace})

    values = {str(e.value) for e in remaining}
    if len(values) == 1:
        # Several readings, one number: the paper says the same thing twice.
        return Projection(status=OK, entry=remaining[0], candidates=candidates,
                          provenance={**provenance, "stage_two": trace + [
                              f"{len(remaining)} readings agree on {remaining[0].value!r}"]})
    return Projection(
        status=CONTRADICTORY, candidates=remaining,
        provenance={**provenance, "stage_two": trace},
        reason=("the batch reports " + " and ".join(sorted(values))
                + " for the same reading of this target, so the paper's own "
                  "account of it is not settled here"))


def _scope_fits(entry_scope: PopulationScope | None, requested: PopulationScope,
                required_axes: list[str], contract: PopulationContract | None) -> bool:
    """An entry fits when every REQUIRED axis is stated and compatible."""
    from react_review.audit.scope import scope_verdict

    outcome = scope_verdict(requested, entry_scope or PopulationScope(),
                            required_axes=required_axes, contract=contract)
    return outcome.status == "ok"



