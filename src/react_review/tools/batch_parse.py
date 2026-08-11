"""Reading a batched response without letting one bad line destroy the rest.

The Phase 7 parser returned an empty list the moment any entry failed
validation. For a single-target question that was defensible: the one answer was
unusable, so there was nothing to keep. For a batch it is not. One malformed
reading out of five would discard four good ones and fail every claim in the
group — a failure mode that grows with the batch it is supposed to make safe.

So failure has levels, and each one costs only what it should:

*The response is not a batch.* Nothing can be read; every claim in the group is
unresolved, for one stated reason.

*One entry is unusable.* That entry is dropped with its reason recorded, and
the others are read. A claim that needed it is unresolved; a claim that did not
never knows.

*The evidence does not bind.* A population or timepoint the paper did not print
beside the value is not that value's — the entry survives, but with that axis
unstated rather than borrowed from somewhere else in the paper.

The two levels above these — an arm mapping that is not unique, and an arm the
batch never reported — belong to projection, where the claim is known.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from react_review.normalize.anchors import normalised_contains, value_supported_by_quote
from react_review.normalize.cohorts import distinguishing_tokens
from react_review.normalize.population import PopulationScope, classify_population
from react_review.schemas.batch import (
    ARM,
    COMPARISON,
    STUDY,
    AggregationSet,
    BatchCohortCount,
    RejectedAggregationSet,
    BatchEntry,
    EntryIdentity,
    EvidenceAnchor,
    PartitionWitness,
)
from react_review.tools.evidence_binding import binding_verdict, bound
from react_review.tools.target_assignment import _label_in_quote
from react_review.tools.value_components import parse_component_block


class BatchReading(BaseModel):
    """Everything one batched response yielded, and everything it cost."""

    entries: list[BatchEntry] = Field(default_factory=list)
    #: Set only when NOTHING could be read; every claim in the group fails.
    batch_error: str = ""
    #: Entries dropped one by one, with the reason each was dropped.
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    nothing_reported_reason: str = ""
    #: One entry per population the response offered components for. A set is
    #: present only when ALL of its components survived: a partial set is not a
    #: partition of anything.
    aggregation_sets: list[AggregationSet] = Field(default_factory=list)
    #: The sets that could not be read, each carrying whatever was legible about
    #: it. Whether a broken set costs a claim anything depends on what it was
    #: about, and that can only be judged where the claim is known.
    rejected_sets: list[RejectedAggregationSet] = Field(default_factory=list)
    #: Why a set could not be read. Independent of ``rejected``: losing the
    #: aggregation never costs an explicit total the same response carries — but
    #: it is never silently forgotten either, because a malformed set and an
    #: absent one are different facts about the response, and reporting the first
    #: as the second is how a broken answer comes to look like a missing one.
    aggregation_errors: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> list[BatchEntry]:
        return [e for e in self.entries if e.usable]

    @property
    def aggregation_malformed(self) -> bool:
        """Whether the aggregation block was BROKEN, as opposed to absent."""
        return bool(self.aggregation_errors)

    def summary(self) -> str:
        return (f"{len(self.usable)} usable reading(s), "
                f"{len(self.rejected)} rejected"
                + (f"; {len(self.aggregation_sets)} aggregation set(s)"
                   if self.aggregation_sets else "")
                + (f"; aggregation unusable: {self.aggregation_errors[0]}"
                   if self.aggregation_errors else "")
                + (f"; batch failed: {self.batch_error}" if self.batch_error else ""))


def parse_batch(raw: object, document: str, *, target_shape: str = ARM,
                aggregable: bool = False) -> BatchReading:
    """Read a v5 batch response, isolating what fails from what does not."""
    if not isinstance(raw, dict):
        return BatchReading(batch_error="the response is not a JSON object")
    readings = raw.get("readings")
    if readings is None:
        return BatchReading(batch_error="the response carries no `readings` list")
    if not isinstance(readings, list):
        return BatchReading(batch_error="`readings` is not a list")

    entries: list[BatchEntry] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(readings):
        entry, reason = parse_one_entry(item, index, document,
                                        target_shape=target_shape)
        if entry is None:
            rejected.append({"index": index, "reason": reason})
        else:
            entries.append(entry)

    sets, bad_sets, errors = (parse_aggregation(raw, document) if aggregable
                              else ([], [], []))
    return BatchReading(
        entries=entries, rejected=rejected, aggregation_sets=sets,
        rejected_sets=bad_sets, aggregation_errors=errors,
        nothing_reported_reason=str(raw.get("nothing_reported_reason") or "").strip())


def parse_aggregation(raw: dict, document: str) -> tuple[
        list[AggregationSet], list[RejectedAggregationSet], list[str]]:
    """Read the sets of components, each population on its own terms.

    The isolation runs in two directions here, and they are opposite on purpose.
    BETWEEN sets it is generous: who was randomised and who was analysed are
    different questions, so a malformed answer to one costs only that one.
    INSIDE a set it is absolute: dropping a bad component and summing the rest
    would produce a total missing an arm, which is a wrong answer rather than a
    lost one, and a wrong total is the failure this whole module exists to
    prevent.
    """
    sets_raw = raw.get("aggregation_sets")
    if sets_raw in (None, "", []):
        return [], [], []
    if not isinstance(sets_raw, list):
        # Nothing here can be attributed to a population, so nothing can be
        # shown to be irrelevant to the claim either.
        return [], [RejectedAggregationSet(
            errors=["`aggregation_sets` is not a list"])], [
            "`aggregation_sets` is not a list"]

    sets: list[AggregationSet] = []
    rejected: list[RejectedAggregationSet] = []
    for index, item in enumerate(sets_raw):
        parsed, refusal = _parse_set(item, index, document)
        if parsed is None:
            rejected.append(refusal)
        else:
            sets.append(parsed)
    return sets, rejected, [e for r in rejected for e in r.errors]


def _parse_set(item: object, index: int, document: str
               ) -> tuple[AggregationSet | None, RejectedAggregationSet | None]:
    """One population, its timepoint, its arms, and the witness tying them.

    A refusal carries whatever was legible before it failed. Which population a
    broken set was about decides whether it costs this claim anything, and that
    is not knowable here.
    """
    where = f"aggregation set {index}"
    scope: PopulationScope | None = None
    phrase = timepoint_phrase = ""

    def fail(reason: str) -> tuple[None, RejectedAggregationSet]:
        return None, RejectedAggregationSet(
            source_index=index, population=scope, population_phrase=phrase,
            timepoint_phrase=timepoint_phrase, errors=[reason])

    if not isinstance(item, dict):
        return fail(f"{where} is not a structured object")

    phrase = str(item.get("population_phrase") or "").strip()
    if not phrase:
        return fail(f"{where} does not say which people it counts")
    witness = str(item.get("population_quote") or "").strip()
    if not witness or not normalised_contains(document, witness):
        return fail(f"{where} counts {phrase!r} but gives no passage of the "
                    "document saying so")
    if not normalised_contains(witness, phrase):
        return fail(f"{where} claims the population {phrase!r}, which its own "
                    "witness passage does not contain")
    scope = classify_population(phrase, source="same_quote")
    if not scope.stated:
        return fail(f"{where} counts {phrase!r}, which is not a population this "
                    "contract recognises")

    timepoint_phrase = str(item.get("timepoint_phrase") or "").strip()
    timepoint_quote = ""
    if timepoint_phrase:
        timepoint_quote = str(item.get("timepoint_quote") or "").strip() or witness
        if not normalised_contains(document, timepoint_quote):
            return fail(f"{where} claims the timepoint {timepoint_phrase!r} with "
                        "no passage of the document behind it")
        if not normalised_contains(timepoint_quote, timepoint_phrase):
            return fail(f"{where} claims the timepoint {timepoint_phrase!r}, which "
                        "its own witness passage does not contain")

    counts_raw = item.get("cohort_counts")
    if not isinstance(counts_raw, list) or not counts_raw:
        return fail(f"{where} offers no arm counts")
    counts: list[BatchCohortCount] = []
    for position, entry in enumerate(counts_raw):
        count, reason = _parse_component(entry, position, document)
        if count is None:
            return fail(f"{where}: {reason}")    # all or nothing, see above
        # Being listed inside a set does not make a number belong to that set's
        # people. Without this an analysed count sits among allocated ones and is
        # added to them, because nothing ever asked the paper whether it counts
        # the same population — the very substitution the sets were introduced to
        # prevent, performed one level down.
        reason = _component_belongs(count, phrase, timepoint_phrase, document)
        if reason:
            return fail(f"{where}: {reason}")
        counts.append(count)

    seen: dict[tuple, BatchCohortCount] = {}
    for count in counts:
        key = tuple(sorted(distinguishing_tokens(count.arm_label)))
        if not key:
            return fail(f"{where} offers a count for {count.arm_label!r}, which "
                        "carries no word that could distinguish an arm")
        if key in seen:
            # Twice with the SAME count is the more dangerous case: the sum still
            # looks plausible, and 316 + 316 is a study of 632 people that never
            # existed.
            return fail(f"{where} offers the arm {count.arm_label!r} twice "
                        f"({seen[key].count} and {count.count}), so adding them "
                        "would count some people more than once")
        seen[key] = count

    partition, reason = _parse_partition(item.get("partition"), where, document)
    if partition is None:
        return fail(reason)
    reason = _partition_describes_this_set(partition, scope, witness,
                                           timepoint_phrase, timepoint_quote,
                                           where, document)
    if reason:
        return fail(reason)
    reason = _census_is_read_not_asserted(partition, counts, where)
    if reason:
        return fail(reason)

    return AggregationSet(
        population_type=str(item.get("population_type") or "").strip(),
        population_phrase=phrase, population_quote=witness,
        timepoint_phrase=timepoint_phrase, timepoint_quote=timepoint_quote,
        cohort_counts=counts, partition=partition, population=scope,
        source_index=index), None


def _parse_component(item: object, index: int,
                     document: str) -> tuple[BatchCohortCount | None, str]:
    """One addend, anchored in the paper — both its number and its arm's name.

    It carries no population: the SET has the population, so a component cannot
    disagree with the set it is in.
    """
    if not isinstance(item, dict):
        return None, f"component {index} is not a structured object"
    label = str(item.get("arm_label") or "").strip()
    quote = str(item.get("quote") or "").strip()
    count = _positive_integer(item.get("count"))
    if count is None:
        return None, (f"the count for {label or 'an unnamed arm'} is not one "
                      "positive whole number")
    if not quote or not normalised_contains(document, quote):
        return None, (f"the count {count} for {label or 'an unnamed arm'} has no "
                      "quote that is a contiguous passage of the document")
    if not value_supported_by_quote(quote, str(count)):
        return None, f"the quote for {label or 'an unnamed arm'} does not print {count}"
    if not label:
        return None, f"the component counting {count} names no arm"
    if not _label_in_quote(label, quote):
        return None, f"the quote for {count} does not name the arm {label!r}"
    return BatchCohortCount(arm_label=label, count=count, quote=quote,
                            source_index=index), ""


def _component_belongs(count: BatchCohortCount, population_phrase: str,
                       timepoint_phrase: str, document: str) -> str:
    """Whether THIS number is one of the people (and moments) the set describes."""
    for phrase, axis in ((population_phrase, "population"),
                         (timepoint_phrase, "timepoint")):
        if not phrase:
            continue
        verdict, reason = binding_verdict(phrase, str(count.count),
                                          quote=count.quote, document=document)
        if not bound(verdict):
            return (f"the count {count.count} for {count.arm_label!r} is not "
                    f"printed with the {axis} {phrase!r} this set counts: {reason}")
    return ""


def _partition_describes_this_set(partition: PartitionWitness,
                                  scope: PopulationScope, population_quote: str,
                                  timepoint_phrase: str, timepoint_quote: str,
                                  where: str, document: str) -> str:
    """Whether the sentence licensing the sum is about THESE people, at this time.

    Everything else in a set is now tied to its population, and this was not: a
    sentence saying the ANALYSED population fell into two groups could license
    adding up two ALLOCATED counts. It is the same substitution as before, made
    at the one place that decides whether adding is allowed at all — so a
    partition that names a different population is refused, and one that names
    none is accepted only where the paper puts it beside this set's own
    population words, which is the weakest link a reader could still follow.
    """
    if not partition.anchored:
        # No locatable passage at all. The policy refuses that, in those words;
        # this check is about WHOM a passage describes, so it has nothing to add.
        return ""
    stated = classify_population(partition.quote, source="same_quote")

    # Every axis, one at a time. Agreeing about the basis is not agreeing: a
    # sentence saying the analysis population fell into three groups is silent
    # about whether the ITT set did, and one written at randomisation is silent
    # about week 12. An axis the set declares is met only by a partition that
    # declares the same thing, or that the paper prints beside the witness for
    # THAT axis — never by the partition being near some other axis's witness.
    for axis, ours, theirs, witness in (
            ("population basis", scope.basis,
             stated.basis if stated.stated else "", population_quote),
            ("analysis set",
             scope.analysis_set if scope.axis_stated("analysis_set") else "",
             stated.analysis_set if stated.axis_stated("analysis_set") else "",
             population_quote),
            ("timepoint", timepoint_phrase,
             timepoint_phrase if timepoint_phrase and normalised_contains(
                 partition.quote, timepoint_phrase) else "",
             timepoint_quote)):
        if not ours:
            continue                     # the set does not claim this axis
        if theirs and theirs != ours:
            return (f"{where} counts {ours!r} and its partition passage describes "
                    f"{theirs!r}, which is a statement about a different set of "
                    "people")
        if axis == "timepoint" and not theirs:
            # No proximity fallback for WHEN. Two sentences printed together can
            # still be about different moments — a randomisation sentence sits
            # beside an allocation sentence and describes the same people, but a
            # baseline table sits beside a week-12 table and does not describe
            # the same visit. A partition at a stated moment has to state it.
            return (f"{where} counts at {ours!r} and its partition passage does "
                    "not say it holds at that moment; groups complete at one "
                    "visit need not be complete at another")
        if not theirs and not _same_block(partition.quote, witness, document):
            return (f"{where} counts {ours!r}, and its partition passage neither "
                    f"says so nor sits with the passage that does, so nothing "
                    f"establishes that these arms are all of {ours!r}")
    if not stated.stated and not scope.basis:
        return (f"{where} gives a partition passage that names no population at "
                "all, and neither do its counts")
    return ""


def _same_block(one: str, other: str, document: str) -> str | bool:
    """Whether two passages sit close enough to be about the same thing."""
    verdict, _ = binding_verdict(one, "", quote=other, document=document)
    return bound(verdict)


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

#: PLURAL nouns for a set of study participants. Plural because the passage has
#: to be counting groups: "group 3" names one, "3 treatment cycles" counts
#: something else entirely, and singular "treatment" is a thing given rather
#: than a set of people ("3 mg treatment").
_GROUP_NOUNS = ("groups", "arms", "cohorts", "regimens")

#: The only words allowed between the number and the noun. A closed list, so
#: that "3 mg treatment arms" cannot creep in on the strength of "mg" being
#: short.
_GROUP_QUALIFIERS = ("treatment", "study", "parallel", "randomised", "randomized",
                     "assigned", "trial", "intervention", "active")


def _census_is_read_not_asserted(partition: PartitionWitness,
                                 counts: list[BatchCohortCount],
                                 where: str) -> str:
    """Whether the arm census came out of the passage, or out of the model.

    ``declared_arm_count`` exists to make ``complete`` falsifiable. Taken on
    trust it does the opposite: a paper that says "one of three groups" and a
    response that declares two arms then agree with each other perfectly, and
    two arms are summed into a study total. So the number has to be findable in
    the passage the response itself pointed at, and the names have to be named
    there.
    """
    if partition.declared_arm_count is None and not partition.declared_arm_labels:
        return ""                       # absent; the policy decides what that costs
    quote = _normalise(partition.quote)
    if not quote:
        return (f"{where} declares how many groups there are but gives no passage "
                "to have read it from")

    if partition.declared_arm_count is not None:
        n = partition.declared_arm_count
        if not _states_group_count(quote, n):
            return (f"{where} says the population was divided into {n} groups, "
                    "which its own partition passage does not state")
        if n != len(counts):
            return (f"{where}: the passage describes {n} groups and "
                    f"{len(counts)} arm count(s) were offered")

    if partition.declared_arm_labels:
        missing = [x for x in partition.declared_arm_labels
                   if not normalised_contains(partition.quote, x)]
        if missing:
            return (f"{where} names {', '.join(missing)} as groups of this "
                    "population, which its own partition passage does not name")
        declared = {tuple(sorted(distinguishing_tokens(x)))
                    for x in partition.declared_arm_labels}
        offered = {tuple(sorted(distinguishing_tokens(c.arm_label)))
                   for c in counts}
        if declared != offered:
            # Equality, not containment. An extra arm nobody declared is as
            # wrong as a declared one nobody counted: either way the components
            # are not the set the passage described.
            return (f"{where}: the passage names "
                    f"{len(partition.declared_arm_labels)} group(s) and the "
                    f"counts offered are not the same set of arms")
        if (partition.declared_arm_count is not None
                and partition.declared_arm_count != len(partition.declared_arm_labels)):
            return (f"{where} says {partition.declared_arm_count} groups and then "
                    f"names {len(partition.declared_arm_labels)} of them")
    return ""


def _states_group_count(normalised_quote: str, n: int) -> bool:
    """Whether the passage says there are N GROUPS, in the grammar of saying so.

    Nearness was the previous rule and it is not a rule: "3 mg treatment" puts a
    number two words from a noun and means a dose, "5-year treatment" means a
    duration, "3 treatment cycles" counts visits, and "group 3" names one arm
    rather than counting three. What licenses a sum is the paper stating how
    many groups there ARE, which English says in a small number of ways:

        three groups · 3 treatment arms · one of three groups
        divided into 3 cohorts · randomised to two regimens

    So the number must be followed by an optional qualifier from a closed list
    and then a PLURAL noun for a study group. Singular "treatment" is not one:
    a treatment is a thing given, not a set of people.
    """
    tokens = normalised_quote.split()
    word = next((w for w, v in _NUMBER_WORDS.items() if v == n), "")
    wanted = {str(n), word} - {""}
    for index, token in enumerate(tokens):
        if token not in wanted:
            continue
        rest = tokens[index + 1:index + 3]
        if not rest:
            continue
        if rest[0] in _GROUP_NOUNS:
            return True
        if len(rest) > 1 and rest[0] in _GROUP_QUALIFIERS and rest[1] in _GROUP_NOUNS:
            return True
    return False


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text or "").lower().split())


def _parse_partition(body: object, where: str,
                     document: str) -> tuple[PartitionWitness | None, str]:
    """The claim that these arms partition this population, and its proof."""
    if not isinstance(body, dict):
        return None, (f"{where} makes no partition claim, so nothing licenses "
                      "adding its arms together")
    complete, reason = _strict_bool(body.get("complete"), "complete", where)
    if complete is None:
        return None, reason
    exclusive, reason = _strict_bool(body.get("mutually_exclusive"),
                                     "mutually_exclusive", where)
    if exclusive is None:
        return None, reason

    declared = body.get("declared_arm_count")
    arm_count = None
    if declared not in (None, ""):
        arm_count = _positive_integer(declared)
        if arm_count is None:
            return None, (f"{where} says the population was divided into "
                          f"{declared!r} groups, which is not a whole number")
    labels_raw = body.get("declared_arm_labels") or []
    if not isinstance(labels_raw, list):
        return None, f"{where} gives `declared_arm_labels` that are not a list"

    quote = str(body.get("quote") or "").strip()
    return PartitionWitness(
        complete=complete, mutually_exclusive=exclusive, quote=quote,
        reason=str(body.get("reason") or "").strip(),
        declared_arm_count=arm_count,
        declared_arm_labels=[str(x).strip() for x in labels_raw if str(x).strip()],
        anchored=bool(quote) and normalised_contains(document, quote)), ""


def _strict_bool(value: object, field: str, where: str) -> tuple[bool | None, str]:
    """Only ``true`` and ``false`` are answers; anything else is malformed.

    ``bool("false")`` is ``True`` in Python. A model that answered with the
    string "false" would therefore have been read as asserting the opposite of
    what it said — the one misreading that turns a refusal into a release, and
    the reason this is not a coercion but a type check.
    """
    if type(value) is bool:
        return value, ""
    return None, (f"{where} answers {field} with {value!r}; only a JSON true or "
                  "false says anything about the paper, and a string or a number "
                  "cannot be read as either without guessing which was meant")


def _positive_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    text = str(value or "").strip()
    return int(text) if re.fullmatch(r"[1-9][0-9]*", text) else None


def parse_one_entry(item: object, index: int, document: str, *,
                    target_shape: str = ARM) -> tuple[BatchEntry | None, str]:
    """Validate ONE reading. Returns ``(entry, reason)``; entry is None if refused."""
    if not isinstance(item, dict):
        return None, "the reading is not a structured object"

    value = item.get("value")
    value = None if value is None else str(value).strip() or None
    quote = str(item.get("quote") or "").strip()
    if value is None:
        return None, "the reading carries no value"
    if not quote:
        return None, f"the reading for {value!r} has no supporting quote"
    if not normalised_contains(document, quote):
        return None, (f"the quote for {value!r} is not a contiguous passage of "
                      "the source document")
    if not value_supported_by_quote(quote, value):
        return None, f"the quote does not contain the value {value!r}"

    identity, reason = _identity_for(item, target_shape, quote, value)
    if identity is None:
        return None, reason

    components, component_error = parse_component_block(item.get("value_components"))
    if component_error:
        return None, f"the reading for {value!r} is unusable: {component_error}"

    population, population_anchor = _axis_scope(item, quote, value, document)
    timepoint_phrase, timepoint_anchor = _axis_phrase(
        item, "timepoint_phrase", "timepoint_quote", quote, value, document)
    effect, effect_anchor = _axis_phrase(
        item, "effect_definition", "effect_quote", quote, value, document)

    identity = identity.model_copy(update={
        "population": population,
        "timepoint_phrase": timepoint_phrase,
        "effect_definition": effect,
    })
    return BatchEntry(
        identity=identity, value=value, unit=str(item.get("unit") or "").strip(),
        quote=quote, components=components, raw_index=index,
        population_anchor=population_anchor, timepoint_anchor=timepoint_anchor,
        effect_anchor=effect_anchor), ""


def _identity_for(item: dict, target_shape: str, quote: str,
                  value: str) -> tuple[EntryIdentity | None, str]:
    """Which target this reading is about — named by the paper, in its own quote."""
    if target_shape == ARM:
        label = str(item.get("arm_label") or "").strip()
        if not label:
            return None, "the reading names no arm"
        if not _label_in_quote(label, quote):
            return None, f"the quote does not name the arm {label!r}"
        return EntryIdentity(target_shape=ARM, arm_label=label), ""
    if target_shape == COMPARISON:
        left = str(item.get("left_label") or "").strip()
        right = str(item.get("right_label") or "").strip()
        if not left or not right:
            return None, "the reading does not name both sides of the comparison"
        for side in (left, right):
            if not _label_in_quote(side, quote):
                return None, f"the quote does not name {side!r}"
        return EntryIdentity(target_shape=COMPARISON,
                             comparison_pair=(left, right)), ""
    if target_shape == STUDY:
        return EntryIdentity(target_shape=STUDY,
                             arm_label=str(item.get("scope_label") or "").strip()), ""
    return None, f"unknown target shape {target_shape!r}"


def _axis_scope(item: dict, quote: str, value: str,
                document: str) -> tuple[PopulationScope, EvidenceAnchor | None]:
    """The population this reading counts — only if the paper put it here.

    A phrase from elsewhere in the paper is not this value's population, however
    genuine the sentence it came from. When nothing binds, the population is
    read from the value's own quote, which is the strictest reading available:
    if those words are not there either, the answer is unknown, and unknown is
    what a scope check is entitled to refuse.
    """
    phrase = str(item.get("population_phrase") or "").strip()
    if not phrase:
        return classify_population(quote), None

    verdict, _ = binding_verdict(phrase, value, quote=quote, document=document)
    if bound(verdict):
        scope = classify_population(phrase, source=verdict)
        if scope.stated:
            return scope, EvidenceAnchor(quote=phrase or quote)
        # The phrase binds but carries no population this contract recognises;
        # fall through to the quote rather than record a scope nobody defined.
        return classify_population(quote), None

    # Unbound: the axis stays unstated. Borrowing it is exactly the error.
    #
    # There used to be a second chance here, in which the model could name a
    # separate passage carrying the phrase. It could never do anything: the
    # binding check already searches the whole document for the phrase near this
    # value, so a passage that would have bound was bound the first time, and one
    # that would not is precisely the cross-block borrowing being refused.
    return classify_population(quote), None


def _axis_phrase(item: dict, phrase_key: str, quote_key: str, quote: str,
                 value: str, document: str) -> tuple[str, EvidenceAnchor | None]:
    """A phrase the paper prints beside this value, or nothing at all."""
    phrase = str(item.get(phrase_key) or "").strip()
    if not phrase:
        return "", None
    verdict, _ = binding_verdict(phrase, value, quote=quote, document=document)
    if bound(verdict):
        return phrase, EvidenceAnchor(quote=str(item.get(quote_key) or "") or quote)
    return "", None
