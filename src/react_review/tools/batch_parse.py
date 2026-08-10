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
    BatchAggregationEvidence,
    BatchCohortCount,
    BatchEntry,
    EntryIdentity,
    EvidenceAnchor,
    PartitionAssessment,
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
    #: Components offered toward a total, once each has been anchored. Present
    #: only when every component survived: a partial set is not a partition.
    aggregation_evidence: BatchAggregationEvidence | None = None
    #: Why the components cannot be summed. Independent of ``rejected``: losing
    #: the aggregation never costs an explicit total the same response carries.
    aggregation_errors: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> list[BatchEntry]:
        return [e for e in self.entries if e.usable]

    def summary(self) -> str:
        return (f"{len(self.usable)} usable reading(s), "
                f"{len(self.rejected)} rejected"
                + (f"; {len(self.aggregation_evidence.cohort_counts)} component(s)"
                   if self.aggregation_evidence else "")
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

    evidence, errors = (parse_aggregation(raw, document) if aggregable
                        else (None, []))
    return BatchReading(
        entries=entries, rejected=rejected,
        aggregation_evidence=evidence, aggregation_errors=errors,
        nothing_reported_reason=str(raw.get("nothing_reported_reason") or "").strip())


def parse_aggregation(raw: dict,
                      document: str) -> tuple[BatchAggregationEvidence | None, list[str]]:
    """Read the components of a total, all-or-nothing.

    Unlike a reading, a component has no value on its own: it is only ever an
    addend. So the isolation that protects the readings is wrong here — dropping
    one bad component and summing the rest would produce a total that is missing
    an arm, which is a wrong answer rather than a lost one. One bad component
    therefore costs the whole aggregation, and nothing else.
    """
    counts_raw = raw.get("cohort_counts")
    if counts_raw in (None, "", []):
        return None, []
    if not isinstance(counts_raw, list):
        return None, ["`cohort_counts` is not a list"]

    counts: list[BatchCohortCount] = []
    errors: list[str] = []
    for index, item in enumerate(counts_raw):
        count, reason = _parse_component(item, index, document)
        if count is None:
            errors.append(reason)
        else:
            counts.append(count)

    seen: dict[tuple, int] = {}
    for count in counts:
        key = tuple(sorted(distinguishing_tokens(count.arm_label)))
        if key in seen and seen[key] != count.count:
            errors.append(f"the arm {count.arm_label!r} is given two different "
                          f"counts ({seen[key]} and {count.count}), so the "
                          "paper's own account of it is not settled")
        seen[key] = count.count

    if errors:
        return None, errors
    return BatchAggregationEvidence(
        cohort_counts=counts, partition=_parse_partition(raw, document)), []


def _parse_component(item: object, index: int,
                     document: str) -> tuple[BatchCohortCount | None, str]:
    """One addend, anchored in the paper — value, arm name and population alike."""
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

    phrase = str(item.get("population_phrase") or "").strip()
    verdict, reason = binding_verdict(phrase, str(count), quote=quote,
                                      document=document) if phrase else ("", "")
    if not phrase or not bound(verdict):
        return None, (f"the arm {label!r} does not say which people its {count} "
                      "counts" if not phrase else
                      f"the population {phrase!r} is not printed with the {count} "
                      f"it is supposed to describe: {reason}")
    scope = classify_population(phrase, source=verdict)
    if not scope.stated:
        return None, (f"{phrase!r} is not a population this contract recognises, "
                      f"so what the {count} for {label!r} counts is unknown")
    return BatchCohortCount(arm_label=label, count=count, quote=quote,
                            population_phrase=phrase, source_index=index,
                            population=scope), ""


def _parse_partition(raw: dict, document: str) -> PartitionAssessment:
    """The claim that these arms cover the population once each — with its proof.

    The flags are carried as given and the quote is checked. Nothing is decided
    here: a ``True`` that turns out to be unanchored is refused later, by the
    policy, which is the only place allowed to permit arithmetic.
    """
    body = raw.get("partition")
    if not isinstance(body, dict):
        return PartitionAssessment()
    quote = str(body.get("quote") or "").strip()
    return PartitionAssessment(
        complete=bool(body.get("complete")),
        mutually_exclusive=bool(body.get("mutually_exclusive")),
        quote=quote, reason=str(body.get("reason") or "").strip(),
        population_phrase=str(body.get("population_phrase") or "").strip(),
        anchored=bool(quote) and normalised_contains(document, quote))


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
