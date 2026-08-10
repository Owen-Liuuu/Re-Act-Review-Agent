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

from typing import Any

from pydantic import BaseModel, Field

from react_review.normalize.anchors import normalised_contains, value_supported_by_quote
from react_review.normalize.population import PopulationScope, classify_population
from react_review.schemas.batch import (
    ARM,
    COMPARISON,
    STUDY,
    BatchEntry,
    EntryIdentity,
    EvidenceAnchor,
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

    @property
    def usable(self) -> list[BatchEntry]:
        return [e for e in self.entries if e.usable]

    def summary(self) -> str:
        return (f"{len(self.usable)} usable reading(s), "
                f"{len(self.rejected)} rejected"
                + (f"; batch failed: {self.batch_error}" if self.batch_error else ""))


def parse_batch(raw: object, document: str, *, target_shape: str = ARM) -> BatchReading:
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
    return BatchReading(
        entries=entries, rejected=rejected,
        nothing_reported_reason=str(raw.get("nothing_reported_reason") or "").strip())


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

    supporting = str(item.get("population_quote") or "").strip()
    if supporting:
        verdict, _ = binding_verdict(phrase, value, quote=quote, document=document)
        if bound(verdict):
            return classify_population(phrase, source=verdict), EvidenceAnchor(
                quote=supporting)
    # Unbound: the axis stays unstated. Borrowing it is exactly the error.
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
