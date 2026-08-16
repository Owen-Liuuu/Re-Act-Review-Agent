"""Deciding WHICH arm an extracted value belongs to — deterministically.

Phase 6B's dominant failure was not a misread number: it was a correctly read
number attached to the wrong arm. Asked for a single-agent arm's time-to-event
outcome the extractor returned the combination arm's, and asked for one hazard
ratio it returned another; nothing downstream could tell, because a value and a
cohort name arrived with no evidence tying them together.

So the model is asked to ENUMERATE what the paper reports for the field — every
arm (or every comparison) with its own verbatim quote — and this module decides
which of those the request was for. Three deterministic ideas do the work. The
examples below use placeholder arm names ``X`` and ``Y``; nothing here is tied to
a therapy area.

*Both-way label affinity.* An arm named "X" is a subset of "X plus Y", so
one-directional overlap cannot separate a single-agent arm from a combination
arm. Scoring in both directions makes the label that also accounts for the other
side's words win.

*Global one-to-one assignment.* Even both-way affinity ties when the review says
"Y + placebo" and the paper says "Y group" — the combination arm shares that
word. Assigning ALL the review's arms to ALL the paper's arms at once, and
requiring a single best assignment, resolves what no per-cell choice can. A tie
is refused, never broken by position.

*Locality inside the quote.* One sentence often carries every arm's value. The
value the paper prints next to the target arm — under the same one-to-one rule,
and with longer arm names masked first so "Y group" cannot match inside
"X-plus-Y group" — is the one that may be returned.

Anything that does not resolve uniquely is an explicit unresolved outcome with a
reason. The nearest value is never the answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import permutations

from pydantic import BaseModel, Field

from react_review.normalize.anchors import (
    flatten,
    normalised_contains,
    value_supported_by_quote,
)
from react_review.normalize.population import PopulationScope, classify_population
from react_review.normalize.cohorts import (
    ComparisonTarget,
    distinguishing_tokens,
    label_affinity,
)
from react_review.tools.value_components import parse_component_block

# A label must explain at least this much of a candidate before it can be paired
# with it at all, and the best assignment must beat the runner-up by this margin
# before it counts as the single reading of the evidence.
MIN_AFFINITY = 0.34
MIN_MARGIN = 0.08
# A comparison is only "the same comparison" when BOTH sides are a good match.
# The bar is higher than MIN_AFFINITY because a weak fit on one side is how a
# paper's mirror-image comparison gets mistaken for the requested one.
MIN_PAIR_SIDE = 0.5
# Assignment is exhaustive over permutations. Beyond this the paper is not the
# three-arm trial this reasoning was designed for, and refusing is safer than
# an approximation nobody has checked.
MAX_ARMS = 7

_STATUS_OK = "ok"


class ArmEvidence(BaseModel):
    """One arm the paper reports for this field, in the paper's own words."""

    label: str
    value: str | None = None
    unit: str = ""
    quote: str = ""
    # Which population this entry's own quote is talking about. Read from that
    # quote and nothing else, so one arm's analysis set cannot be attributed to
    # another's allocation sentence.
    population: PopulationScope | None = None
    # The parts of this entry's own value, as the response reported them. They
    # belong to the ENTRY rather than to the response as a whole: the entry the
    # assignment picks is not always the one the model would have picked, and
    # its components have to travel with it.
    components: dict[str, float] = Field(default_factory=dict)


class ComparisonEvidence(BaseModel):
    """One comparison the paper reports for this field, direction preserved."""

    left_label: str
    right_label: str
    value: str | None = None
    unit: str = ""
    quote: str = ""
    population: PopulationScope | None = None
    components: dict[str, float] = Field(default_factory=dict)


@dataclass
class TargetAssignment:
    """What the requested target resolved to, and why."""

    # ok | not_reported | ambiguous | direction_inverted | unsupported |
    # inconsistent | too_many
    status: str = "unsupported"
    paper_label: str = ""
    value: str | None = None
    unit: str = ""
    quote: str = ""
    reason: str = ""
    margin: float = 0.0
    mapping: dict[str, str] = field(default_factory=dict)
    components: dict[str, float] = field(default_factory=dict)
    population: PopulationScope | None = None
    # The other values reported in the same evidence, so a shared sentence's
    # second interval cannot be read as this value's.
    rival_values: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == _STATUS_OK


# --- parsing the model's enumeration -------------------------------------

def parse_arms(raw: object, paper_text: str) -> tuple[list[ArmEvidence], str]:
    """Validate the enumerated arms. Every item must carry its OWN evidence."""
    if raw in (None, ""):
        return [], ""
    if not isinstance(raw, list):
        return [], "arms_reported is not a list"
    arms: list[ArmEvidence] = []
    for item in raw:
        if not isinstance(item, dict):
            return [], "an entry in arms_reported is not a structured object"
        label = str(item.get("label") or "").strip()
        quote = str(item.get("quote") or "").strip()
        value = item.get("value")
        value = None if value is None else str(value).strip() or None
        if not label:
            return [], "an entry in arms_reported has no arm label"
        if not quote:
            return [], f"the entry for arm {label!r} has no supporting quote"
        if not normalised_contains(paper_text, quote):
            return [], (f"the quote for arm {label!r} is not a contiguous "
                        "passage of the source document")
        if not _label_in_quote(label, quote):
            return [], f"the quote for arm {label!r} does not name that arm"
        if value is not None and not value_supported_by_quote(quote, value):
            return [], (f"the quote for arm {label!r} does not contain the value "
                        f"{value!r} attributed to it")
        components, component_error = parse_component_block(
            item.get("value_components"))
        if component_error:
            return [], f"the entry for arm {label!r} is unusable: {component_error}"
        arms.append(ArmEvidence(label=label, value=value,
                                unit=str(item.get("unit") or "").strip(),
                                quote=quote, components=components,
                                population=classify_population(quote)))
    labels = [_key(a.label) for a in arms]
    if len(set(labels)) != len(labels):
        return [], "two enumerated arms carry the same label"
    return arms, ""


def parse_comparisons(raw: object, paper_text: str) -> tuple[list[ComparisonEvidence], str]:
    """Validate the enumerated comparisons. BOTH sides must be in the quote."""
    if raw in (None, ""):
        return [], ""
    if not isinstance(raw, list):
        return [], "comparisons_reported is not a list"
    out: list[ComparisonEvidence] = []
    for item in raw:
        if not isinstance(item, dict):
            return [], "an entry in comparisons_reported is not a structured object"
        left = str(item.get("left_label") or "").strip()
        right = str(item.get("right_label") or "").strip()
        quote = str(item.get("quote") or "").strip()
        value = item.get("value")
        value = None if value is None else str(value).strip() or None
        if not left or not right:
            return [], "a comparison entry does not name both of its sides"
        if _key(left) == _key(right):
            return [], f"a comparison entry names {left!r} on both sides"
        if not quote:
            return [], f"the comparison {left!r} vs {right!r} has no quote"
        if not normalised_contains(paper_text, quote):
            return [], (f"the quote for {left!r} vs {right!r} is not a contiguous "
                        "passage of the source document")
        for side in (left, right):
            if not _label_in_quote(side, quote):
                return [], (f"the quote for {left!r} vs {right!r} does not name "
                            f"{side!r}, so the direction cannot be confirmed")
        if value is not None and not value_supported_by_quote(quote, value):
            return [], (f"the quote for {left!r} vs {right!r} does not contain "
                        f"{value!r}")
        components, component_error = parse_component_block(
            item.get("value_components"))
        if component_error:
            return [], (f"the entry for {left!r} vs {right!r} is unusable: "
                        f"{component_error}")
        out.append(ComparisonEvidence(left_label=left, right_label=right,
                                      value=value,
                                      unit=str(item.get("unit") or "").strip(),
                                      quote=quote, components=components,
                                      population=classify_population(quote)))
    return out, ""


# --- assignment ----------------------------------------------------------

def assign_arms(
    review_labels: dict[str, str], paper_labels: list[str],
    *, min_affinity: float = MIN_AFFINITY, min_margin: float = MIN_MARGIN,
) -> tuple[dict[str, str], str, float]:
    """Match the review's arms to the paper's arms, all at once.

    Returns ``(mapping, reason, margin)``. ``mapping`` is review key -> paper
    label and is empty when no single assignment is defensible.
    """
    keys = [k for k, v in review_labels.items() if (v or "").strip()]
    papers = [p for p in paper_labels if (p or "").strip()]
    if not keys or not papers:
        return {}, "there is nothing to assign on one of the two sides", 0.0
    if len(keys) > MAX_ARMS or len(papers) > MAX_ARMS:
        return {}, (f"more than {MAX_ARMS} arms on one side; this is not "
                    "assigned deterministically"), 0.0

    score = {(k, p): label_affinity(review_labels[k], p) for k in keys for p in papers}
    scored: list[tuple[float, dict[str, str]]] = []
    short, long_ = (keys, papers) if len(keys) <= len(papers) else (papers, keys)
    keys_first = short is keys
    for combo in permutations(long_, len(short)):
        pairs = list(zip(short, combo)) if keys_first else [(b, a) for a, b in zip(short, combo)]
        kept = {k: p for k, p in pairs if score[(k, p)] >= min_affinity}
        if not kept:
            continue
        scored.append((sum(score[(k, p)] for k, p in kept.items()), kept))
    if not scored:
        return {}, "no paper arm resembles any arm the review reports", 0.0

    scored.sort(key=lambda item: -item[0])
    best_total, best = scored[0]
    rivals = [total for total, mapping in scored[1:] if mapping != best]
    margin = best_total - rivals[0] if rivals else best_total
    if rivals and margin < min_margin:
        return {}, ("two different readings of the paper's arms fit the review "
                    "equally well, so which arm this value belongs to is not "
                    "established"), margin
    return best, "", margin


def values_consistent(
    text: str, claims: list[tuple[str, str]], *, min_gap: int = 4,
) -> tuple[bool, str]:
    """Whether each value sits next to the arm it was attributed to.

    ``claims`` is (arm label, value). One sentence commonly lists every arm's
    value, so the check is again global: the reading that keeps every value
    closest to its own arm must be the reading the extraction claims.
    """
    pairs = [(label, value) for label, value in claims
             if label and value and _numeric_anchor(value)]
    if len(pairs) < 2:
        return True, ""      # nothing to confuse it with
    flat = flatten(text)
    spans = _masked_label_spans(flat, [label for label, _ in pairs])
    positions = {value: _value_positions(flat, value) for _, value in pairs}
    # Arms this text does not mention cannot be confused with anything in it.
    # They are dropped rather than used to skip the check, so one absent arm
    # cannot switch the guard off for the arms that ARE side by side here.
    pairs = [(label, value) for label, value in pairs
             if spans.get(label) and positions[value]]
    if len(pairs) < 2:
        return True, ""

    labels = [label for label, _ in pairs]
    values = [value for _, value in pairs]
    best_cost, best_order, second = None, None, None
    for order in permutations(values):
        cost = sum(_distance(spans[label], positions[value])
                   for label, value in zip(labels, order))
        if best_cost is None or cost < best_cost:
            best_cost, second, best_order = cost, best_cost, order
        elif second is None or cost < second:
            second = cost
    if best_order != tuple(values):
        closest = dict(zip(labels, best_order))
        wrong = next(label for label, value in pairs if closest[label] != value)
        return False, (f"in the quoted text the value nearest {wrong!r} is "
                       f"{closest[wrong]!r}, not the {dict(pairs)[wrong]!r} "
                       "attributed to it")
    if second is not None and second - best_cost < min_gap:
        return False, ("the quoted text places these values too close to two "
                       "different arms to say which value belongs to which")
    return True, ""


# --- resolving one request ----------------------------------------------

def resolve_arm_target(
    *, target_key: str, review_labels: dict[str, str], arms: list[ArmEvidence],
    quote_text: str = "",
) -> TargetAssignment:
    """Which enumerated arm is the requested one, and what did it report."""
    if not arms:
        return TargetAssignment(
            status="not_reported",
            reason="the extraction listed no arm for this field, so no value "
                   "could be attributed to the requested arm")
    if target_key not in review_labels or not (review_labels.get(target_key) or "").strip():
        return TargetAssignment(
            status="unsupported",
            reason=f"the review provides no label for cohort {target_key!r}, so "
                   "the paper's arms cannot be matched against it")

    mapping, reason, margin = assign_arms(review_labels, [a.label for a in arms])
    if not mapping:
        return TargetAssignment(status="ambiguous", reason=reason, margin=margin)
    paper_label = mapping.get(target_key, "")
    if not paper_label:
        return TargetAssignment(
            status="not_reported", margin=margin, mapping=mapping,
            reason=(f"none of the arms the paper reports for this field matches "
                    f"{review_labels[target_key]!r}"))

    arm = next(a for a in arms if a.label == paper_label)
    if arm.value is None:
        return TargetAssignment(
            status="not_reported", paper_label=paper_label, margin=margin,
            mapping=mapping,
            reason=f"the paper's {paper_label!r} row reports no value for this field")

    ok, why = values_consistent(
        quote_text or arm.quote, [(a.label, a.value) for a in arms if a.value])
    if not ok:
        return TargetAssignment(status="inconsistent", paper_label=paper_label,
                                margin=margin, mapping=mapping, reason=why)
    return TargetAssignment(
        status=_STATUS_OK, paper_label=paper_label, value=arm.value,
        unit=arm.unit, quote=arm.quote, margin=margin, mapping=mapping,
        components=arm.components, population=arm.population,
        rival_values=[a.value for a in arms if a is not arm and a.value])


def resolve_comparison_target(
    *, comparison: ComparisonTarget, review_labels: dict[str, str],
    comparisons: list[ComparisonEvidence],
) -> TargetAssignment:
    """Which enumerated comparison is the requested pair, in the requested order."""
    if not comparisons:
        return TargetAssignment(
            status="not_reported",
            reason="the extraction listed no comparison for this field, so the "
                   "requested comparison could not be located")

    sides, why = resolve_sides(comparison, review_labels)
    if not sides:
        return TargetAssignment(status="unsupported", reason=why)
    left_label, right_label = sides

    # Score every enumerated comparison in the requested direction, and in the
    # inverted one. An inverted hazard ratio is a DIFFERENT number, so a paper
    # that only reports the mirror image has not answered this question.
    direct = [(_pair_score(left_label, right_label, c), c) for c in comparisons]
    inverted = [(_pair_score(right_label, left_label, c), c) for c in comparisons]
    direct.sort(key=lambda item: -item[0])
    inverted.sort(key=lambda item: -item[0])

    best, candidate = direct[0]
    runner_up = direct[1][0] if len(direct) > 1 else 0.0
    if best <= 0.0:
        if inverted and inverted[0][0] > 0.0:
            return TargetAssignment(
                status="direction_inverted", margin=inverted[0][0],
                reason=(f"the paper reports this comparison the other way round "
                        f"({inverted[0][1].left_label!r} versus "
                        f"{inverted[0][1].right_label!r}); the inverse is a "
                        "different quantity and is not returned as this one"))
        return TargetAssignment(
            status="not_reported",
            reason=(f"no comparison the paper reports matches {left_label!r} "
                    f"versus {right_label!r}"))
    if best - runner_up < MIN_MARGIN:
        return TargetAssignment(
            status="ambiguous", margin=best - runner_up,
            reason=("two of the paper's comparisons fit the requested pair "
                    "equally well, so the value cannot be attributed to one"))
    if inverted[0][0] > best:
        return TargetAssignment(
            status="direction_inverted", margin=inverted[0][0] - best,
            reason=("the closest match reports the comparison in the opposite "
                    "direction, which is a different quantity"))
    if candidate.value is None:
        return TargetAssignment(
            status="not_reported", paper_label=_pair_name(candidate),
            reason="the matching comparison reports no value")
    return TargetAssignment(
        status=_STATUS_OK, paper_label=_pair_name(candidate),
        value=candidate.value, unit=candidate.unit, quote=candidate.quote,
        margin=best - runner_up,
        mapping={comparison.left: candidate.left_label,
                 comparison.right: candidate.right_label},
        components=candidate.components, population=candidate.population,
        rival_values=[c.value for c in comparisons if c is not candidate and c.value])


# --- helpers -------------------------------------------------------------

def _key(label: str) -> str:
    return " ".join(sorted(distinguishing_tokens(label))) or label.strip().lower()


def _label_in_quote(label: str, quote: str) -> bool:
    """Whether the quote actually NAMES this arm.

    The arm's words must appear together and as a whole name. A loose token
    check accepts "the Y group" as evidence for a sentence that only says "the
    X-plus-Y group", which is precisely the confusion that put one arm's value on
    another arm's row.
    """
    pattern = _label_pattern(label)
    if pattern is None:
        return True
    return pattern.search(flatten(quote)) is not None


def resolve_sides(
    comparison: ComparisonTarget, review_labels: dict[str, str],
) -> tuple[tuple[str, str] | None, str]:
    """The review's words for each side of a comparison — resolved TOGETHER.

    Side by side does not work: a trial's short name for its control arm ("Y")
    fits both the control arm ("Y + placebo") and the combination arm ("X + Y")
    exactly as well, and each side's own best guess is a coin flip. The two sides
    are one problem — they must land on DIFFERENT arms — so they are assigned
    jointly, exactly as the arms themselves are.
    """
    labels = [label for label in review_labels.values() if (label or "").strip()]
    if len(labels) < 2:
        return None, (f"the review reports fewer than two arms, so "
                      f"{comparison.raw!r} cannot be resolved to a pair")
    if len(labels) > MAX_ARMS:
        return None, (f"more than {MAX_ARMS} arms; the comparison "
                      f"{comparison.raw!r} is not resolved deterministically")

    pinned = {
        slot: next((review_labels[k] for k in review_labels
                    if _slug(k) == _slug(side) and (review_labels[k] or "").strip()),
                   "")
        for slot, side in (("left", comparison.left), ("right", comparison.right))
    }
    scored: list[tuple[float, tuple[str, str]]] = []
    for left, right in permutations(labels, 2):
        if (pinned["left"] and left != pinned["left"]) or (
                pinned["right"] and right != pinned["right"]):
            continue
        score = (_side_score(comparison.left, left, other=comparison.right)
                 + _side_score(comparison.right, right, other=comparison.left))
        scored.append((score, (left, right)))
    if not scored:
        return None, (f"the comparison {comparison.raw!r} does not resolve to "
                      "two distinct arms this review reports")

    scored.sort(key=lambda item: -item[0])
    best, pair = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best <= 0.0:
        return None, (f"neither side of {comparison.raw!r} matches an arm this "
                      "review reports")
    if len(scored) > 1 and best - runner_up < MIN_MARGIN:
        return None, (f"the sides of {comparison.raw!r} fit two different pairs "
                      "of this review's arms equally well")
    return pair, ""


def _side_score(side: str, label: str, *, other: str) -> float:
    """How well one comparison side matches one review arm.

    With a penalty for contamination: if the arm's label carries the OTHER
    side's agent while this side's own name does not, the assignment has quietly
    turned "X versus Y" into a comparison involving the combination arm. That is
    a different claim, so it must not win a tie.
    """
    score = label_affinity(side, label)
    if score <= 0.0:
        return 0.0
    side_tokens = distinguishing_tokens(side)
    other_tokens = distinguishing_tokens(other)
    acquired = other_tokens - side_tokens
    if acquired and acquired <= distinguishing_tokens(label):
        return score / 2
    return score


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def _pair_score(left: str, right: str, candidate: ComparisonEvidence) -> float:
    """How well a reported comparison IS the requested one, both sides summed.

    Summed rather than min-ed: two candidates can share their weaker side (both
    are "…versus the Y group") and be told apart only by the stronger one. A
    floor on each side separately stops a lopsided fit from winning.
    """
    scores = (label_affinity(left, candidate.left_label),
              label_affinity(right, candidate.right_label))
    if min(scores) < MIN_PAIR_SIDE:
        return 0.0
    return sum(scores)


def _pair_name(candidate: ComparisonEvidence) -> str:
    return f"{candidate.left_label} vs {candidate.right_label}"


def _masked_label_spans(flat: str, labels: list[str]) -> dict[str, list[tuple[int, int]]]:
    """Where each label occurs, longest label first so it claims its own text.

    Without masking, "Y group" matches inside "X-plus-Y group" and every locality
    measurement collapses.
    """
    remaining = flat
    spans: dict[str, list[tuple[int, int]]] = {}
    for label in sorted(labels, key=lambda l: -len(distinguishing_tokens(l))):
        pattern = _label_pattern(label)
        found: list[tuple[int, int]] = []
        if pattern is not None:
            for match in pattern.finditer(remaining):
                found.append((match.start(), match.end()))
        spans[label] = found
        for start, end in found:
            remaining = remaining[:start] + ("\x00" * (end - start)) + remaining[end:]
    return spans


def _label_pattern(label: str) -> re.Pattern | None:
    tokens = [t for t in re.findall(r"[a-z0-9]+", flatten(label))
              if t in distinguishing_tokens(label)]
    if not tokens:
        return None
    body = r"[^a-z0-9]{0,3}".join(re.escape(t) for t in tokens)
    # The boundary excludes a neighbouring hyphen as well as letters: in
    # "X-plus-Y group" the tail really does read "Y group", and treating that as
    # an occurrence of the Y arm is the whole failure. A hyphenated compound is
    # one name.
    return re.compile(rf"(?<![a-z0-9-]){body}(?![a-z0-9-])")


def _numeric_anchor(value: str) -> str:
    match = re.search(r"-?\d+(?:\.\d+)?", value or "")
    return match.group(0) if match else ""


def _value_positions(flat: str, value: str) -> list[int]:
    anchor = _numeric_anchor(value)
    if not anchor:
        return []
    return [m.start() for m in
            re.finditer(rf"(?<![0-9.]){re.escape(anchor)}(?![0-9])", flat)]


def _distance(spans: list[tuple[int, int]], positions: list[int]) -> int:
    best = None
    for start, end in spans:
        for position in positions:
            gap = 0 if start <= position <= end else min(abs(position - start),
                                                         abs(position - end))
            best = gap if best is None else min(best, gap)
    return best if best is not None else 10**6
