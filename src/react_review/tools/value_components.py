"""The parts of a reported value, kept as parts.

Phase 6B's second failure: asked for a hazard ratio the extractor returned
``0.42`` from a sentence that prints ``0.42; 99.5% CI, 0.31 to 0.57``. The
interval was in the evidence and was dropped on the way out, so the comparator
saw a bare point estimate — indistinguishable from a paper that reports no
interval at all — and the review's own interval went unchecked.

So the extraction returns the components explicitly, and this module decides
whether to believe them:

*Anchoring.* Every component must be printed in the same quote as the value, and
that quote has already been checked against the paper. Nothing is inferred and
nothing is computed.

*Attribution.* A sentence often carries two intervals ("0.42; 99.5% CI, 0.31 to
0.57 … and … 0.57; 99.5% CI, 0.43 to 0.76"). An interval counts as this value's
only when this value is the estimate it sits nearest to, so one comparison's
bounds cannot be handed to another.

*Completeness.* If the quote declares an interval the response did not return,
the extraction is INCOMPLETE — partial and review-required. It is never filled
in from the text: a value the model did not report is not a value it read.
"""
from __future__ import annotations

import re

from react_review.normalize.anchors import flatten
from react_review.schemas.evidence import SourceNumericComponents

# "95% CI, 8.9 to 16.7" · "99.5% CI, 0.31 to 0.57" · "95% confidence interval
# [CI], 4.3 to 9.5" · "95% CI 0.60-0.92". The level needs two digits, so a
# percentage that is itself the value ("37.5%") cannot be read as one.
_INTERVAL = re.compile(
    r"(?P<level>\d{2}(?:\.\d+)?)\s*%\s*(?:ci\b|confidence interval)"
    r"[^0-9+-]{0,15}"
    r"(?P<lower>-?\d+(?:\.\d+)?)\s*(?:to|–|—|-|,)\s*(?P<upper>-?\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

_COMPONENT_KEYS = ("point_estimate", "ci_level", "ci_lower", "ci_upper")

OK = "ok"
INCOMPLETE = "incomplete"
PROTOCOL_ERROR = "protocol_error"


def parse_component_block(raw: object) -> tuple[dict[str, float], str]:
    """Read the model's component numbers. Anything unreadable is refused."""
    if raw in (None, "", {}):
        return {}, ""
    if not isinstance(raw, dict):
        return {}, "value_components is not a structured object"
    out: dict[str, float] = {}
    for key in _COMPONENT_KEYS:
        value = raw.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        number = _as_number(value)
        if number is None:
            return {}, f"{key} is not a number: {value!r}"
        out[key] = number
    return out, ""


def verify_components(
    claimed: dict[str, float], *, value: str, quote: str,
    rival_values: list[str] | None = None,
) -> tuple[SourceNumericComponents, str, str]:
    """Check the claimed components against the value and its quote.

    Returns ``(components, status, reason)`` with status ``ok`` |
    ``incomplete`` | ``protocol_error``.
    """
    point = _leading_number(value)
    flat_quote = flatten(quote)
    anchored: dict[str, bool] = {}

    claimed_point = claimed.get("point_estimate")
    if claimed_point is not None and point is not None and claimed_point != point:
        return (SourceNumericComponents(), PROTOCOL_ERROR,
                f"the response reports a point estimate of {claimed_point:g} but "
                f"returns {value!r} as the value")
    if claimed_point is None:
        claimed_point = point
    anchored["point_estimate"] = _prints(flat_quote, claimed_point)
    if claimed_point is not None and not anchored["point_estimate"]:
        return (SourceNumericComponents(), PROTOCOL_ERROR,
                f"the point estimate {claimed_point:g} is not printed in the "
                "supporting quote")

    printed = _interval_for(flat_quote, claimed_point, rival_values or [])
    bounds = {key: claimed.get(key) for key in ("ci_level", "ci_lower", "ci_upper")}
    given = {key: number for key, number in bounds.items() if number is not None}

    if given:
        for key, number in given.items():
            anchored[key] = _prints(flat_quote, number)
            if not anchored[key]:
                return (SourceNumericComponents(), PROTOCOL_ERROR,
                        f"{key.replace('_', ' ')} {number:g} is not printed in the "
                        "supporting quote")
        if printed is None:
            return (SourceNumericComponents(), PROTOCOL_ERROR,
                    "the response reports an interval that the supporting quote "
                    "does not state for this value")
        for key, number in given.items():
            if printed.get(key) is not None and printed[key] != number:
                return (SourceNumericComponents(), PROTOCOL_ERROR,
                        f"the quote states {key.replace('_', ' ')} "
                        f"{printed[key]:g} for this value, not {number:g}")

    missing = [key for key in ("ci_level", "ci_lower", "ci_upper")
               if printed and printed.get(key) is not None and key not in given]
    components = SourceNumericComponents(
        point_estimate=claimed_point,
        ci_level=given.get("ci_level"), ci_lower=given.get("ci_lower"),
        ci_upper=given.get("ci_upper"),
        anchored={k: v for k, v in anchored.items() if k},
        missing=missing)
    if missing:
        # Deliberately NOT filled in from the quote. The point of the check is to
        # know what the extraction actually read; silently completing it would
        # give a partial answer the credit of a complete one.
        components.status = INCOMPLETE
        components.reason = (
            "the quote states " + ", ".join(m.replace("_", " ") for m in missing)
            + " for this value, which the extraction did not return")
        return components, INCOMPLETE, components.reason
    components.status = OK
    return components, OK, ""


def canonical_value(components: SourceNumericComponents) -> str:
    """The components written back as one comparable string, or ""."""
    if components.point_estimate is None:
        return ""
    text = _trim(components.point_estimate)
    if components.complete_interval:
        text += (f" ({_trim(components.ci_level)}% CI "
                 f"{_trim(components.ci_lower)}-{_trim(components.ci_upper)})")
    return text


def quote_states_interval(quote: str, value: str,
                          rival_values: list[str] | None = None) -> bool:
    """Whether the quote prints an interval belonging to this value."""
    return _interval_for(flatten(quote), _leading_number(value),
                         rival_values or []) is not None


def _interval_for(flat_quote: str, point: float | None,
                  rival_values: list[str]) -> dict[str, float] | None:
    """The interval this point estimate owns, if any.

    With several estimates in one sentence the nearest interval to THIS estimate
    is only its own when no other estimate is nearer to that interval. Otherwise
    the sentence does not settle whose interval it is, and nothing is claimed.
    """
    if point is None:
        return None
    matches = list(_INTERVAL.finditer(flat_quote))
    if not matches:
        return None
    positions = _positions(flat_quote, point)
    if not positions:
        return None
    rivals = [number for number in
              (_leading_number(v) for v in rival_values)
              if number is not None and number != point]
    rival_positions = [p for number in rivals for p in _positions(flat_quote, number)]

    best = min(matches, key=lambda m: min(abs(m.start() - p) for p in positions))
    mine = min(abs(best.start() - p) for p in positions)
    if rival_positions:
        theirs = min(abs(best.start() - p) for p in rival_positions)
        if theirs <= mine:
            return None
    return {"ci_level": float(best.group("level")),
            "ci_lower": float(best.group("lower")),
            "ci_upper": float(best.group("upper"))}


def _positions(flat: str, number: float) -> list[int]:
    """Where this number is printed, compared as a NUMBER not as a string.

    A paper writes "0.60" and a JSON response carries 0.6; a string match would
    call that a fabricated bound and reject a correct extraction. Trailing
    zeros, a leading "+", and "0.42" against ".42" are all the same quantity.
    """
    return [match.start() for match in _NUMBER.finditer(flat)
            if _equal(float(match.group(0)), number)]


def _equal(one: float, other: float) -> bool:
    return abs(one - other) <= max(abs(one), abs(other), 1.0) * 1e-9


def _prints(flat: str, number: float | None) -> bool:
    return number is not None and bool(_positions(flat, number))


def _leading_number(value: str | None) -> float | None:
    match = _NUMBER.search(str(value or ""))
    return float(match.group(0)) if match else None


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = _NUMBER.search(str(value))
    return float(match.group(0)) if match else None


def _trim(number: float | None) -> str:
    if number is None:
        return ""
    text = f"{number:g}"
    return text
