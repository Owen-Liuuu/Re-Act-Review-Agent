"""Deterministic control over a model's equivalence claim.

The model is used for what it is good at — reading two phrasings and saying
whether they mean the same thing — and then its answer is CHECKED rather than
trusted. The checks are cheap, reproducible, and independent of the model:

  numeric non-drift   two different numbers are never "the same thing", whatever
                      the rationale says. This is the load-bearing one: it keeps
                      the whole numeric rigour of the audit intact.
  evidence anchoring  the span the model cites must actually occur in the quote,
                      so it cannot invent what the source says.
  polarity            a negation on exactly one side is a difference.
  confidence          below the band, or a relation of "unknown", is not a verdict.

Anything the checks reject becomes NOT_COMPARABLE with the reason attached, and
anything they accept as a broader/narrower match is flagged for a human — the
audit never quietly records "France, surgical ICU" as "France".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from react_review.core.enums import AuditLabel
from react_review.normalize.numeric import parse_numeric
from react_review.schemas.semantic import SemanticVerdict

# Provisional: chosen to be roughly sensible, NOT calibrated. A model's stated
# confidence is not a probability, so treat movements around these as untested.
DEFAULT_MIN_CONFIDENCE = 0.70
DEFAULT_MIN_ANCHOR = 0.60

_NEGATION = re.compile(
    r"(?<![a-z])(no|not|none|never|without|absent|absence|non|un|neither|nor)(?![a-z])",
    re.I)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class ControlOutcome:
    """What the controls decided, and which one had the last word."""

    label: AuditLabel
    reason: str
    review_required: bool = False
    failed_control: str = ""
    checks: dict[str, bool] = field(default_factory=dict)


def _normalize(text: str) -> str:
    """Case, spacing and punctuation folded — an exact byte match would reject
    correct evidence over a PDF's spacing or a hyphen."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def anchored_in(span: str, quote: str) -> bool:
    """Whether the cited span really is a contiguous piece of the quote."""
    if not span:
        return False
    if not quote:
        return False
    return _normalize(span) in _normalize(quote)


def numbers_agree(review_value: str, source_value: str, rel_tolerance: float) -> bool:
    """Whether every number present on both sides matches, component by component.

    Uses the structured parse, so ``45/120 (37.5%)`` is compared as counts and a
    percentage rather than as a bag of three unrelated numbers.
    """
    rv, sv = parse_numeric(review_value), parse_numeric(source_value)
    r_pct = rv.pct if rv.pct is not None else rv.derived_pct
    s_pct = sv.pct if sv.pct is not None else sv.derived_pct

    pairs: list[tuple[float, float]] = [
        (a, b) for a, b in ((r_pct, s_pct), (rv.ci_lower, sv.ci_lower),
                            (rv.ci_upper, sv.ci_upper))
        if a is not None and b is not None
    ]
    # The leading number of "45/120 (37.5%)" is an event count, not a value —
    # comparing it against a bare "37.5" is comparing different quantities.
    proportion = rv.events is not None or sv.events is not None
    if not proportion and rv.primary is not None and sv.primary is not None:
        pairs.append((rv.primary, sv.primary))
    if rv.events is not None and sv.events is not None and rv.events != sv.events:
        return False
    if not pairs:
        # Numbers on one side only: nothing to contradict, but nothing to confirm.
        return not (_NUM_RE.search(review_value or "") and _NUM_RE.search(source_value or ""))
    return all(abs(a - b) <= max(abs(a), abs(b), 1e-9) * rel_tolerance for a, b in pairs)


# Which specificity direction each relation asserts. A verdict that answers the
# two questions inconsistently has refuted itself: whichever half is wrong, the
# pair cannot be believed, and a broader relation is NOT harmless — it is the
# case where the review has narrowed or widened what the source actually says.
_RELATION_SIDES = {
    "same": {"neither"},
    "review_broader": {"source"},      # review less specific ⇒ source says more
    "source_broader": {"review"},      # source less specific ⇒ review says more
    "different": {"unknown", "neither"},
    "unknown": {"unknown"},
}


def direction_consistent(verdict: SemanticVerdict) -> tuple[bool, str]:
    """Whether a verdict agrees with itself about direction and equivalence.

    Deterministic, and independent of how the rationale happens to be worded —
    parsing "the review is more specific" out of free text breaks on a negation,
    a reordering, or a model that answers in another language.
    """
    relation = (verdict.relation or "unknown").strip().lower()
    side = (verdict.more_specific_side or "").strip().lower()
    allowed = _RELATION_SIDES.get(relation)
    if allowed is None:
        return False, f"relation {verdict.relation!r} is not one this audit defines"
    # A verdict recorded before the direction contract did not state a side. It
    # cannot be checked, and calling that a contradiction would retroactively
    # convict every response in an existing recording. The equivalence checks
    # below still apply: they need nothing the older contract did not provide.
    if side and side not in allowed:
        return False, (
            f"the verdict says relation={relation} but names "
            f"{side!r} as the more specific side; "
            f"{relation} requires {' or '.join(sorted(allowed))}")
    if relation == "same" and not verdict.equivalent:
        return False, "the verdict says the values are the same but not equivalent"
    if relation == "different" and verdict.equivalent:
        return False, "the verdict says the values differ but are equivalent"
    return True, ""


def direction_stated(verdict: SemanticVerdict) -> bool:
    """Whether this verdict answered the direction question at all.

    Lets a control record "checked and consistent" separately from "there was
    nothing to check", instead of presenting an unasked question as a pass.
    """
    return bool((verdict.more_specific_side or "").strip())


def _polarity_differs(a: str, b: str) -> bool:
    return bool(_NEGATION.search(a or "")) != bool(_NEGATION.search(b or ""))


# The confidence band is a guess, so a run should say how much rests on it
# rather than present 0.70 as if it were measured.
DEFAULT_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)


def threshold_sensitivity(
    verdicts, thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID,
) -> dict[float, int]:
    """How many equivalence claims would be ACCEPTED at each candidate band.

    A flat curve means the threshold is not doing much and the verdicts stand on
    their own; a steep one means the result is an artefact of an uncalibrated
    number, and the reader deserves to know which of the two they are reading.
    """
    accepting = [v.confidence for v in verdicts
                 if v.relation in ("same", "review_broader", "source_broader")]
    return {t: sum(1 for c in accepting if c >= t) for t in thresholds}


def format_threshold_sensitivity(counts: dict[float, int], total: int) -> str:
    """One line: 'confidence 0.50:12  0.60:12  0.70:11  0.80:7  0.90:2 of 14'."""
    if not counts or not total:
        return ""       # nothing was judged; a row of zeroes would suggest otherwise
    cells = "  ".join(f"{t:.2f}:{n}" for t, n in sorted(counts.items()))
    return f"accepted at confidence  {cells}   (of {total} claim(s) judged)"


def apply_semantic_control(
    verdict: SemanticVerdict, *, review_value: str, source_value: str,
    source_quote: str = "", rel_tolerance: float = 0.01,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_anchor: float = DEFAULT_MIN_ANCHOR,
) -> ControlOutcome:
    """Turn a model's claim into an audit label — or refuse it."""
    del min_anchor                      # anchoring is containment, not a ratio
    checks: dict[str, bool] = {}

    # 1. Numeric non-drift — before anything else, and never overridable.
    checks["numeric"] = numbers_agree(review_value, source_value, rel_tolerance)
    if not checks["numeric"]:
        return ControlOutcome(
            AuditLabel.MISMATCH,
            "the values contain different numbers, so they are not the same thing "
            f"whatever the wording suggests (model said: {verdict.relation})",
            failed_control="numeric", checks=checks)

    # 2. Polarity.
    checks["polarity"] = not _polarity_differs(review_value, source_value)
    if not checks["polarity"]:
        return ControlOutcome(
            AuditLabel.MISMATCH,
            "one side is negated and the other is not",
            failed_control="polarity", checks=checks)

    # 3. Confidence and relation.
    relation = verdict.relation
    checks["confidence"] = verdict.confidence >= min_confidence
    if relation == "different":
        return ControlOutcome(AuditLabel.MISMATCH,
                              verdict.rationale or "the values denote different things",
                              checks=checks)
    if relation not in ("same", "review_broader", "source_broader") or not checks["confidence"]:
        return ControlOutcome(
            AuditLabel.NOT_COMPARABLE,
            f"equivalence not established (relation={relation or 'unknown'}, "
            f"confidence={verdict.confidence:.2f} < {min_confidence:.2f})"
            + (f": {verdict.rationale}" if verdict.rationale else ""),
            review_required=True, failed_control="confidence", checks=checks)

    # 4. Evidence anchoring — only meaningful when a quote was captured.
    if source_quote:
        checks["anchor"] = anchored_in(verdict.evidence_span, source_quote)
        if not checks["anchor"]:
            return ControlOutcome(
                AuditLabel.NOT_COMPARABLE,
                "the model's supporting span is not in the quoted source text, so "
                "the claim is not grounded in the paper",
                review_required=True, failed_control="anchor", checks=checks)

    if relation == "same":
        return ControlOutcome(AuditLabel.MATCH,
                              verdict.rationale or "the values denote the same thing",
                              checks=checks)
    # Broader/narrower IS a difference in what is claimed, even when defensible;
    # recording it as a plain match is how "France, surgical ICU" silently
    # becomes "France".
    return ControlOutcome(
        AuditLabel.MATCH,
        f"{relation.replace('_', ' ')}: {verdict.rationale}".strip().rstrip(":"),
        review_required=True, checks=checks)
