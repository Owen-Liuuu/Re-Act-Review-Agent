"""Deterministic verification of an LLM-proposed field_type mapping.

The DKB agent PROPOSES a field_type on a knowledge-base miss — a hypothesis,
never a fact. Before the resolver accepts it as a usable CANDIDATE, these
deterministic checks must pass; otherwise the mapping is rejected and the field
is kept UNRESOLVED (needs human review). No LLM here: the LLM understands, this
code judges its guess against the knowledge base.

Checks
  grounding : evidence-backed — the LLM cited retrieved candidates (or mapped to
              an existing concept), or is confident enough (>= min_confidence).
              A blind, low-confidence invention is rejected.
  unit      : if the chosen field_type is a KNOWN concept with an expected unit,
              the column's unit must be the same KIND (length vs volume vs mass).
              mm vs cm is fine (both length; a scale mismatch caught at compare);
              cm3 for a length concept is a CONTRADICTION → concept confusion.
  range     : a numeric value outside the concept's plausible_range is
              implausible for that mapping.

:func:`evidence_contradicts` reruns the unit + range checks against the value and
unit the Collector later extracts FROM THE SOURCE — closing the loop so the
source evidence validates (or refutes) the translation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from react_review.dkb.base import KnowledgeBase
from react_review.dkb.schema import KnowledgeEntry
from react_review.normalize.numeric import primary_number
from react_review.normalize.units import normalize_unit, unit_kind


@dataclass(frozen=True)
class CandidateVerdict:
    """Outcome of verifying one proposed mapping. ``ok`` gates acceptance."""

    ok: bool
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)


def _unit_conflict(unit: str, entry: KnowledgeEntry) -> bool:
    """True when the column unit is a DIFFERENT physical kind than the concept's."""
    if not unit or not entry.default_unit:
        return False
    nu = normalize_unit(unit)
    expected = {normalize_unit(entry.default_unit),
                *(normalize_unit(u) for u in entry.unit_equivalences)}
    expected.discard("")
    if nu in expected:
        return False                       # exact / equivalent unit — fine
    ku, ke = unit_kind(unit), unit_kind(entry.default_unit)
    if ku == "unknown" or ke == "unknown":
        return False                       # can't judge → never reject
    return ku != ke


def _range_conflict(value: object, entry: KnowledgeEntry) -> tuple[bool, float | None]:
    if not entry.plausible_range or value is None:
        return False, None
    p = primary_number(value)
    if p is None:
        return False, None
    lo, hi = entry.plausible_range[0], entry.plausible_range[1]
    return (not lo <= p <= hi), p


def verify_candidate(
    field_type: str,
    *,
    kb: KnowledgeBase,
    unit: str = "",
    value: object = None,
    is_new: bool,
    confidence: float,
    grounded_on: list[str],
    min_confidence: float = 0.35,
) -> CandidateVerdict:
    """Judge whether an LLM-proposed ``field_type`` may be used as a candidate."""
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    grounded = bool(grounded_on) or not is_new
    checks["grounding"] = grounded or confidence >= min_confidence
    if not checks["grounding"]:
        reasons.append(
            f"ungrounded low-confidence guess (confidence={confidence:.2f}, "
            f"grounded_on={grounded_on or []})")

    checks["unit"] = True
    checks["range"] = True
    entry = kb.entries.get(field_type)
    if entry is not None:                  # only a KNOWN concept has an expectation
        if _unit_conflict(unit, entry):
            checks["unit"] = False
            reasons.append(
                f"unit '{unit}' is a different kind than {field_type} expects "
                f"('{entry.default_unit}')")
        bad_range, p = _range_conflict(value, entry)
        if bad_range:
            checks["range"] = False
            lo, hi = entry.plausible_range
            reasons.append(
                f"value {p} outside plausible range [{lo}, {hi}] for {field_type}")

    return CandidateVerdict(ok=all(checks.values()), reason="; ".join(reasons),
                            checks=checks)


def evidence_contradicts(
    field_type: str, *, kb: KnowledgeBase, unit: str = "", value: object = None,
) -> str:
    """Back-check: does SOURCE-extracted unit/value contradict this field_type?

    Returns a human-readable reason when it does, else ``""``. Unit + range only —
    grounding was a parse-time signal. Lets the Collector use source evidence to
    validate a CANDIDATE translation (extraction refutes a bad guess).
    """
    entry = kb.entries.get(field_type)
    if entry is None:
        return ""
    if _unit_conflict(unit, entry):
        return (f"source unit '{unit}' is a different kind than {field_type} "
                f"expects ('{entry.default_unit}')")
    bad_range, p = _range_conflict(value, entry)
    if bad_range:
        lo, hi = entry.plausible_range
        return f"source value {p} outside plausible range [{lo}, {hi}] for {field_type}"
    return ""
