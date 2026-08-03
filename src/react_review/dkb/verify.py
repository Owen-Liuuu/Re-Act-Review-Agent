"""Deterministic verification of an LLM-proposed field_type mapping.

The DKB agent PROPOSES a field_type on a knowledge-base miss — a hypothesis,
never a fact. Before the resolver accepts it as a usable CANDIDATE, these
deterministic checks must pass; otherwise the mapping is rejected and the field
is kept UNRESOLVED (needs human review). No LLM here: the LLM understands, this
code judges its guess against the knowledge base.

Checks
  self-contract : for a NEW concept, the proposal's declared value type, unit
                  and scope must not contradict the value actually observed.
                  Model confidence is recorded but never used as a gate.
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


def _new_concept_checks(
    proposal: KnowledgeEntry | None, *, unit: str, value: object,
) -> tuple[dict[str, bool], list[str]]:
    """Make a new proposal obey the contract it declared itself.

    These checks establish absence of a deterministic contradiction; they do
    not establish that the proposed clinical concept is correct.
    """
    checks = {
        "proposal_present": proposal is not None,
        "declared_value_type": False,
        "observed_value_type": False,
        "declared_scope": False,
        "declared_unit": False,
    }
    if proposal is None:
        return checks, ["new concept did not include a proposal contract"]

    reasons: list[str] = []
    value_type = (proposal.value_type or "").strip().lower()
    checks["declared_value_type"] = value_type in {"numeric", "text", "categorical"}
    if not checks["declared_value_type"]:
        reasons.append(f"unsupported declared value_type {proposal.value_type!r}")

    # Numeric is the declaration that can be contradicted safely. Text and
    # categorical fields may legitimately contain numeric codes, so a number
    # alone is not enough to reject either of those declarations.
    checks["observed_value_type"] = (
        value is None
        or value_type in {"text", "categorical"}
        or (value_type == "numeric" and primary_number(value) is not None)
    )
    if (value is not None and checks["declared_value_type"]
            and not checks["observed_value_type"]):
        reasons.append(
            f"declared value_type 'numeric' contradicts observed value {value!r}")

    checks["declared_scope"] = proposal.scope in {"study", "cohort"}
    if not checks["declared_scope"]:
        reasons.append(f"unsupported declared scope {proposal.scope!r}")

    observed, declared = normalize_unit(unit), normalize_unit(proposal.default_unit)
    if observed and not declared:
        checks["declared_unit"] = False
    elif not observed or observed == declared:
        checks["declared_unit"] = True
    else:
        observed_kind, declared_kind = unit_kind(unit), unit_kind(proposal.default_unit)
        # Unknown compound/custom units cannot be judged safely. Known physical
        # kinds, however, must agree (length vs volume must never pass).
        checks["declared_unit"] = (
            observed_kind == "unknown" or declared_kind == "unknown"
            or observed_kind == declared_kind)
    if not checks["declared_unit"]:
        if not declared:
            reasons.append(
                f"proposal omitted default_unit despite observed unit {unit!r}")
        else:
            reasons.append(
                f"declared unit {proposal.default_unit!r} contradicts observed unit {unit!r}")
    return checks, reasons


def verify_candidate(
    field_type: str,
    *,
    kb: KnowledgeBase,
    unit: str = "",
    value: object = None,
    is_new: bool,
    confidence: float,
    grounded_on: list[str],
    proposal: KnowledgeEntry | None = None,
    min_confidence: float = 0.35,
) -> CandidateVerdict:
    """Judge whether an LLM-proposed ``field_type`` may be used as a candidate."""
    checks: dict[str, bool] = {}
    reasons: list[str] = []

    # Kept in the signature for API compatibility and provenance only. GLM's
    # self-reported confidence was empirically constant across opposing answers,
    # so it must not decide whether a proposal is accepted.
    _ = confidence, min_confidence, grounded_on
    checks["confidence_not_used"] = True

    if is_new:
        contract_checks, contract_reasons = _new_concept_checks(
            proposal, unit=unit, value=value)
        checks.update(contract_checks)
        reasons.extend(contract_reasons)

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
