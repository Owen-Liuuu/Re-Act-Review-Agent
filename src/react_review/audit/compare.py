"""``compare_values`` — the deterministic audit verdict for one value pair.

Rule (Tier-3), in order:
  1. Units present and different            -> UNIT_MISMATCH
  2. Either primary value unparseable       -> NOT_COMPARABLE
  3. relative error <= field tolerance       -> MATCH
  4. otherwise                               -> MISMATCH

The relative error is ``abs(a-b)/max(|a|,|b|,eps)`` on the primary numeric value
(the central estimate), matching how the benchmark's ``expected_label`` was
derived. Unit handling is a separate axis and takes precedence, so a value that
is numerically identical but reported in a different unit is a UNIT_MISMATCH.
"""
from __future__ import annotations

from react_review.core.enums import AuditLabel
from react_review.normalize import normalize_unit, primary_number, units_differ
from react_review.schemas.audit import MatchResult

_EPS = 1e-9

Value = str | int | float | None


def compare_values(
    *,
    field_type: str,
    review_value: Value,
    source_value: Value,
    review_unit: str = "",
    source_unit: str = "",
    rel_tolerance: float = 0.01,
    audit_id: str = "",
    study_id: str = "",
    group: str = "-",
) -> MatchResult:
    """Compare a review value against a source value and label the outcome.

    Args:
        field_type: canonical concept (used only for reporting here; the
            caller resolves the tolerance and passes ``rel_tolerance``).
        review_value / source_value: verbatim cell strings or numbers.
        review_unit / source_unit: reported units (blank = not asserted).
        rel_tolerance: relative-error MATCH bound for this field_type.

    Returns:
        A :class:`MatchResult` with the label, relative error, and a reason.
    """
    base = dict(
        audit_id=audit_id,
        study_id=study_id,
        group=group,
        field_type=field_type,
        review_value=review_value,
        review_unit=review_unit,
        source_value=source_value,
        source_unit=source_unit,
        tolerance_pct=round(rel_tolerance * 100.0, 4),
    )

    # 1. Unit axis takes precedence.
    if units_differ(review_unit, source_unit):
        return MatchResult(
            **base,
            label=AuditLabel.UNIT_MISMATCH,
            rel_error_pct=None,
            reason=(
                f"unit differs (review {normalize_unit(review_unit)} "
                f"vs source {normalize_unit(source_unit)})"
            ),
        )

    # 2. Parse primary values.
    a = primary_number(review_value)
    b = primary_number(source_value)
    if a is None or b is None:
        return MatchResult(
            **base,
            label=AuditLabel.NOT_COMPARABLE,
            rel_error_pct=None,
            reason="a primary numeric value could not be parsed",
        )

    # 3/4. Relative-error band.
    rel = abs(a - b) / max(abs(a), abs(b), _EPS)
    rel_pct = round(rel * 100.0, 4)
    if rel <= rel_tolerance:
        return MatchResult(
            **base,
            label=AuditLabel.MATCH,
            rel_error_pct=rel_pct,
            reason=f"{a} vs {b} = {rel_pct}% <= {base['tolerance_pct']}%",
        )
    return MatchResult(
        **base,
        label=AuditLabel.MISMATCH,
        rel_error_pct=rel_pct,
        reason=f"{a} vs {b} = {rel_pct}% > {base['tolerance_pct']}%",
    )
