"""``compare_values`` — the deterministic audit verdict for one value pair.

Rule (Tier-3), in order:
  1. Units present and different                 -> UNIT_MISMATCH
  2. Either primary value unparseable            -> NOT_COMPARABLE
  3. MEAN relative error > mean tolerance         -> MISMATCH
  4. Both sides have an SD and the SD relative
     error > SD tolerance                         -> MISMATCH (SD-driven)
  5. otherwise                                    -> MATCH

Dual band: the mean (primary value) uses the tight band (MVP 1%); the SD uses a
separate, looser band (MVP 3%) and is only checked when BOTH sides report a
``mean ± sd`` spread — so it catches an SD transcription error (e.g. 1.7 vs
1.77) without false-flagging a value that only reports a range/IQR or a point.
Unit handling is a separate axis and takes precedence.
"""
from __future__ import annotations

from react_review.core.enums import AuditLabel
from react_review.normalize import normalize_unit, parse_numeric, units_differ
from react_review.schemas.audit import MatchResult

_EPS = 1e-9

Value = str | int | float | None


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), _EPS)


def compare_values(
    *,
    field_type: str,
    review_value: Value,
    source_value: Value,
    review_unit: str = "",
    source_unit: str = "",
    rel_tolerance: float = 0.01,
    sd_rel_tolerance: float = 0.03,
    audit_id: str = "",
    study_id: str = "",
    group: str = "-",
) -> MatchResult:
    """Compare a review value against a source value and label the outcome."""
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
        sd_tolerance_pct=round(sd_rel_tolerance * 100.0, 4),
    )

    # 1. Unit axis takes precedence.
    if units_differ(review_unit, source_unit):
        return MatchResult(
            **base,
            label=AuditLabel.UNIT_MISMATCH,
            reason=(
                f"unit differs (review {normalize_unit(review_unit)} "
                f"vs source {normalize_unit(source_unit)})"
            ),
        )

    # 2. Parse both sides (keeps mean + SD).
    rv = parse_numeric(review_value)
    sv = parse_numeric(source_value)
    if rv.primary is None or sv.primary is None:
        return MatchResult(
            **base,
            label=AuditLabel.NOT_COMPARABLE,
            reason="a primary numeric value could not be parsed",
        )

    # 3. Mean band.
    mean_rel = _rel(rv.primary, sv.primary)
    mean_pct = round(mean_rel * 100.0, 4)
    if mean_rel > rel_tolerance:
        return MatchResult(
            **base,
            label=AuditLabel.MISMATCH,
            rel_error_pct=mean_pct,
            reason=f"mean {rv.primary} vs {sv.primary} = {mean_pct}% > {base['tolerance_pct']}%",
        )

    # 4. SD band — only when BOTH sides report an SD.
    sd_pct: float | None = None
    if (
        rv.spread_kind == "sd"
        and sv.spread_kind == "sd"
        and rv.spread is not None
        and sv.spread is not None
    ):
        sd_rel = _rel(rv.spread, sv.spread)
        sd_pct = round(sd_rel * 100.0, 4)
        if sd_rel > sd_rel_tolerance:
            return MatchResult(
                **base,
                label=AuditLabel.MISMATCH,
                rel_error_pct=mean_pct,
                sd_rel_error_pct=sd_pct,
                reason=(
                    f"mean agrees ({mean_pct}%) but SD {rv.spread} vs {sv.spread} "
                    f"= {sd_pct}% > {base['sd_tolerance_pct']}%"
                ),
            )

    # 5. Match.
    return MatchResult(
        **base,
        label=AuditLabel.MATCH,
        rel_error_pct=mean_pct,
        sd_rel_error_pct=sd_pct,
        reason=f"mean {mean_pct}% <= {base['tolerance_pct']}%"
        + (f", SD {sd_pct}% <= {base['sd_tolerance_pct']}%" if sd_pct is not None else ""),
    )
