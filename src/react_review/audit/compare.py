"""``compare_values`` — the deterministic audit verdict for one value pair.

Rule (Tier-3), in order:
  0. Either value missing / explicitly unavailable -> NOT_COMPARABLE
  1. Units present and different                   -> UNIT_MISMATCH
  2. Either primary value unparseable              -> NOT_COMPARABLE
  3. MEAN relative error > mean tolerance           -> MISMATCH
  4. Both sides have an SD and the SD relative
     error > SD tolerance                         -> MISMATCH (SD-driven)
  5. otherwise                                      -> MATCH

Dual band: the mean (primary value) uses the tight band (MVP 1%); the SD uses a
separate, looser band (MVP 3%) and is only checked when BOTH sides report a
``mean ± sd`` spread — so it catches an SD transcription error (e.g. 1.7 vs
1.77) without false-flagging a value that only reports a range/IQR or a point.
Unit handling is a separate axis and takes precedence.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from react_review.core.enums import AuditLabel
from react_review.normalize import normalize_unit, parse_numeric, units_differ
from react_review.normalize.numeric import NumericValue
from react_review.schemas.audit import MatchResult

_EPS = 1e-9
# Interval bounds are estimated, not measured, so they carry more rounding noise
# than a point estimate; a band as tight as the mean's would flag transcription
# rounding as a discrepancy.
_CI_TOLERANCE = 0.05

Value = str | int | float | None

# A stated absence is not a value.  This list is deliberately shared with the
# semantic escalation gate: neither deterministic unit comparison nor an LLM
# may turn an unavailable source value into evidence about that value.
_MISSING_VALUES = {
    "", "-", "—", "na", "n/a", "nr", "nd", "ne", "none", "null",
    "not reported", "not applicable", "not available", "not stated",
    "not reached", "unknown",
}


def is_missing_value(value: object) -> bool:
    """Return True when *value* explicitly carries no auditable value."""
    return value is None or (
        isinstance(value, str) and value.strip().lower() in _MISSING_VALUES
    )


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), _EPS)


def _within(a: float, b: float, rel_tolerance: float) -> bool:
    return _rel(a, b) <= rel_tolerance


def _compare_components(
    rv: NumericValue, sv: NumericValue, base: dict, *,
    rel_tolerance: float, p_value_abs_tolerance: float, null_value: float | None,
) -> MatchResult | None:
    """Decide a pair that reports structure, or return None to fall through.

    Each component is judged against its OWN counterpart, never pooled into one
    list of numbers: in ``45/120 (37.5%)`` the three numbers mean different
    things. Anything the parse recognised but this function cannot compare is
    recorded in ``components_unconsumed`` and forces human review, so a partial
    check never presents itself as a complete one.
    """
    mine, theirs = rv.components(), sv.components()
    structural = (mine | theirs) - {"sd"}            # sd is handled by the mean band
    if not structural:
        return None

    def result(label, reason, *, compared, unconsumed=(), review=False, **extra):
        return MatchResult(**base, label=label, reason=reason,
                           components_compared=sorted(compared),
                           components_unconsumed=sorted(unconsumed),
                           review_required=review, match_mode="structured", **extra)

    # A cell that contradicts itself cannot be audited against anything.
    for side, value in (("review", rv), ("source", sv)):
        if not value.self_consistent(rel_tolerance):
            return result(
                AuditLabel.NOT_COMPARABLE,
                f"the {side} value is internally inconsistent: {value.events}/"
                f"{value.total} is {value.derived_pct:.4g}%, not {value.pct}%",
                compared=["events", "pct"])

    compared: list[str] = []
    unconsumed: list[str] = []

    # --- a relational bound ("p < 0.001") ---
    if "operator" in structural:
        verdict = _compare_operator(rv, sv, p_value_abs_tolerance, rel_tolerance)
        if verdict is not None:
            label, reason = verdict
            return result(label, reason, compared=["operator"])
        compared.append("operator")

    # --- proportions: events/total and percentages meet through the percentage ---
    if structural & {"events", "pct"}:
        r_pct = rv.pct if rv.pct is not None else rv.derived_pct
        s_pct = sv.pct if sv.pct is not None else sv.derived_pct
        if r_pct is not None and s_pct is not None:
            if not _within(r_pct, s_pct, rel_tolerance):
                return result(AuditLabel.MISMATCH,
                              f"proportion {r_pct:.4g}% vs {s_pct:.4g}%",
                              compared=["pct"], rel_error_pct=round(_rel(r_pct, s_pct) * 100, 4))
            compared.append("pct")
            if rv.events is not None and sv.events is not None:
                if rv.events != sv.events or rv.total != sv.total:
                    return result(AuditLabel.MISMATCH,
                                  f"counts {rv.events}/{rv.total} vs "
                                  f"{sv.events}/{sv.total}", compared=["events", "pct"])
                compared.append("events")
            elif "events" in structural:
                unconsumed.append("events")   # only one side reported the counts
        else:
            unconsumed.extend(structural & {"events", "pct"})

    # --- confidence LEVEL ---
    # Checked before the bounds, because it decides what the bounds mean. A 95%
    # interval and a 99.5% interval are different quantities: the same numbers
    # at different levels do not agree, and different numbers at different
    # levels are not evidence of a transcription error. Compared exactly — a
    # level is a stated convention, not a measurement with a tolerance.
    if "ci_level" in structural:
        if rv.ci_level is not None and sv.ci_level is not None:
            if rv.ci_level != sv.ci_level:
                return result(
                    AuditLabel.MISMATCH,
                    f"the review reports a {rv.ci_level:g}% confidence interval "
                    f"and the source a {sv.ci_level:g}% one, so their bounds are "
                    "not the same quantity",
                    compared=[*compared, "ci_level"])
            compared.append("ci_level")
        else:
            # One side states the level and the other does not. The bounds may
            # still be compared, but the pair cannot be called fully verified.
            unconsumed.append("ci_level")

    # --- confidence interval ---
    if "ci" in structural:
        if rv.ci_lower is not None and sv.ci_lower is not None:
            if null_value is not None and rv.crosses(null_value) != sv.crosses(null_value):
                return result(
                    AuditLabel.MISMATCH,
                    f"the intervals disagree about {null_value:g}: review "
                    f"[{rv.ci_lower:g}, {rv.ci_upper:g}] vs source "
                    f"[{sv.ci_lower:g}, {sv.ci_upper:g}] — one excludes it, one does not",
                    compared=["ci"])
            for name, a, b in (("lower", rv.ci_lower, sv.ci_lower),
                               ("upper", rv.ci_upper, sv.ci_upper)):
                if not _within(a, b, _CI_TOLERANCE):
                    return result(
                        AuditLabel.MISMATCH,
                        f"confidence interval {name} bound {a:g} vs {b:g}",
                        compared=["ci"], rel_error_pct=round(_rel(a, b) * 100, 4))
            compared.append("ci")
        else:
            # An interval one side reported and the other did not cannot be
            # verified; the point estimate alone must not read as fully checked.
            unconsumed.append("ci")

    if unconsumed:
        # Fall back to the point estimates ONLY if nothing else was comparable.
        # Where a component did agree — a percentage, say — the leading numbers
        # may not even denote the same quantity: in "45/120 (37.5%)" against
        # "37.5%" they are an event count and a percentage.
        if compared:
            return result(
                AuditLabel.MATCH,
                f"{', '.join(compared)} agree, but only one side reports "
                f"{', '.join(unconsumed)} — that part is unverified",
                compared=compared, unconsumed=unconsumed, review=True)
        mean_ok = _within(rv.primary, sv.primary, rel_tolerance)
        return result(
            AuditLabel.MATCH if mean_ok else AuditLabel.MISMATCH,
            ("the point estimates agree, but " if mean_ok else "point estimates differ; ")
            + f"only one side reports {', '.join(unconsumed)} — not verified",
            compared=compared, unconsumed=unconsumed, review=True,
            rel_error_pct=round(_rel(rv.primary, sv.primary) * 100, 4))

    if compared:
        return result(AuditLabel.MATCH,
                      f"{', '.join(compared)} agree within tolerance",
                      compared=compared)
    return None


def _with_components(numeric, components):
    """Fill in parts the verbatim string does not carry — never overwrite one.

    Only gaps are filled. If the printed value and the reported components
    disagree, the printed value wins here: the extraction contract has already
    refused that response, and this path must not become a second, quieter place
    where a component can change a number.
    """
    if components is None:
        return numeric
    get = (components.get if isinstance(components, dict)
           else lambda key: getattr(components, key, None))
    if str(get("status") or "ok") == "protocol_error":
        return numeric
    updates = {name: get(name) for name in ("ci_level", "ci_lower", "ci_upper")
               if hasattr(numeric, name) and get(name) is not None
               and getattr(numeric, name) is None}
    if not updates:
        return numeric
    return replace(numeric, **updates)


def _compare_operator(
    rv: NumericValue, sv: NumericValue, abs_tolerance: float, rel_tolerance: float,
) -> tuple[AuditLabel, str] | None:
    """Judge a relational bound; None means "agreed, carry on"."""
    if rv.operator and sv.operator:
        if rv.operator != sv.operator:
            return AuditLabel.MISMATCH, (
                f"different bounds: {rv.operator}{rv.primary:g} vs "
                f"{sv.operator}{sv.primary:g}")
        if abs(rv.primary - sv.primary) <= abs_tolerance or _within(
                rv.primary, sv.primary, rel_tolerance):
            return None
        return AuditLabel.MISMATCH, (
            f"same bound, different threshold: {rv.operator}{rv.primary:g} vs "
            f"{sv.operator}{sv.primary:g}")

    # One side states a bound, the other an exact value.
    bound, exact = (rv, sv) if rv.operator else (sv, rv)
    satisfies = (exact.primary < bound.primary if bound.operator == "<" else
                 exact.primary <= bound.primary if bound.operator == "<=" else
                 exact.primary > bound.primary if bound.operator == ">" else
                 exact.primary >= bound.primary)
    if satisfies:
        # "p < 0.001" and "p = 0.0009" do not contradict each other, but they are
        # not the same statement either — one is a bound, the other a measurement.
        return AuditLabel.NOT_COMPARABLE, (
            f"consistent but not exact: {bound.operator}{bound.primary:g} reports a "
            f"bound, {exact.primary:g} an exact value")
    return AuditLabel.MISMATCH, (
        f"{exact.primary:g} violates the reported bound "
        f"{bound.operator}{bound.primary:g}")


def compare_values(
    *,
    field_type: str,
    review_value: Value,
    source_value: Value,
    review_unit: str = "",
    source_unit: str = "",
    rel_tolerance: float = 0.01,
    sd_rel_tolerance: float = 0.03,
    p_value_abs_tolerance: float = 0.0,
    null_value: float | None = None,
    audit_id: str = "",
    study_id: str = "",
    group: str = "-",
    source_components: Any = None,
) -> MatchResult:
    """Compare a review value against a source value and label the outcome.

    ``source_components`` are the parts of the source value as the extraction
    reported and verified them. They matter because a verbatim string can be
    less than what was read: a response that returns ``0.42`` from a sentence
    printing ``0.42; 99.5% CI, 0.31 to 0.57`` used to reach here as a bare point
    estimate, indistinguishable from a paper that states no interval — so the
    review's own interval went unchecked. The verbatim value is still what gets
    displayed; the components only supply parts it does not carry.
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
        sd_tolerance_pct=round(sd_rel_tolerance * 100.0, 4),
    )

    # 0. Existence takes precedence over every assertion about a value.  In
    # particular, a residual unit returned alongside ``source_value=None`` is
    # extraction metadata, not evidence that the review used the wrong unit.
    for side, value in (("review", review_value), ("source", source_value)):
        if is_missing_value(value):
            return MatchResult(
                **base,
                label=AuditLabel.NOT_COMPARABLE,
                reason=f"the {side} value is missing; units were not compared",
            )

    # 1. Unit axis takes precedence once both sides contain real values.
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
    sv = _with_components(parse_numeric(source_value), source_components)
    if rv.primary is None or sv.primary is None:
        return MatchResult(
            **base,
            label=AuditLabel.NOT_COMPARABLE,
            reason="a primary numeric value could not be parsed",
        )

    # 2b. Structured components (a bound, an interval, events/total). These are
    # decided on their own terms BEFORE the mean band, because comparing their
    # leading number as if it were a plain value is what produced wrong verdicts:
    # two hazard ratios of 0.62 matched while one interval crossed 1.0 and the
    # other did not.
    structured = _compare_components(
        rv, sv, base, rel_tolerance=rel_tolerance,
        p_value_abs_tolerance=p_value_abs_tolerance, null_value=null_value)
    if structured is not None:
        return structured

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
