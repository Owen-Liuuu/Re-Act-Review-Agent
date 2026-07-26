"""Tolerance rules and the per-value audit result."""
from __future__ import annotations

from pydantic import BaseModel

from react_review.core.enums import AuditLabel

Value = str | int | float | None


class ToleranceRule(BaseModel):
    """Comparison tolerance for a field_type.

    Attributes:
        field_type: The concept this rule applies to, or ``"*"`` for the
            default rule.
        rel_tolerance: Relative-error upper bound for MATCH on numeric fields
            (e.g. 0.01 = 1%). ``abs(a-b)/max(|a|,|b|,eps) <= rel_tolerance``.
        comparison_family: ``"numeric"`` (relative-error) or ``"text"``
            (reserved for future similarity-based fields).
    """

    field_type: str = "*"
    rel_tolerance: float = 0.01
    comparison_family: str = "numeric"


class MatchResult(BaseModel):
    """Audit outcome for one (study, group, field_type) review↔source pair.

    Attributes:
        label: match / mismatch / unit_mismatch / not_comparable.
        rel_error_pct: relative error of the MEAN / primary values, in percent
            (None when a value could not be parsed or units differ).
        sd_rel_error_pct: relative error of the SD, in percent, when BOTH sides
            report a ``mean ± sd`` spread (None otherwise).
        tolerance_pct / sd_tolerance_pct: the bands applied, in percent.
        reason: short human-readable explanation.
    """

    audit_id: str = ""
    study_id: str = ""
    group: str = "-"
    field_type: str = ""
    review_value: Value = None
    review_unit: str = ""
    source_value: Value = None
    source_unit: str = ""
    label: AuditLabel = AuditLabel.NOT_COMPARABLE
    rel_error_pct: float | None = None
    sd_rel_error_pct: float | None = None
    tolerance_pct: float | None = None
    sd_tolerance_pct: float | None = None
    reason: str = ""
