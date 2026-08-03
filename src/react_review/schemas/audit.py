"""Tolerance rules and the per-value audit result."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

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
    # Part of the join key, plus the cell the review value came from. A result
    # that cannot say WHICH row it judged cannot be re-checked by a human.
    timepoint: str = "single"
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
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
    # numeric | structured | semantic — how this verdict was reached.
    match_mode: str = "numeric"
    # Which structured parts were actually checked, and which were recognised in
    # the text but could not be. A MATCH with something unconsumed is a PARTIAL
    # check, and saying so is the difference between an audit and an assertion.
    components_compared: list[str] = Field(default_factory=list)
    components_unconsumed: list[str] = Field(default_factory=list)
    # A MATCH that still needs a human — the Judge skips matches otherwise.
    review_required: bool = False
    # The model's claim and which controls passed, kept so a semantic verdict can
    # be explained and disputed rather than merely accepted.
    semantic: Any = None
    semantic_relation: str = ""
    semantic_controls: dict[str, bool] = Field(default_factory=dict)
