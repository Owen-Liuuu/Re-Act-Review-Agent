"""Tests for the deterministic audit verdict (compare_values + ToleranceTable)."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.audit import ToleranceTable, compare_values
from react_review.core.enums import AuditLabel

TOL = 0.01  # 1% MVP band


def _label(review, source, *, ru="", su="", tol=TOL, ft="eat_thickness"):
    return compare_values(
        field_type=ft, review_value=review, source_value=source,
        review_unit=ru, source_unit=su, rel_tolerance=tol,
    ).label


def test_exact_match():
    assert _label("6.60 ± 0.71", "6.60 ± 0.71", ru="mm", su="mm") == AuditLabel.MATCH


def test_within_tolerance_matches():
    # 6.60 vs 6.65 = 0.75% <= 1%
    assert _label("6.65", "6.60", ru="mm", su="mm") == AuditLabel.MATCH


def test_just_above_tolerance_mismatches():
    # 23.6 vs 23.31 = 1.23% > 1%
    assert _label("23.6", "23.31", ru="kg/m2", su="kg/m2", ft="bmi") == AuditLabel.MISMATCH


def test_gross_difference_mismatches():
    assert _label("85", "100", ft="sample_size") == AuditLabel.MISMATCH


def test_unit_difference_takes_precedence_over_equal_value():
    # numbers identical, but mm vs cm -> unit_mismatch (the Keles case)
    assert _label("0.7 (0.6–0.9)", "0.7 (0.6–0.9)", ru="mm", su="cm") == AuditLabel.UNIT_MISMATCH


def test_unit_mismatch_on_bmi_typo():
    # 25.8 vs 25.8 but kg/m2 vs kg/m3 (Svanteson source typo)
    assert _label("25.8 ± 3.9", "25.8 ± 3.9", ru="kg/m2", su="kg/m3", ft="bmi") == AuditLabel.UNIT_MISMATCH


def test_blank_units_do_not_trigger_unit_mismatch():
    assert _label("34 (29-39)", "34 (29-39)", ru="", su="", ft="age") == AuditLabel.MATCH


def test_unparseable_is_not_comparable():
    assert _label("not reported", "6.60", ru="mm", su="mm") == AuditLabel.NOT_COMPARABLE


def test_result_carries_rel_error_and_tolerance():
    r = compare_values(
        field_type="bmi", review_value="23.6", source_value="23.31",
        review_unit="kg/m2", source_unit="kg/m2", rel_tolerance=0.01,
    )
    assert r.label == AuditLabel.MISMATCH
    assert r.rel_error_pct == pytest.approx(1.229, abs=0.01)
    assert r.tolerance_pct == pytest.approx(1.0)


# --- ToleranceTable ---

def test_tolerance_table_default_and_override():
    t = ToleranceTable(default_rel_tolerance=0.01, per_field_type={"sample_size": 0.0})
    assert t.rel_tolerance("eat_thickness") == 0.01
    assert t.rel_tolerance("sample_size") == 0.0
    assert t.rel_tolerance("SAMPLE_SIZE") == 0.0  # case-insensitive


def test_tolerance_table_from_yaml():
    cfg = Path(__file__).resolve().parents[2] / "configs" / "tolerances.yaml"
    t = ToleranceTable.from_yaml(cfg)
    assert t.rel_tolerance("anything") == pytest.approx(0.01)
