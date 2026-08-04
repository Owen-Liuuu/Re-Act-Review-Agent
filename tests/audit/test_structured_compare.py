"""Structured numeric values are compared component by component.

Each case here was decided WRONGLY before: the leading number was read as the
whole value, so a bound became a measurement, an event count became a
percentage, and two hazard ratios matched while their intervals disagreed about
significance. None of them failed loudly — they produced clean verdicts.
"""
from __future__ import annotations

import pytest

from react_review.audit.compare import compare_values
from react_review.core.enums import AuditLabel
from react_review.normalize.numeric import parse_numeric


def _cmp(review, source, **kw):
    return compare_values(field_type="x", review_value=review, source_value=source, **kw)


# --- confidence intervals ---

def test_identical_intervals_match():
    r = _cmp("0.62 (95% CI 0.48-0.81)", "0.62 (95% CI 0.48-0.81)")
    # The LEVEL is compared alongside the bounds: both sides say 95%.
    assert r.label is AuditLabel.MATCH
    assert r.components_compared == ["ci", "ci_level"]


def test_different_intervals_are_a_mismatch_despite_the_same_point_estimate():
    # The point estimates are identical; only the interval differs — and it is
    # the interval that decides whether the finding is significant.
    r = _cmp("0.62 (95% CI 0.48-0.81)", "0.62 (95% CI 0.30-1.10)")
    assert r.label is AuditLabel.MISMATCH
    assert "ci" in r.components_compared


def test_intervals_that_disagree_about_the_null_are_a_mismatch_even_when_close():
    # 0.99–1.60 excludes nothing; 1.01–1.60 excludes 1.0. Endpoint arithmetic
    # alone would call these near-identical.
    r = _cmp("1.30 (95% CI 0.99-1.60)", "1.30 (95% CI 1.01-1.60)", null_value=1.0)
    assert r.label is AuditLabel.MISMATCH
    assert "disagree about 1" in r.reason


def test_an_interval_only_one_side_reports_is_unverified_not_verified():
    r = _cmp("0.62 (95% CI 0.48-0.81)", "0.62")
    assert r.label is AuditLabel.MATCH          # the point estimates do agree
    assert r.review_required is True            # …but the interval was not checked
    # Neither the bounds nor the level of a one-sided interval are verified.
    assert r.components_unconsumed == ["ci", "ci_level"]


# --- events / total / percentage ---

def test_a_count_and_a_percentage_meet_through_the_percentage():
    # Previously 45 was compared against 37.5 and called a mismatch.
    r = _cmp("45/120 (37.5%)", "37.5%")
    assert r.label is AuditLabel.MATCH
    assert r.components_compared == ["pct"]
    assert r.components_unconsumed == ["events"] and r.review_required


def test_a_genuinely_different_proportion_is_a_mismatch():
    assert _cmp("45/120 (37.5%)", "40%").label is AuditLabel.MISMATCH


def test_same_proportion_from_different_counts_is_a_mismatch():
    # 45/120 and 9/24 are both 37.5%, but they are not the same finding.
    r = _cmp("45/120 (37.5%)", "9/24 (37.5%)")
    assert r.label is AuditLabel.MISMATCH and "counts" in r.reason


def test_a_cell_that_contradicts_itself_cannot_be_audited():
    r = _cmp("45/120 (99%)", "37.5%")
    assert r.label is AuditLabel.NOT_COMPARABLE
    assert "internally inconsistent" in r.reason


# --- relational bounds ---

@pytest.mark.parametrize("review, source, label", [
    ("p < 0.001", "p < 0.001", AuditLabel.MATCH),
    ("p < 0.001", "p = 0.0009", AuditLabel.NOT_COMPARABLE),   # consistent, not exact
    ("p < 0.001", "p = 0.002", AuditLabel.MISMATCH),          # violates the bound
    ("p < 0.05", "p > 0.05", AuditLabel.MISMATCH),            # opposite directions
    ("p < 0.001", "p < 0.05", AuditLabel.MISMATCH),           # different thresholds
])
def test_a_bound_is_judged_as_a_bound(review, source, label):
    assert _cmp(review, source).label is label


def test_a_bound_met_by_an_exact_value_says_why_it_is_not_a_match():
    r = _cmp("p < 0.001", "p = 0.0009")
    assert "consistent but not exact" in r.reason


def test_p_value_threshold_uses_an_absolute_band():
    # 0.001 vs 0.002 is a 100% relative error; only an absolute band is sensible.
    assert _cmp("p < 0.001", "p < 0.002").label is AuditLabel.MISMATCH
    assert _cmp("p < 0.001", "p < 0.002",
                p_value_abs_tolerance=0.005).label is AuditLabel.MATCH


# --- nothing recognised is left silently unchecked ---

def test_every_recognised_component_is_either_compared_or_declared():
    r = _cmp("0.62 (95% CI 0.48-0.81)", "0.62")
    recognised = parse_numeric("0.62 (95% CI 0.48-0.81)").components()
    assert recognised <= set(r.components_compared) | set(r.components_unconsumed)


def test_a_partially_verified_match_is_marked_for_review():
    assert _cmp("0.62 (95% CI 0.48-0.81)", "0.62").review_required is True
    assert _cmp("6.60 ± 0.71", "6.60 ± 0.71").review_required is False


# --- the existing behaviour is untouched ---

@pytest.mark.parametrize("review, source, label", [
    ("6.60 ± 0.71", "6.60 ± 0.71", AuditLabel.MATCH),
    ("20.57 ± 1.7", "20.57 ± 1.77", AuditLabel.MISMATCH),      # SD-driven, the A003 case
    ("52.3 (36.1-65.5)", "52.3 (36.1-65.5)", AuditLabel.MATCH),
    ("100", "100", AuditLabel.MATCH),
    ("100", "148", AuditLabel.MISMATCH),
    ("Good Quality", "Good Quality", AuditLabel.NOT_COMPARABLE),
])
def test_plain_values_behave_exactly_as_before(review, source, label):
    r = _cmp(review, source)
    assert r.label is label
    assert r.match_mode == "numeric" and not r.components_unconsumed


# --- the confidence LEVEL is its own component (Phase 7C) ---

def test_the_same_bounds_at_different_levels_are_not_the_same_interval():
    """D6B-04: a 95% interval and a 99.5% interval are different quantities.

    MA012's shape — the numbers coincide exactly, and the claim still differs.
    Reading this as a match is the silent release the archive recorded.
    """
    r = compare_values(field_type="hazard_ratio",
                       review_value="0.42 (95% CI 0.31-0.57)",
                       source_value="0.42 (99.5% CI 0.31-0.57)")
    assert r.label is AuditLabel.MISMATCH
    assert "95%" in r.reason and "99.5%" in r.reason
    assert "ci_level" in r.components_compared


def test_the_level_is_compared_exactly_not_within_tolerance():
    """A level is a stated convention; 95 and 95.5 are not "close enough"."""
    r = compare_values(field_type="hazard_ratio",
                       review_value="0.42 (95% CI 0.31-0.57)",
                       source_value="0.42 (95.5% CI 0.31-0.57)")
    assert r.label is AuditLabel.MISMATCH


def test_a_level_only_one_side_states_stays_unverified():
    r = compare_values(field_type="hazard_ratio",
                       review_value="0.42 (95% CI 0.31-0.57)",
                       source_value="0.42 (0.31 to 0.57)")
    assert r.review_required is True
    assert "ci_level" in r.components_unconsumed


def test_a_bare_percentage_is_not_read_as_a_confidence_level():
    """"45/120 (37.5%)" states no interval and therefore no level."""
    assert parse_numeric("45/120 (37.5%)").ci_level is None
    assert parse_numeric("37.5%").ci_level is None
    assert parse_numeric("0.42 (95% CI 0.31-0.57)").ci_level == 95.0
    assert parse_numeric("6.9 (99.5% confidence interval 4.3 to 9.5)").ci_level == 99.5
