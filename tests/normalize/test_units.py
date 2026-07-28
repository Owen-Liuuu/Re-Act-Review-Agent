"""Tests for deterministic unit normalization + difference detection."""
from __future__ import annotations

import pytest

from react_review.normalize.units import normalize_unit, units_differ


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("kg/m²", "kg/m2"),
        ("kg/m2", "kg/m2"),
        ("kg/m3", "kg/m3"),
        ("mm", "mm"),
        (" cm ", "cm"),
        ("cm³", "ml"),       # volume equivalence: cm³ = mL
        ("CM3", "ml"),
        ("cm^3", "ml"),      # caret form folds too
        ("cc", "ml"),
        ("mL", "ml"),
        ("yrs", "yr"),       # age spellings: years = yr = y
        ("years", "yr"),
        ("year", "yr"),
        ("cc/m2", "ml/m2"),  # per-component: cc/m2 = cm3/m2 = ml/m2
        ("cm3/m2", "ml/m2"),
        ("kg/m²", "kg/m2"),  # non-equivalent components pass through unchanged
        (None, ""),
        ("", ""),
    ],
)
def test_normalize_unit(raw, expected):
    assert normalize_unit(raw) == expected


def test_units_differ_true_cases():
    assert units_differ("mm", "cm") is True
    assert units_differ("kg/m2", "kg/m3") is True


def test_units_differ_false_when_equivalent_or_blank():
    assert units_differ("kg/m²", "kg/m2") is False   # equivalent spelling
    assert units_differ("mm", "mm") is False
    assert units_differ("", "mm") is False           # blank = not asserted
    assert units_differ("mm", None) is False
    assert units_differ("cm3", "ml") is False        # volume equivalence
    assert units_differ("cc", "mL") is False
    assert units_differ("yrs", "years") is False     # age spelling
    assert units_differ("cc/m2", "cm3/m2") is False  # compound volume/BSA
