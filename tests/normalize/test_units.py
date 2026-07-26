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
        ("cm³", "cm3"),
        ("CM3", "cm3"),
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
