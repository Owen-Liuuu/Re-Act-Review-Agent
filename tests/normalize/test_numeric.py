"""Tests for the deterministic numeric (primary-value) normaliser."""
from __future__ import annotations

import pytest

from react_review.normalize.numeric import primary_number


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("6.60 ± 0.71", 6.60),
        ("52.3 (36.1-65.5) cm3", 52.3),
        ("52,3", 52.3),                 # European decimal comma
        ("34 (29-39)", 34.0),
        ("7.0180 ± 1.85737", 7.0180),
        ("100", 100.0),
        (100, 100.0),
        (6.6, 6.6),
        ("0.7 (0.6–0.9)", 0.7),         # en-dash range
        ("-1.5 ± 0.2", -1.5),
    ],
)
def test_primary_number_extracts_central_value(raw, expected):
    assert primary_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, "", "   ", "N/A", "not reported", True, False])
def test_primary_number_returns_none_when_no_number(raw):
    assert primary_number(raw) is None
