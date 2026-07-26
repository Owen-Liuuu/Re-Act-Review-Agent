"""Tests for the deterministic numeric (primary-value) normaliser."""
from __future__ import annotations

import pytest

from react_review.normalize.numeric import parse_numeric, primary_number


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


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1,234", 1234.0),          # thousands separator, not a decimal
        ("12,345,678", 12345678.0), # grouped thousands
        ("0,001", 0.001),           # European decimal (leading zero, not thousands)
        ("52,3", 52.3),             # European decimal
        ("12,90", 12.90),           # European decimal, 2 places
        ("0,58933", 0.58933),       # European decimal, many places
    ],
)
def test_comma_disambiguation(raw, expected):
    assert primary_number(raw) == pytest.approx(expected)


# --- structured parse: mean + SD / range ---

def test_parse_sd_form():
    nv = parse_numeric("6.60 ± 0.71")
    assert nv.primary == pytest.approx(6.60)
    assert nv.spread == pytest.approx(0.71)
    assert nv.spread_kind == "sd"
    assert nv.lower == pytest.approx(5.89)
    assert nv.upper == pytest.approx(7.31)


def test_parse_sd_ignores_trailing_junk():
    # "27.0 ± 4.7/27.9" -> mean 27.0, sd 4.7 (the /27.9 is dropped)
    nv = parse_numeric("27.0 ± 4.7/27.9")
    assert nv.primary == pytest.approx(27.0)
    assert nv.spread == pytest.approx(4.7)
    assert nv.spread_kind == "sd"


def test_parse_range_form():
    nv = parse_numeric("34 (29-39)")
    assert nv.primary == pytest.approx(34)
    assert nv.lower == pytest.approx(29)
    assert nv.upper == pytest.approx(39)
    assert nv.spread_kind == "range"


def test_parse_iqr_form():
    nv = parse_numeric("55 (38.3-79.6), Median (IQR)")
    assert nv.primary == pytest.approx(55)
    assert nv.spread_kind == "iqr"


def test_parse_point_value_has_no_spread():
    nv = parse_numeric("100")
    assert nv.primary == pytest.approx(100)
    assert nv.spread is None
    assert nv.spread_kind == ""
