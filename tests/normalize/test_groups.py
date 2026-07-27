"""Tests for deterministic group normalization."""
from __future__ import annotations

import pytest

from react_review.normalize.groups import normalize_group


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("T1DM", "t1dm"),
        ("Type 1 diabetes", "t1dm"),
        ("DM", "t1dm"),
        ("Diabetic children", "t1dm"),
        ("Patient", "t1dm"),
        ("Control", "control"),
        ("Controls", "control"),
        ("healthy controls", "control"),
        ("non-diabetic control", "control"),   # control wins over the 'diabet' token
        ("", "all"),
        ("-", "all"),
        ("combined", "all"),
        ("whatever", "all"),
    ],
)
def test_normalize_group(raw, expected):
    assert normalize_group(raw) == expected
