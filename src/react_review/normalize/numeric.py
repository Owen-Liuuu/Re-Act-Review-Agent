"""Deterministic numeric parsing: pull the PRIMARY value out of a cell string.

The primary value is the first number in a ``mean ± SD`` / ``median (IQR)`` /
``value (range)`` cell — i.e. the central estimate the audit compares. Handles
the formats seen across systematic-review tables:

    "6.60 ± 0.71"          -> 6.60
    "52.3 (36.1-65.5) cm3" -> 52.3
    "52,3"                 -> 52.3   (European decimal comma)
    "34 (29-39)"           -> 34
    "7.0180 ± 1.85737"     -> 7.0180
    100                    -> 100.0

Ported from the reused ``steps/table_comparison/real_impl._normalise_numeric``
and generalised to return a float (or None when no number is present).
"""
from __future__ import annotations

import re

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def primary_number(value: object) -> float | None:
    """Return the first numeric token in ``value`` as a float, or None.

    A leading European decimal comma (``"52,3"``) is treated as a decimal
    point. Thousands separators are not assumed (biomedical table values are
    small), so we only convert commas that sit between digits.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None

    # European decimal comma: "52,3" -> "52.3" (only between digits).
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)

    m = _NUMBER_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None
