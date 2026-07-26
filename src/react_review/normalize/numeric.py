"""Deterministic numeric parsing.

Parses a systematic-review cell string into a structured value that keeps BOTH
the central estimate AND the spread, so the audit can compare means and SDs
separately and the report can show the range. Handles the formats seen across
review tables:

    "6.60 ± 0.71"          -> primary 6.60, spread 0.71 (sd),  [5.89, 7.31]
    "52.3 (36.1-65.5) cm3" -> primary 52.3, range [36.1, 65.5]
    "34 (29-39)"           -> primary 34,   range [29, 39]
    "55 (38.3-79.6), IQR"  -> primary 55,   iqr   [38.3, 79.6]
    "27.0 ± 4.7/27.9"      -> primary 27.0, spread 4.7 (sd)   (trailing junk ignored)
    "52,3"                 -> primary 52.3  (European decimal comma)
    "100"                  -> primary 100
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = r"-?\d+(?:\.\d+)?"
_NUMBER_RE = re.compile(_NUM)
# a parenthetical "(a - b)" range, allowing hyphen / en-dash / em-dash
_RANGE_RE = re.compile(rf"\(\s*({_NUM})\s*[-–—]\s*({_NUM})\s*\)")


@dataclass(frozen=True)
class NumericValue:
    """A parsed numeric cell: central estimate + spread + interval.

    Attributes:
        raw: the original verbatim string.
        primary: the central estimate (mean / median / point value), or None.
        spread: SD (for ``±``) or half-width ((upper-lower)/2 for a range/IQR),
            or None when the cell is a bare point value.
        spread_kind: ``"sd"`` | ``"range"`` | ``"iqr"`` | ``""``.
        lower / upper: interval bounds (± band or explicit range), or None.
    """

    raw: str
    primary: float | None
    spread: float | None = None
    spread_kind: str = ""
    lower: float | None = None
    upper: float | None = None


# A proper thousands-grouped integer: a non-zero lead group then one or more
# ",###" groups (e.g. "1,234", "12,345,678"). Excludes a leading-zero lead so
# a European decimal like "0,001" is NOT mistaken for thousands.
_THOUSANDS_RE = re.compile(r"[1-9]\d{0,2}(?:,\d{3})+")


def _fix_commas(s: str) -> str:
    """Disambiguate comma usage: thousands separator vs European decimal comma.

        "1,234"      -> "1234"     (thousands)
        "12,345,678" -> "12345678" (thousands)
        "52,3"       -> "52.3"     (European decimal)
        "0,001"      -> "0.001"    (European decimal — leading zero, not thousands)
        "12,90"      -> "12.90"    (European decimal)

    A ``d,ddd`` group with a non-zero lead is treated as thousands (and the comma
    removed); any other comma between digits is treated as a decimal point. The
    "exactly 3 digits with a non-zero lead" case (e.g. "12,345") is inherently
    ambiguous and is resolved as thousands, which matches how counts are written.
    """
    s = _THOUSANDS_RE.sub(lambda m: m.group(0).replace(",", ""), s)
    return re.sub(r"(?<=\d),(?=\d)", ".", s)


def parse_numeric(value: object) -> NumericValue:
    """Parse ``value`` into a :class:`NumericValue` (primary may be None)."""
    if value is None or isinstance(value, bool):
        return NumericValue(raw="" if value is None else str(value), primary=None)
    if isinstance(value, (int, float)):
        return NumericValue(raw=str(value), primary=float(value))

    raw = str(value)
    s = _fix_commas(raw.strip())
    if not s:
        return NumericValue(raw=raw, primary=None)

    # normalise the ± variants to a single marker
    s_pm = s.replace("+/-", "±").replace("+-", "±")

    m0 = _NUMBER_RE.search(s_pm)
    if not m0:
        return NumericValue(raw=raw, primary=None)
    primary = float(m0.group(0))

    # SD form: "<primary> ± <sd>"
    m_sd = re.search(rf"({_NUM})\s*±\s*({_NUM})", s_pm)
    if m_sd:
        spread = float(m_sd.group(2))
        return NumericValue(
            raw=raw, primary=primary, spread=spread, spread_kind="sd",
            lower=primary - spread, upper=primary + spread,
        )

    # Range / IQR form: "(a - b)"
    m_rng = _RANGE_RE.search(s)
    if m_rng:
        lower = float(m_rng.group(1))
        upper = float(m_rng.group(2))
        kind = "iqr" if re.search(r"iqr|median", s, re.I) else "range"
        spread = abs(upper - lower) / 2.0
        return NumericValue(
            raw=raw, primary=primary, spread=spread, spread_kind=kind,
            lower=lower, upper=upper,
        )

    return NumericValue(raw=raw, primary=primary)


def primary_number(value: object) -> float | None:
    """Convenience: the central estimate only (or None)."""
    return parse_numeric(value).primary
