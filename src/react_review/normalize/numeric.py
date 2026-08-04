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

# "p < 0.001" asserts a bound. Read as the bare number 0.001 it compares equal
# to a source that reports exactly 0.001, and unequal to one reporting 0.0009 —
# both wrong, in opposite directions.
_OPERATOR_RE = re.compile(r"(?:^|[^a-z0-9])(<=|>=|≤|≥|<|>)\s*(?=[\d.])", re.I)
_OPERATORS = {"≤": "<=", "≥": ">="}

# "0.62 (95% CI 0.48–0.81)" / "0.62 [95%CI: 0.48 to 0.81]" — the interval is the
# clinically decisive half of an effect estimate and was being dropped entirely.
# The label may be abbreviated or spelled out, and a paper often prints both in
# one clause ("95% confidence interval [CI], 4.3 to 9.5"), so a "]" may stand
# between the label and the bounds; only ")" closes the clause.
_CI_RE = re.compile(
    rf"[\(\[][^)\]]*?(?:\bC\.?\s*I\.?\b|confidence\s+interval)"
    rf"[^)]*?({_NUM})\s*(?:to|[-–—,])\s*({_NUM})",
    re.I)

# The level inside that same clause: "95% CI", "99.5% CI", "95% confidence
# interval". Scanned within the matched interval only, so a cell like
# "45/120 (37.5%)" cannot contribute a confidence level.
_CI_LEVEL_RE = re.compile(
    rf"({_NUM})\s*%\s*(?:C\.?\s*I\.?|confidence\s+interval)", re.I)

# "45/120 (37.5%)" — three different quantities in one cell.
_EVENTS_RE = re.compile(r"(?<![\d./])(\d+)\s*/\s*(\d+)(?![\d/])")
_PCT_RE = re.compile(rf"({_NUM})\s*%")


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
        operator: a relational prefix (``<`` ``>`` ``<=`` ``>=``) — ``p < 0.001``
            asserts a bound, not the value 0.001, and comparing it as if it were
            that value is simply wrong.
        ci_lower / ci_upper: a reported confidence interval. Distinct from
            ``lower``/``upper``, which are a ± band or a plain range.
        ci_level: the interval's confidence LEVEL (95, 99.5). Two intervals
            stated at different levels are different quantities even when their
            bounds coincide, so the level is carried as its own component.
        events / total / pct: an ``events/total (pct%)`` cell — three different
            quantities that must be compared against their own counterparts.
    """

    raw: str
    primary: float | None
    spread: float | None = None
    spread_kind: str = ""
    lower: float | None = None
    upper: float | None = None
    operator: str = ""
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_level: float | None = None
    events: int | None = None
    total: int | None = None
    pct: float | None = None

    @property
    def derived_pct(self) -> float | None:
        """The percentage implied by ``events/total``.

        Lets ``45/120 (37.5%)`` be compared against a bare ``37.5``: the two
        cells report the same quantity in different forms, and pairing only
        identical component names would call them incomparable.
        """
        if self.events is None or not self.total:
            return None
        return 100.0 * self.events / self.total

    def self_consistent(self, rel_tolerance: float = 0.01) -> bool:
        """Whether this cell's own ``events/total`` agrees with its own ``pct``.

        A cell that contradicts itself cannot be audited against anything — the
        disagreement is in the review, before any comparison happens.
        """
        derived = self.derived_pct
        if derived is None or self.pct is None:
            return True
        return abs(derived - self.pct) <= max(abs(self.pct), 1e-9) * rel_tolerance

    def crosses(self, null: float) -> bool | None:
        """Whether the confidence interval spans ``null`` (1.0 for a ratio).

        This is the clinical reading of an interval: 0.48–0.81 excludes 1.0 and
        0.30–1.10 does not, so the same point estimate means "significant" in one
        paper and "not significant" in the other. Endpoint arithmetic alone can
        miss that when the bounds are close to the null.
        """
        if self.ci_lower is None or self.ci_upper is None:
            return None
        lo, hi = sorted((self.ci_lower, self.ci_upper))
        return lo <= null <= hi

    def components(self) -> set[str]:
        """Which structured parts this cell actually reported.

        The audit uses this to refuse a MATCH that ignored something the parse
        recognised: a confidence interval that was read and then never compared
        is worse than one that was never read, because the result looks checked.
        """
        found = set()
        if self.operator:
            found.add("operator")
        if self.ci_lower is not None or self.ci_upper is not None:
            found.add("ci")
        if self.ci_level is not None:
            found.add("ci_level")
        if self.events is not None or self.total is not None:
            found.add("events")
        if self.pct is not None:
            found.add("pct")
        if self.spread_kind == "sd" and self.spread is not None:
            found.add("sd")
        return found


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

    # Structured forms, checked before the generic ones because each carries a
    # meaning the plain-number reading would silently discard.

    # events/total, optionally with the percentage spelled out
    m_events = _EVENTS_RE.search(s_pm)
    if m_events:
        m_pct = _PCT_RE.search(s_pm)
        events, total = int(m_events.group(1)), int(m_events.group(2))
        return NumericValue(
            raw=raw, primary=float(events), events=events, total=total,
            pct=float(m_pct.group(1)) if m_pct else None)

    # a confidence interval around a point estimate
    m_ci = _CI_RE.search(s_pm)
    if m_ci:
        lo, hi = float(m_ci.group(1)), float(m_ci.group(2))
        m_level = _CI_LEVEL_RE.search(m_ci.group(0))
        return NumericValue(raw=raw, primary=primary,
                            ci_lower=min(lo, hi), ci_upper=max(lo, hi),
                            ci_level=float(m_level.group(1)) if m_level else None)

    # a relational bound ("p < 0.001"): the number is a threshold, not a value
    m_op = _OPERATOR_RE.search(s_pm)
    if m_op and s_pm.index(m_op.group(1)) < m0.start() + 1:
        op = _OPERATORS.get(m_op.group(1), m_op.group(1))
        after = _NUMBER_RE.search(s_pm[m_op.end() - 1:])
        return NumericValue(raw=raw, operator=op,
                            primary=float(after.group(0)) if after else primary)

    # a bare percentage keeps its pct component so it can meet an events/total cell
    m_only_pct = _PCT_RE.fullmatch(s_pm.strip())
    if m_only_pct:
        return NumericValue(raw=raw, primary=primary, pct=float(m_only_pct.group(1)))

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
