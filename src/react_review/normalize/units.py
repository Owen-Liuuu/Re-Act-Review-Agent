"""Deterministic unit normalization + difference detection.

Canonicalises the surface form of a unit so equivalent spellings compare equal
(``kg/m²`` == ``kg/m2``) while genuinely different units stay different
(``mm`` != ``cm``, ``kg/m2`` != ``kg/m3``). Used to raise ``unit_mismatch``.
"""
from __future__ import annotations

# Domain-UNIVERSAL unit equivalences (physics/spelling, not domain knowledge):
# fold equivalent spellings of the same quantity to one canonical token so they
# compare equal. Applied PER /-separated component so compound units line up too
# (cc/m2 == cm3/m2 == ml/m2). Volume: cm³ = cc = mL. Age: years = yr = y.
_EQUIVALENTS = {
    "cm3": "ml", "cc": "ml",
    "milliliter": "ml", "millilitre": "ml",
    "milliliters": "ml", "millilitres": "ml",
    "years": "yr", "year": "yr", "yrs": "yr", "y": "yr",
}


def normalize_unit(unit: str | None) -> str:
    """Lower-case, strip spaces, fold superscripts to digits, and canonicalise
    universally-equivalent units per component (cm³/cc/mL → ml; years/yrs → yr;
    cc/m2 → ml/m2).

    Returns ``""`` for a missing / blank unit (which the compare step treats as
    "no unit asserted", so it never triggers a unit_mismatch on its own).
    """
    if not unit:
        return ""
    u = str(unit).strip().lower().replace(" ", "")
    u = u.replace("²", "2").replace("³", "3").replace("^", "").replace("·", "")
    return "/".join(_EQUIVALENTS.get(p, p) for p in u.split("/"))


def units_differ(a: str | None, b: str | None) -> bool:
    """True only when BOTH units are present and their canonical forms differ.

    A blank unit on either side is treated as "not asserted" and does not count
    as a difference — the numeric tolerance rule then decides the outcome.
    """
    na, nb = normalize_unit(a), normalize_unit(b)
    if not na or not nb:
        return False
    return na != nb


# Coarse physical KIND of a simple unit — used to catch a CONCEPT confusion (a
# length reported where a volume is expected), which is different from a mere
# scale mismatch (mm vs cm, both length). Compound units (kg/m2, mg/dl, cm3/m2)
# are deliberately left "unknown": too varied to classify safely, and callers
# treat unknown as "cannot judge" rather than a conflict.
_UNIT_KIND = {
    "mm": "length", "cm": "length", "m": "length", "um": "length",
    "µm": "length", "nm": "length",
    "ml": "volume", "l": "volume", "dl": "volume",          # cm3/cc fold to ml
    "kg": "mass", "g": "mass", "mg": "mass", "ug": "mass", "µg": "mass",
    "yr": "time", "mo": "time", "month": "time", "months": "time",
    "wk": "time", "week": "time", "weeks": "time",
    "day": "time", "days": "time", "d": "time",
    "h": "time", "hr": "time", "min": "time", "s": "time", "sec": "time",
    "%": "percent", "percent": "percent",
    "n": "count", "count": "count",
    "mmhg": "pressure", "bpm": "rate",
}


def unit_kind(unit: str | None) -> str:
    """Coarse physical kind of a unit (length / volume / mass / time / …).

    Returns ``"unknown"`` for blank, compound, or unrecognised units. Callers
    should treat ``"unknown"`` as "cannot judge" and NOT infer a conflict.
    """
    u = normalize_unit(unit)
    if not u or "/" in u:
        return "unknown"
    return _UNIT_KIND.get(u, "unknown")
