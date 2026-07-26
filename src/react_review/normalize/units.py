"""Deterministic unit normalization + difference detection.

Canonicalises the surface form of a unit so equivalent spellings compare equal
(``kg/m²`` == ``kg/m2``) while genuinely different units stay different
(``mm`` != ``cm``, ``kg/m2`` != ``kg/m3``). Used to raise ``unit_mismatch``.
"""
from __future__ import annotations


def normalize_unit(unit: str | None) -> str:
    """Lower-case, strip spaces, and fold superscripts to digits.

    Returns ``""`` for a missing / blank unit (which the compare step treats as
    "no unit asserted", so it never triggers a unit_mismatch on its own).
    """
    if not unit:
        return ""
    u = str(unit).strip().lower().replace(" ", "")
    u = u.replace("²", "2").replace("³", "3").replace("·", "")
    return u


def units_differ(a: str | None, b: str | None) -> bool:
    """True only when BOTH units are present and their canonical forms differ.

    A blank unit on either side is treated as "not asserted" and does not count
    as a difference — the numeric tolerance rule then decides the outcome.
    """
    na, nb = normalize_unit(a), normalize_unit(b)
    if not na or not nb:
        return False
    return na != nb
