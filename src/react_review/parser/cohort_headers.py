"""Discover arm names from column-header geometry — no model, no disease words.

A review that prints arms only in headers (``N MIE`` / ``N OE``, or
``… (MIE) Events`` / ``… (OE) Events``) never fills ``cohort_label`` or
``cohort_labels_seen``. The registry then stays empty and every cell becomes
``group=all``. The contrast lives in the headers themselves: two or more
columns share a measure word, and the remainder differs. That remainder is the
arm. If the split is not unique, this module returns nothing rather than the
nearest guess.
"""
from __future__ import annotations

import re
from collections import defaultdict

# Measure words, not arm names. Matching is whole-token / suffix, never a
# substring inside a disease or drug word. Longer tokens first so
# "events" wins over a later one-letter rule.
_MEASURES = ("events", "total")
_PREFIX_MEASURES = frozenset({"n"})
_PAREN = re.compile(r"\(([^)]+)\)")
_SPACE = re.compile(r"\s+")


def _fold(text: str) -> str:
    return _SPACE.sub(" ", (text or "").strip())


def _is_measure_token(text: str) -> bool:
    token = _fold(text).lower()
    return token in _MEASURES or token in _PREFIX_MEASURES


def _split_measure(header: str) -> tuple[str, str] | None:
    """Return ``(measure, remainder)`` when ``header`` is measure + something else."""
    text = _fold(header)
    if not text:
        return None
    lower = text.lower()
    for measure in _MEASURES:
        suffix = " " + measure
        if lower.endswith(suffix):
            remainder = text[: -len(measure)].strip(" -/:,")
            return measure, remainder
        prefix = measure + " "
        if lower.startswith(prefix):
            remainder = text[len(measure):].strip(" -/:,")
            return measure, remainder
        if lower == measure:
            return measure, ""
    parts = text.split()
    if len(parts) >= 2 and parts[0].lower() in _PREFIX_MEASURES:
        return parts[0].lower(), " ".join(parts[1:]).strip(" -/:,")
    return None


def _arm_token(remainder: str) -> str | None:
    """The distinguishing fragment of a remainder, or None if it is not an arm.

    A single parenthetical abbreviation is preferred (``(MIE)`` → ``MIE``)
    because short headers (``N MIE``) and long forest paths share it. A
    remainder that is itself a measure word is refused — that is the forest
    ``Events`` / ``Total`` trap.
    """
    text = _fold(remainder).strip(" -/:,")
    if not text or _is_measure_token(text):
        return None
    parens = [p.strip() for p in _PAREN.findall(text) if p.strip()]
    if len(parens) == 1 and not _is_measure_token(parens[0]):
        return parens[0]
    return text


def arm_labels_from_headers(headers: list[str]) -> list[str]:
    """Arm labels implied by a shared-measure / differing-remainder split.

    Returns unique labels in first-seen order. Empty when no contrast is
    structural — including a table whose count columns are bare ``Events`` /
    ``Total`` with no arm remainder.
    """
    by_measure: dict[str, list[str]] = defaultdict(list)
    for header in headers:
        split = _split_measure(header)
        if split is None:
            continue
        measure, remainder = split
        token = _arm_token(remainder)
        if token is None:
            continue
        by_measure[measure].append(token)

    seen: dict[str, str] = {}
    ordered: list[str] = []
    for tokens in by_measure.values():
        unique: list[str] = []
        local: dict[str, str] = {}
        for token in tokens:
            key = token.casefold()
            if key not in local:
                local[key] = token
                unique.append(token)
        if len(unique) < 2:
            continue
        for token in unique:
            key = token.casefold()
            if key not in seen:
                seen[key] = token
                ordered.append(token)
    return ordered
