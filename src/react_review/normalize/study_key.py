"""The one place a citation becomes a study id.

There were two: the parser slugged a table's author cell, and the matcher pulled
a surname out of the result. Two rules for the same identity drift apart, and the
identity is what the whole audit joins on.

The rule keeps surname particles and folds accents, because dropping either makes
distinct papers collide: "van den Berg 2020" and "van Rooij 2020" both reduce to
``van_2020`` under a first-word rule, and every claim from one is then attributed
to the other. Risk grows with review size — a cardiology review has nine studies,
an oncology one has eighty.
"""
from __future__ import annotations

import re
import unicodedata

# Particles that belong to the surname rather than preceding it.
_PARTICLES = {"van", "von", "de", "del", "della", "der", "den", "di", "da", "dos",
              "du", "el", "al", "la", "le", "bin", "ibn", "mac", "mc", "st"}

_YEAR = re.compile(r"(19|20)\d{2}")
_STOPWORDS = {"et", "al", "and", "the"}


# Letters that carry no combining accent to strip, so NFKD leaves them alone.
# Author names in medical literature are routinely Turkish, Scandinavian or
# Polish; without these the surname is truncated at the first such letter
# (``Yazıcı`` → ``yaz``) and two different authors can share the result.
_LETTERS = str.maketrans({
    "ı": "i", "İ": "I", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
    "å": "a", "Å": "A", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
    "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ß": "ss", "œ": "oe", "Œ": "OE",
})


def _fold(text: str) -> str:
    """Strip accents so ``Yazıcı`` and ``Yazici`` are one name, not two."""
    decomposed = unicodedata.normalize("NFKD", (text or "").translate(_LETTERS))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def surname_of(citation: str) -> str:
    """The leading surname of a citation, particles included."""
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", _fold(citation or ""))
    parts: list[str] = []
    for word in words:
        low = word.lower()
        if low in _STOPWORDS:
            break
        parts.append(low)
        if low not in _PARTICLES:      # a particle continues into the real surname
            break
    return re.sub(r"[^a-z0-9]", "", "".join(parts))


def year_of(text: str) -> str:
    m = _YEAR.search(text or "")
    return m.group(0) if m else ""


def study_key(citation: str, *, taken: set[str] | None = None) -> str:
    """``"Ahmad et al. [2022]"`` → ``ahmad_2022``; particles and accents survive.

    ``taken`` disambiguates a genuine collision (two different papers reducing to
    the same key) with a ``_b`` / ``_c`` suffix, rather than letting the second
    silently overwrite the first.
    """
    surname, year = surname_of(citation), year_of(citation)
    base = "_".join(p for p in (surname, year) if p)
    if not base:
        base = re.sub(r"[^a-z0-9]+", "_", _fold(citation or "").lower()).strip("_")
    base = base or "study"
    if taken is None or base not in taken:
        return base
    for suffix in "bcdefghijklmnopqrstuvwxyz":
        candidate = f"{base}_{suffix}"
        if candidate not in taken:
            return candidate
    return base


def key_parts(study_id: str) -> tuple[str, str]:
    """``(surname, year)`` of an existing study id, for fuzzy matching."""
    sid = _fold(study_id or "").strip().lower()
    year = year_of(sid)
    head = re.sub(r"_(19|20)\d{2}.*$", "", sid)
    return re.sub(r"[^a-z0-9]", "", head), year
