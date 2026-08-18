"""Study identity: a table row's printed label, and how it meets a citation.

A table row is identified by the cell the review printed plus a year when the
year sits in a neighbouring column. That string is the join key. It is not a
surname slug: two papers whose first word is the same stay distinct because
their own words and years stay in the key.

Matching a row to a reference list (or to ``included_studies.csv``) is a
separate question. The candidate's own id is usually a slug from a full
citation; the row's key is the table's words. Pairing uses the year when both
sides have one, and the row's content words against the citation (or id) text.
Zero or two hits is a refusal, never a guess.

``study_key`` remains for turning a full citation into a compact alias when
there is no table cell — the reference list, collision suffixes. It is not
applied to table rows.
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


def join_key(verbatim: str, year: str = "") -> str:
    """The table's own words, with a year appended only when the cell lacks one.

    Whitespace is collapsed; spelling and order are not rewritten. A year
    already printed in the cell is left where it is, so ``Ahmad et al. [2022]``
    and ``Ahmad 2022`` stay themselves. A separate Year-column value is used
    only when the cell has no ``19xx``/``20xx``.
    """
    label = re.sub(r"\s+", " ", (verbatim or "").strip())
    y = year_of(year) or year_of(label)
    if y and not _YEAR.search(label):
        return f"{label} {y}".strip()
    return label or y or "study"


def label_tokens(text: str) -> set[str]:
    """Content words of a label, minus citation boilerplate and years."""
    return {
        w for w in re.findall(r"[a-z0-9]+", _fold(text or "").lower())
        if w not in _STOPWORDS and not _YEAR.fullmatch(w)
    }


def _collapsed(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _YEAR.sub("", _fold(text or "").lower()))


def identities_match(query: str, candidate: str) -> bool:
    """Whether a table identity and a citation/id name the same paper.

    Domain-agnostic: any review's own words, not one citation style. A year on
    both sides must agree. Initials must appear as their own token so ``J``
    cannot match ``journal``. A short slug (``yaz_2011``) may still meet a
    longer id (``yazici_2011``) by prefix on the collapsed letters.
    """
    q = re.sub(r"\s+", " ", (query or "").strip())
    c = re.sub(r"\s+", " ", (candidate or "").strip())
    if not q or not c:
        return False
    if q.casefold() == c.casefold():
        return True
    qy, cy = year_of(q), year_of(c)
    if qy and cy and qy != cy:
        return False
    qt, ct = label_tokens(q), label_tokens(c)
    if qt and ct and all(_token_covered(t, ct) for t in qt):
        return True
    qh, ch = _collapsed(q), _collapsed(c)
    return bool(qh and ch and (ch.startswith(qh) or qh.startswith(ch)))


def _token_covered(token: str, candidate_tokens: set[str]) -> bool:
    if token in candidate_tokens:
        return True
    if len(token) < 3:
        return False
    return any(ctok.startswith(token) for ctok in candidate_tokens if len(ctok) >= len(token))


def best_identity_match(
    query: str, candidates: list[tuple[str, str]],
) -> str | None:
    """The unique candidate id that matches ``query``, or None.

    Each candidate is ``(id, extra_text)``. Extra text is the printed citation
    when there is one; matching looks at id and extra together so a table
    label ``Li J et al. 2015`` can meet a slug ``li_2015`` whose citation
    still names J.
    """
    q = (query or "").strip()
    if not q or not candidates:
        return None
    hits: list[str] = []
    exact_tokens: list[str] = []
    for cid, extra in candidates:
        if not cid:
            continue
        if q.casefold() == cid.casefold() or q.casefold() == (extra or "").strip().casefold():
            return cid
        haystack = f"{cid} {extra or ''}".strip()
        if identities_match(q, haystack) or identities_match(q, cid):
            hits.append(cid)
            qt = label_tokens(q)
            if qt and (qt <= label_tokens(haystack) or qt <= label_tokens(cid)):
                exact_tokens.append(cid)
    uniq = list(dict.fromkeys(hits))
    if len(uniq) == 1:
        return uniq[0]
    # "Smith 2020" must not also claim Smithson 2020: prefer the candidate
    # whose own tokens include every query word, not merely a prefix of one.
    exact = list(dict.fromkeys(exact_tokens))
    return exact[0] if len(exact) == 1 else None


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
