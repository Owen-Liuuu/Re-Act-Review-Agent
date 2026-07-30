"""Deterministic confidence gate for reference reconciliation.

A fuzzy citation → work match is a HYPOTHESIS: auditing against the WRONG paper
(a similar title) is worse than not finding it at all. So a candidate is accepted
only when its metadata agreement — title (the anchor) + author + year + journal —
clears a threshold. This mirrors the DKB verify-gate philosophy: the online
service proposes, deterministic code judges the guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from react_review.tools.search.models import CandidateWork, ReferenceQuery

# Title dominates; author/year/journal corroborate. Only components present on
# BOTH sides count, and the score is renormalized by their weight.
_WEIGHTS = {"title": 0.6, "author": 0.2, "year": 0.1, "journal": 0.1}
DEFAULT_THRESHOLD = 0.72


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).strip()


def _title_sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _surnames(authors: list[str]) -> set[str]:
    out: set[str] = set()
    for a in authors:
        m = re.search(r"[A-Za-z][A-Za-z\-']+", a or "")
        if m:
            out.add(m.group(0).lower())
    return out


def _author_overlap(q: list[str], c: list[str]) -> float:
    qs, cs = _surnames(q), _surnames(c)
    if not qs or not cs:
        return 0.0
    return len(qs & cs) / len(qs)


def _year_score(qy: int | None, cy: int | None) -> float:
    if qy is None or cy is None:
        return 0.0
    if qy == cy:
        return 1.0
    return 0.5 if abs(qy - cy) == 1 else 0.0


def _journal_score(q: str, c: str) -> float:
    nq, nc = _norm(q), _norm(c)
    if not nq or not nc:
        return 0.0
    if nq in nc or nc in nq:
        return 1.0
    return SequenceMatcher(None, nq, nc).ratio()


def score_match(query: ReferenceQuery, cand: CandidateWork) -> float:
    """0..1 metadata-agreement score between a citation query and a candidate.

    Title is the anchor: with no title on both sides the match is untrustworthy
    and scores 0, so a year- or journal-only coincidence never passes the gate.
    """
    q_title = query.title or query.citation
    if not (q_title and cand.title):
        return 0.0
    comps = [
        (_WEIGHTS["title"], _title_sim(q_title, cand.title), True),
        (_WEIGHTS["author"], _author_overlap(query.authors, cand.authors),
         bool(query.authors and cand.authors)),
        (_WEIGHTS["year"], _year_score(query.year, cand.year),
         query.year is not None and cand.year is not None),
        (_WEIGHTS["journal"], _journal_score(query.journal, cand.journal),
         bool(query.journal and cand.journal)),
    ]
    num = sum(w * v for w, v, ok in comps if ok)
    den = sum(w for w, v, ok in comps if ok)
    return num / den if den else 0.0


@dataclass(frozen=True)
class ReferenceMatch:
    candidate: CandidateWork
    confidence: float
    accepted: bool


def evaluate(
    query: ReferenceQuery, candidate: CandidateWork, *, threshold: float = DEFAULT_THRESHOLD,
) -> ReferenceMatch:
    s = score_match(query, candidate)
    return ReferenceMatch(candidate=candidate, confidence=s, accepted=s >= threshold)
