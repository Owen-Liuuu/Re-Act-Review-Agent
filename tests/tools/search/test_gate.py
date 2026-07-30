"""The reference-reconciliation confidence gate (deterministic metadata scoring)."""
from __future__ import annotations

from react_review.tools.search import (
    DEFAULT_THRESHOLD,
    CandidateWork,
    ReferenceQuery,
    evaluate,
    score_match,
)

Q = ReferenceQuery(title="Epicardial fat thickness in type 1 diabetes",
                   authors=["Ahmad A", "Smith B"], year=2022, journal="Journal of Cardiology")


def _cand(**kw) -> CandidateWork:
    base = dict(doi="10.1/x", title="Epicardial fat thickness in type 1 diabetes",
                authors=["Ahmad A", "Jones C"], year=2022,
                journal="Journal of Cardiology", source="crossref")
    base.update(kw)
    return CandidateWork(**base)


def test_strong_match_scores_high_and_accepts():
    assert score_match(Q, _cand()) >= 0.85
    assert evaluate(Q, _cand()).accepted


def test_unrelated_title_scores_low_and_rejects():
    bad = _cand(doi="10.9/z", title="A randomized trial of kidney-stone lithotripsy",
                authors=["Zed Z"], year=1999, journal="Urology")
    assert score_match(Q, bad) < DEFAULT_THRESHOLD
    assert not evaluate(Q, bad).accepted


def test_no_title_anchor_scores_zero():
    # matching year but NO title on the candidate → untrustworthy → 0 (no year-only pass)
    assert score_match(Q, _cand(title="")) == 0.0


def test_title_only_query_uses_title_alone():
    q = ReferenceQuery(title="Epicardial fat thickness in type 1 diabetes")
    assert score_match(q, _cand()) >= 0.95            # exact title; nothing else asserted


def test_year_off_by_one_lowers_but_does_not_disqualify():
    assert score_match(Q, _cand(year=2022)) > score_match(Q, _cand(year=2023)) >= DEFAULT_THRESHOLD
