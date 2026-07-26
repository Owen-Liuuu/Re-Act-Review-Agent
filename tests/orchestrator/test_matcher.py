"""Tests for the review↔source match-table join."""
from __future__ import annotations

from react_review.orchestrator.matcher import build_pairs, match_key
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem


def _rv(study, group, ft, value="1"):
    return ReviewDataItem(study_id=study, group=group, field_type=ft, value=value)


def _sv(study, group, ft, value="1"):
    return SourceEvidenceItem(study_id=study, group=group, field_type=ft, source_value=value)


def test_key_is_normalized():
    assert match_key("Ahmad_2022", " T1DM ", "Single", "EAT_thickness") == (
        "ahmad_2022", "t1dm", "single", "eat_thickness"
    )


def test_exact_pairing():
    review = [_rv("s1", "t1dm", "bmi"), _rv("s1", "control", "bmi")]
    source = [_sv("s1", "control", "bmi"), _sv("s1", "t1dm", "bmi")]
    pairs, ur, us = build_pairs(review, source)
    assert len(pairs) == 2 and not ur and not us
    # each review paired with the SAME group's source
    for r, s in pairs:
        assert r.group == s.group


def test_unmatched_review_and_source():
    review = [_rv("s1", "t1dm", "bmi"), _rv("s1", "t1dm", "age")]
    source = [_sv("s1", "t1dm", "bmi"), _sv("s1", "t1dm", "eat_thickness")]
    pairs, ur, us = build_pairs(review, source)
    assert len(pairs) == 1
    assert [r.field_type for r in ur] == ["age"]
    assert [s.field_type for s in us] == ["eat_thickness"]


def test_source_consumed_at_most_once():
    review = [_rv("s1", "t1dm", "bmi"), _rv("s1", "t1dm", "bmi")]  # duplicate claim
    source = [_sv("s1", "t1dm", "bmi")]
    pairs, ur, us = build_pairs(review, source)
    assert len(pairs) == 1
    assert len(ur) == 1  # the second review claim finds no free source
    assert not us
