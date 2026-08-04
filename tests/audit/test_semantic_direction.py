"""The semantic verdict must agree with itself about direction.

Phase 6B recorded a verdict whose rationale said the review was the MORE
specific side while its label said ``review_broader`` — which the prompt defines
as the review being LESS specific. Nothing in the pipeline could notice, because
only the label was ever read.

These tests pin the contract. Wiring it into the controls is Phase 7D-2; here it
is a pure function, so the table can be checked without a model.
"""
from __future__ import annotations

import pytest

from react_review.audit.semantic_control import (
    direction_consistent,
    direction_stated,
)
from react_review.schemas.semantic import SemanticVerdict


def _verdict(**kw) -> SemanticVerdict:
    body = {"relation": "same", "more_specific_side": "neither",
            "equivalent": True, "confidence": 0.9}
    body.update(kw)
    return SemanticVerdict(**body)


@pytest.mark.parametrize("relation,side", [
    ("same", "neither"),
    ("review_broader", "source"),
    ("source_broader", "review"),
    ("different", "unknown"),
    ("different", "neither"),
    ("unknown", "unknown"),
])
def test_consistent_combinations_pass(relation, side):
    ok, reason = direction_consistent(_verdict(
        relation=relation, more_specific_side=side,
        equivalent=(relation != "different")))
    assert ok is True and reason == ""


@pytest.mark.parametrize("relation,side", [
    ("review_broader", "review"),     # the Phase 6B MA003 shape
    ("source_broader", "source"),
    ("same", "review"),
    ("same", "source"),
    ("unknown", "review"),
    ("different", "source"),
])
def test_contradictory_direction_is_caught(relation, side):
    ok, reason = direction_consistent(_verdict(
        relation=relation, more_specific_side=side,
        equivalent=(relation != "different")))
    assert ok is False
    assert relation in reason and side in reason


def test_ma003_recorded_verdict_is_rejected():
    """The real Phase 6B response: the rationale and the label disagree."""
    ok, reason = direction_consistent(SemanticVerdict(
        relation="review_broader", more_specific_side="review",
        equivalent=False, confidence=1.0,
        rationale=("The review specifies exact drug doses (1 mg/kg and 3 mg/kg), "
                   "while the source only states the combination of drugs "
                   "without any dosage information.")))
    assert ok is False
    assert "more specific side" in reason


def test_same_must_be_equivalent():
    ok, reason = direction_consistent(_verdict(relation="same", equivalent=False))
    assert ok is False and "not equivalent" in reason


def test_different_must_not_be_equivalent():
    ok, reason = direction_consistent(_verdict(
        relation="different", more_specific_side="unknown", equivalent=True))
    assert ok is False and "differ but are equivalent" in reason


def test_an_undefined_relation_is_rejected():
    ok, reason = direction_consistent(_verdict(relation="narrower"))
    assert ok is False and "not one this audit defines" in reason


def test_default_verdict_is_self_consistent():
    """An unparsed/empty verdict must not look like a contradiction."""
    ok, _ = direction_consistent(SemanticVerdict())
    assert ok is True


def test_a_verdict_recorded_before_this_contract_is_not_convicted():
    """The Phase 6B recordings state no side; that is unasked, not wrong."""
    legacy = SemanticVerdict(relation="review_broader", equivalent=False,
                             confidence=1.0, rationale="recorded in Phase 6B")
    assert direction_stated(legacy) is False
    assert direction_consistent(legacy) == (True, "")
    # …but its equivalence claims are still checkable with what it did record.
    ok, reason = direction_consistent(
        SemanticVerdict(relation="same", equivalent=False))
    assert ok is False and "not equivalent" in reason


def test_an_explicit_unknown_side_is_still_checked():
    """Asked and answered "unknown" is an answer, and must fit the relation."""
    verdict = SemanticVerdict(relation="review_broader",
                              more_specific_side="unknown", equivalent=False)
    assert direction_stated(verdict) is True
    ok, _ = direction_consistent(verdict)
    assert ok is False
