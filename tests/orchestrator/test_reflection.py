"""Tests for the Reflection Decider (deterministic threshold routing)."""
from __future__ import annotations

import pytest

from react_review.core.enums import ReflectionDecision
from react_review.orchestrator.reflection import (
    ReflectionDecider,
    ReflectionSignals,
)

D = ReflectionDecision


def _decide(decider: ReflectionDecider, **kw) -> ReflectionDecision:
    return decider.decide(ReflectionSignals(**kw)).decision


@pytest.fixture
def decider() -> ReflectionDecider:
    return ReflectionDecider(accept_threshold=0.7, escalate_threshold=0.4, max_attempts=3)


def test_retrieval_failure_retries_then_escalates(decider):
    assert _decide(decider, retrieval_ok=False, attempt=0) == D.RETRY
    assert _decide(decider, retrieval_ok=False, attempt=1) == D.RETRY
    # last allowed attempt (attempt+1 == max) -> no more retries -> escalate
    assert _decide(decider, retrieval_ok=False, attempt=2) == D.ESCALATE


def test_dual_llm_disagreement_retries_then_escalates(decider):
    assert _decide(decider, agreement=False, attempt=0) == D.RETRY
    assert _decide(decider, agreement=False, attempt=2) == D.ESCALATE


def test_high_confidence_accepts(decider):
    assert _decide(decider, confidence=0.9) == D.ACCEPT
    assert _decide(decider, confidence=0.7) == D.ACCEPT  # boundary inclusive


def test_low_confidence_escalates(decider):
    assert _decide(decider, confidence=0.3) == D.ESCALATE
    assert _decide(decider, confidence=0.39) == D.ESCALATE


def test_middling_confidence_retries_then_escalates(decider):
    assert _decide(decider, confidence=0.5, attempt=0) == D.RETRY
    assert _decide(decider, confidence=0.5, attempt=2) == D.ESCALATE


def test_no_confidence_signal_accepts_when_nothing_wrong(decider):
    # MVP: confidence placeholder -> None; with retrieval ok and no disagreement,
    # there is no negative signal, so accept.
    assert _decide(decider, confidence=None, agreement=True, retrieval_ok=True) == D.ACCEPT
    assert _decide(decider, confidence=None) == D.ACCEPT


def test_retrieval_failure_takes_precedence_over_high_confidence(decider):
    # Even a high confidence cannot accept a value we failed to retrieve.
    assert _decide(decider, confidence=0.95, retrieval_ok=False, attempt=0) == D.RETRY


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        ReflectionDecider(accept_threshold=0.3, escalate_threshold=0.6)


def test_outcome_carries_reason(decider):
    out = decider.decide(ReflectionSignals(confidence=0.9))
    assert out.decision == D.ACCEPT
    assert "0.90" in out.reason and "0.70" in out.reason
