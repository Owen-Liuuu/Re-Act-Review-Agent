"""B1: every reflection retry has a classified reason; there is no unknown."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from react_review.tools.retry_reason import (
    RETRY_REASONS,
    classify_retry_reason,
    is_call_failed,
    retries_from_processing_records,
)

_BASELINE = Path("output/runs/contract16-d/package.json")


def test_the_four_reasons_are_the_closed_set():
    assert RETRY_REASONS == (
        "call_failed", "not_found", "low_confidence", "disagreement")


def test_call_failed_is_an_exhausted_transport_not_a_paper_miss():
    truncated = SimpleNamespace(
        found=False, error="truncated: reasoning used the whole budget",
        not_found_reason="the extraction call failed: truncated: ...",
        permanent_failure=False, agreement=None, confidence=None)
    network = {
        "found": False,
        "error": "OpenAI API error after 1 attempts: network error: ReadError('')",
        "not_found_reason": "the extraction call failed: OpenAI API error after 1 attempts",
        "permanent_failure": False,
    }
    assert is_call_failed(truncated) and classify_retry_reason(truncated) == "call_failed"
    assert classify_retry_reason(network) == "call_failed"


def test_not_found_is_an_answered_miss():
    result = {
        "found": False, "error": "", "not_found_reason": "the paper does not report age",
        "permanent_failure": False, "target_check": "ok",
    }
    assert not is_call_failed(result)
    assert classify_retry_reason(result) == "not_found"


def test_low_confidence_and_disagreement_are_classified():
    low = SimpleNamespace(
        found=True, error="", not_found_reason="", permanent_failure=False,
        agreement=None, confidence=0.5)
    clash = SimpleNamespace(
        found=True, error="", not_found_reason="", permanent_failure=False,
        agreement=False, confidence=0.9)
    assert classify_retry_reason(low) == "low_confidence"
    assert classify_retry_reason(clash) == "disagreement"


def test_a_found_retry_without_a_signal_is_not_unknown_it_is_a_bug():
    with pytest.raises(ValueError, match="no disagreement or low-confidence"):
        classify_retry_reason(SimpleNamespace(
            found=True, error="", not_found_reason="", permanent_failure=False,
            agreement=None, confidence=None))


def test_a_retry_without_a_previous_result_is_not_unknown():
    with pytest.raises(ValueError, match="no previous extract result"):
        classify_retry_reason(None)


@pytest.mark.skipif(not _BASELINE.is_file(), reason="contract16-d package absent")
def test_contract16_d_retries_sum_to_repeated_attempts_with_no_unknown():
    package = json.loads(_BASELINE.read_text(encoding="utf-8"))
    expected = int((package.get("telemetry") or {}).get("repeated_attempts") or 0)
    counts = retries_from_processing_records(package.get("processing_records") or [])
    assert expected == 65
    assert sum(counts.values()) == expected
    assert set(counts) == set(RETRY_REASONS)
    assert counts["call_failed"] == 37
    assert counts["not_found"] == 28
    assert counts["low_confidence"] == 0
    assert counts["disagreement"] == 0
