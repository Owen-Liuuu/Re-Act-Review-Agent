"""Asking once, and what happens when the answer is unusable.

The shape of failure is the whole design here. A batch that never arrived is
worth another try; a batch with one bad line is not, because asking again buys
the same refusal at the same price. And nothing, at any point, drops to the
single-target contract: that would put half a run's answers under a profile the
artifact does not name, and would make the cost of batching unmeasurable, since
every fallback quietly adds back the calls the batch was meant to save.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from react_review.llm.base import LLMBackend
from react_review.schemas.batch import ARM, BatchQuestionId
from react_review.tools.extract_batch import (
    BAD_SHAPE,
    NOT_JSON,
    TRANSPORT,
    ExtractSourceBatchTool,
)
from react_review.tools.extraction_cache import ExtractionCache, ExtractionCacheMiss

DOCUMENT = ("A total of 945 patients underwent randomization: 316 patients were "
            "assigned to the nivolumab group, 314 to the nivolumab-plus-"
            "ipilimumab group, and 315 to the ipilimumab group.")

GOOD = {"readings": [
    {"arm_label": "nivolumab group", "value": "316", "quote": DOCUMENT,
     "population_phrase": "underwent randomization"},
    {"arm_label": "ipilimumab group", "value": "315", "quote": DOCUMENT,
     "population_phrase": "underwent randomization"}]}


class _Backend(LLMBackend):
    """Answers a scripted sequence, so a retry can be given a different reply."""

    def __init__(self, *replies) -> None:
        super().__init__()
        self._replies = list(replies)
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        reply = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def _question(**kw) -> BatchQuestionId:
    body = dict(study_id="larkin", target_shape=ARM, field_type="cohort_n",
                raw_field_name="Intervention arm, n", concept="cohort size",
                document_sha256="ABC", prompt_version="v5", prompt_sha256="P1")
    body.update(kw)
    return BatchQuestionId(**body)


def _read(tool, prompt="PROMPT"):
    return asyncio.run(tool.read(question=_question(), prompt=prompt,
                                 document=DOCUMENT))


# --- the ordinary case ------------------------------------------------------

def test_one_prompt_yields_every_reading():
    tool = ExtractSourceBatchTool(_Backend(GOOD))
    record = _read(tool)
    assert record.usable
    assert [e.value for e in record.reading.usable] == ["316", "315"]
    assert len(record.attempts) == 1 and not record.attempts[0].served_from_cache


def test_the_response_is_kept_once_not_per_claim():
    """Claims reference the record; copying it would multiply it by the group."""
    record = _read(ExtractSourceBatchTool(_Backend(GOOD)))
    assert record.model_payload == GOOD
    assert record.reading is not None


# --- retry: only what another try could fix --------------------------------

def test_a_transport_failure_is_retried_under_the_same_contract():
    tool = ExtractSourceBatchTool(_Backend(TimeoutError("gateway"), GOOD))
    record = _read(tool)
    assert record.usable
    assert [a.failure for a in record.attempts] == [TRANSPORT, ""]
    # A new attempt number, so each try is its own recording rather than two
    # answers competing for one cache slot.
    assert record.attempts[0].cache_key != record.attempts[1].cache_key


def test_a_reply_that_is_not_json_is_retried():
    tool = ExtractSourceBatchTool(_Backend("I'm afraid I can't do that", GOOD))
    record = _read(tool)
    assert record.usable
    assert record.attempts[0].failure in (NOT_JSON, TRANSPORT)


def test_json_without_the_top_level_shape_is_retried():
    tool = ExtractSourceBatchTool(_Backend({"nope": 1}, GOOD))
    record = _read(tool)
    assert record.usable
    assert record.attempts[0].failure == BAD_SHAPE


def test_retries_are_bounded_and_the_last_failure_is_the_record_s():
    tool = ExtractSourceBatchTool(_Backend({"nope": 1}), max_attempts=2)
    record = _read(tool)
    assert not record.usable
    assert record.failure == BAD_SHAPE
    assert len(record.attempts) == 2
    assert "readings" in record.detail


def test_one_bad_line_is_not_retried():
    """The parser already isolated it; asking again buys the same refusal."""
    mixed = {"readings": [GOOD["readings"][0],
                          {"arm_label": "ipilimumab group", "value": "315",
                           "quote": "a quote the paper does not contain"}]}
    backend = _Backend(mixed)
    record = _read(ExtractSourceBatchTool(backend))
    assert record.usable and backend.calls == 1
    assert len(record.reading.usable) == 1 and len(record.reading.rejected) == 1


def test_nothing_ever_falls_back_to_the_single_target_contract():
    """Not a behaviour to be tested indirectly: the tool has no such path."""
    import inspect

    from react_review.tools import extract_batch

    source = inspect.getsource(extract_batch)
    assert "targeted_v4" not in source and "legacy_v3" not in source
    assert "ExtractSourceValueTool" not in source


# --- recordings -------------------------------------------------------------

def test_a_recording_run_writes_one_entry_per_attempt(tmp_path):
    cache = ExtractionCache(tmp_path / "cache.json")
    tool = ExtractSourceBatchTool(_Backend(GOOD), cache=cache, cache_mode="record")
    record = _read(tool)
    assert record.usable
    assert cache.get(record.attempts[0].cache_key) == GOOD


def test_a_replay_run_serves_the_recording_without_a_backend(tmp_path):
    cache = ExtractionCache(tmp_path / "cache.json")
    recorder = ExtractSourceBatchTool(_Backend(GOOD), cache=cache,
                                      cache_mode="record")
    _read(recorder)
    replayer = ExtractSourceBatchTool(None, cache=cache, cache_mode="replay")
    record = _read(replayer)
    assert record.usable and record.served_from_cache


def test_a_replay_miss_stops_the_run(tmp_path):
    """An artifact that says replay has to mean it."""
    cache = ExtractionCache(tmp_path / "empty.json")
    tool = ExtractSourceBatchTool(None, cache=cache, cache_mode="replay")
    with pytest.raises(ExtractionCacheMiss):
        _read(tool)


def test_a_reworded_prompt_is_a_different_recording(tmp_path):
    cache = ExtractionCache(tmp_path / "cache.json")
    tool = ExtractSourceBatchTool(_Backend(GOOD), cache=cache, cache_mode="record")
    first = _read(tool, prompt="PROMPT")
    second = _read(tool, prompt="PROMPT ")
    assert first.attempts[0].cache_key != second.attempts[0].cache_key


def test_the_cache_key_is_about_the_words_sent_not_the_claims_consuming_them():
    """Two runs that send the same words may share a recording."""
    tool = ExtractSourceBatchTool(_Backend(GOOD))
    one = _read(tool)
    two = asyncio.run(tool.read(question=_question(study_id="other"),
                                prompt="PROMPT", document=DOCUMENT))
    assert one.attempts[0].cache_key == two.attempts[0].cache_key
    assert one.question.identity() != two.question.identity()


# --- construction refuses what it cannot do --------------------------------

def test_a_replay_tool_needs_a_cache():
    with pytest.raises(ValueError, match="requires a cache"):
        ExtractSourceBatchTool(None, cache_mode="replay")


def test_a_live_tool_needs_a_backend():
    with pytest.raises(ValueError, match="needs a backend"):
        ExtractSourceBatchTool(None, cache_mode="live")


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="live, record, or replay"):
        ExtractSourceBatchTool(_Backend(GOOD), cache_mode="whenever")
