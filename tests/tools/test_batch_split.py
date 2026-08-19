"""Locate then transcribe: two calls, and a value must live in its OWN quote.

Batch transcribe shows every quote at once. Copying claim B's number onto
claim A is the SRMA failure this split exists to refuse — aslan putting age
into BMI, yazici filling four cells from one ``31±8``. Checking "the value
appears in some quote" would miss it.
"""
from __future__ import annotations

import asyncio
import json

from react_review.llm.base import LLMBackend
from react_review.schemas.batch import ARM, BatchQuestionId
from react_review.tools.batch_split import (
    own_quote_supports_value,
    parse_locate,
    merge_located_and_transcribed,
)
from react_review.tools.extract_batch import ExtractSourceBatchTool

BMI = "Arm A BMI was 22.1 kg/m2."
AGE = "Arm A age was 31 ± 8 years."
DOCUMENT = f"{BMI} {AGE}"


def test_a_value_in_its_own_quote_is_accepted():
    ok, reason = own_quote_supports_value("31 ± 8", AGE, [BMI])
    assert ok and reason == ""


def test_a_value_that_only_appears_in_a_sibling_quote_is_refused():
    """The load-bearing check. 'In some quote' would pass this and must not."""
    ok, reason = own_quote_supports_value("31 ± 8", BMI, [AGE])
    assert not ok
    assert "another reading's quote" in reason


def test_a_value_in_neither_quote_is_refused_without_accusing_a_sibling():
    ok, reason = own_quote_supports_value("99", BMI, [AGE])
    assert not ok
    assert "another reading's quote" not in reason


def test_merge_rejects_a_cross_row_copy_and_keeps_the_honest_row():
    located = [
        {"arm_label": "A BMI", "quote": BMI},
        {"arm_label": "A age", "quote": AGE},
    ]
    transcribed = {"readings": [
        {"index": 0, "value": "31 ± 8", "source_field_name": "Age"},
        {"index": 1, "value": "31 ± 8", "source_field_name": "Age"},
    ]}
    reading, names = merge_located_and_transcribed(
        located, transcribed, DOCUMENT, target_shape=ARM)
    assert any("another reading's quote" in r["reason"] for r in reading.rejected)
    assert [e.value for e in reading.usable] == ["31 ± 8"]
    assert names[0] == "Age"


def test_locate_strips_values_the_model_was_not_asked_for():
    raw = {"readings": [
        {"arm_label": "A", "quote": BMI, "value": "22.1", "unit": "kg/m2"}]}
    kept, _ = parse_locate(raw, DOCUMENT)
    assert kept and "value" not in kept[0] and "unit" not in kept[0]


class _SplitBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        if "Do NOT return a value" in prompt:
            return json.dumps({"readings": [
                {"arm_label": "A BMI", "quote": BMI},
                {"arm_label": "A age", "quote": AGE}]})
        return json.dumps({"readings": [
            {"index": 0, "value": "22.1", "unit": "kg/m2",
             "source_field_name": "BMI"},
            {"index": 1, "value": "31 ± 8", "unit": "years",
             "source_field_name": "Age"}]})


def _question() -> BatchQuestionId:
    return BatchQuestionId(
        study_id="s", target_shape=ARM, field_type="age",
        raw_field_name="Age / BMI", concept="age", document_sha256="ABC",
        prompt_version="split", prompt_sha256="P1", unit_hint="years")


def test_read_split_is_two_calls_however_many_readings_were_located():
    backend = _SplitBackend()
    tool = ExtractSourceBatchTool(backend)
    record = asyncio.run(tool.read_split(
        question=_question(),
        locate_prompt=("header\nDo NOT return a value, a unit, or "
                       "value_components.\n" + DOCUMENT),
        document=DOCUMENT))
    assert backend.calls == 2
    assert [e.value for e in record.reading.usable] == ["22.1", "31 ± 8"]
    assert record.field_names[0] == "BMI"
    assert record.field_names[1] == "Age"


def test_read_split_skips_transcribe_when_locate_finds_nothing():
    class _Empty(_SplitBackend):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.calls += 1
            return json.dumps({"readings": [],
                               "nothing_reported_reason": "not in this paper"})

    backend = _Empty()
    tool = ExtractSourceBatchTool(backend)
    record = asyncio.run(tool.read_split(
        question=_question(),
        locate_prompt="Do NOT return a value\n" + DOCUMENT,
        document=DOCUMENT))
    assert backend.calls == 1
    assert record.reading is not None
    assert not record.reading.usable


def test_read_split_uses_the_locate_backend_for_the_first_call():
    locate = _SplitBackend()
    transcribe = _SplitBackend()
    tool = ExtractSourceBatchTool(
        transcribe, locate_backend=locate, transcribe_backend=transcribe)
    asyncio.run(tool.read_split(
        question=_question(),
        locate_prompt="Do NOT return a value\n" + DOCUMENT,
        document=DOCUMENT))
    assert locate.calls == 1 and transcribe.calls == 1


def test_read_split_rejects_a_value_that_only_lives_in_a_sibling_quote():
    """Tamper: the number is real, just not in THIS claim's quote."""

    class _CrossRow(_SplitBackend):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.calls += 1
            if "Do NOT return a value" in prompt:
                return json.dumps({"readings": [
                    {"arm_label": "A BMI", "quote": BMI},
                    {"arm_label": "A age", "quote": AGE}]})
            return json.dumps({"readings": [
                {"index": 0, "value": "31 ± 8", "unit": "years",
                 "source_field_name": "Age"},
                {"index": 1, "value": "31 ± 8", "unit": "years",
                 "source_field_name": "Age"}]})

    backend = _CrossRow()
    record = asyncio.run(ExtractSourceBatchTool(backend).read_split(
        question=_question(),
        locate_prompt="Do NOT return a value\n" + DOCUMENT,
        document=DOCUMENT))
    assert backend.calls == 2
    assert any("another reading's quote" in r["reason"] for r in record.reading.rejected)
    assert [e.value for e in record.reading.usable] == ["31 ± 8"]

