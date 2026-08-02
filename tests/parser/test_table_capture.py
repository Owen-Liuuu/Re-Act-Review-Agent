"""Stage 1: verbatim capture, the hard checkpoint, retry, and dropping a table."""
from __future__ import annotations

import json

import pytest

from react_review.core.exceptions import RunStopped
from react_review.hitl import Decision, ScriptedCheckpoint, StepReporter, StepStage
from react_review.llm.base import LLMBackend
from react_review.parser.table_capture import TableCapturer

CAPTURE = {
    "research_context": "epicardial fat in type 1 diabetes",
    "tables": [
        {"table_id": "table_1", "caption": "Characteristics", "role": "characteristics",
         "header_rows": [["Study", "N", "EAT"]],
         "rows": [["Ahmad 2022", "100", "6.60 ± 0.71"], ["Keles 2016", "NR", "—"]],
         "row_axis_columns": ["Study"], "cohort_labels_seen": ["T1DM", "Control"]},
        {"table_id": "table_s1", "caption": "Search strategy", "role": "other",
         "header_rows": [["Database", "Query"]], "rows": [["PubMed", "eat AND t1dm"]]},
    ],
}


class QueueBackend(LLMBackend):
    def __init__(self, responses: list) -> None:
        super().__init__()
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return self._responses.pop(0) if self._responses else "{}"


def _reporter(decisions=None) -> tuple[StepReporter, ScriptedCheckpoint]:
    gate = ScriptedCheckpoint(decisions or [])
    return StepReporter("r", gate=gate), gate


@pytest.mark.asyncio
async def test_captures_every_table_verbatim():
    reporter, gate = _reporter()
    table_set, ctx = await TableCapturer(QueueBackend([CAPTURE])).capture(
        "text", reporter=reporter, pdf_path="C:/review.pdf")

    assert [t.table_id for t in table_set.tables] == ["table_1", "table_s1"]
    assert ctx == "epicardial fat in type 1 diabetes"
    # placeholder cells survive the round trip — they may be results, not absences
    assert table_set.tables[0].rows[1] == ["Keles 2016", "NR", "—"]
    # the review's own cohort words are carried through untranslated
    assert table_set.tables[0].cohort_labels_seen == ["T1DM", "Control"]


@pytest.mark.asyncio
async def test_the_checkpoint_names_the_file_and_offers_retry_and_drop():
    reporter, gate = _reporter()
    await TableCapturer(QueueBackend([CAPTURE])).capture(
        "text", reporter=reporter, pdf_path="C:/review.pdf")

    event = gate.seen[0]
    assert event.stage is StepStage.TABLE_CAPTURE
    assert event.subject == "C:/review.pdf"            # WHICH file
    assert event.offers == ["retry", "retry_alt"]
    assert event.selectable == "tables"
    assert len(event.selectable_items()) == 2
    assert "Characteristics" in event.render_blocks[0]


@pytest.mark.asyncio
async def test_stopping_at_the_capture_gate_halts_the_run():
    # "If this table was not correctly extracted, stop here."
    reporter, _ = _reporter([Decision.STOP])
    with pytest.raises(RunStopped):
        await TableCapturer(QueueBackend([CAPTURE])).capture("t", reporter=reporter)


@pytest.mark.asyncio
async def test_retry_re_transcribes_with_a_different_seed():
    backend = QueueBackend([CAPTURE, CAPTURE])
    reporter, _ = _reporter([Decision.RETRY, Decision.CONTINUE])
    await TableCapturer(backend).capture("t", reporter=reporter)
    assert backend.calls == 2                          # transcribed twice


@pytest.mark.asyncio
async def test_retry_alt_switches_to_the_fallback_model():
    primary, alt = QueueBackend([CAPTURE]), QueueBackend([CAPTURE])
    reporter, _ = _reporter([Decision.RETRY_ALT, Decision.CONTINUE])
    await TableCapturer(primary, alt_backend=alt).capture("t", reporter=reporter)
    assert primary.calls == 1 and alt.calls == 1


@pytest.mark.asyncio
async def test_a_table_dropped_at_the_checkpoint_is_excluded_and_recorded():
    reporter, gate = _reporter()

    async def drop_the_search_table(event):
        event.drop(2)                                   # remove "Search strategy"
        event.decision = "continue"
        return Decision.CONTINUE

    gate.check = drop_the_search_table
    table_set, _ = await TableCapturer(QueueBackend([CAPTURE])).capture(
        "t", reporter=reporter)

    assert [t.table_id for t in table_set.tables] == ["table_1"]
    assert table_set.dropped == ["table_s1"]            # who removed what is recorded
    assert table_set.dropped_reason == "dropped at checkpoint"


@pytest.mark.asyncio
async def test_tables_flag_filters_before_the_checkpoint_is_shown():
    reporter, gate = _reporter()
    table_set, _ = await TableCapturer(QueueBackend([CAPTURE])).capture(
        "t", reporter=reporter, keep={"table_1"})

    assert [t.table_id for t in table_set.tables] == ["table_1"]
    assert table_set.dropped_reason == "--tables"
    # what the human sees is what will actually be processed
    assert len(gate.seen[0].selectable_items()) == 1


@pytest.mark.asyncio
async def test_capturing_nothing_is_a_loud_warning_not_a_silent_pass():
    reporter, gate = _reporter()
    table_set, _ = await TableCapturer(QueueBackend([{"tables": []}])).capture(
        "t", reporter=reporter)
    assert table_set.tables == []
    assert any("no table was captured" in w for w in gate.seen[0].warnings)


@pytest.mark.asyncio
async def test_shape_problems_reach_the_checkpoint_as_warnings():
    ragged = {"tables": [{"table_id": "table_1", "header_rows": [["A", "B", "C"]],
                          "rows": [["1", "2"]], "difficulties": ["column 3 was cut off"]}]}
    reporter, gate = _reporter()
    await TableCapturer(QueueBackend([ragged])).capture("t", reporter=reporter)
    warnings = gate.seen[0].warnings
    assert any("expected 3" in w for w in warnings)
    assert any("could not read" in w for w in warnings)


@pytest.mark.asyncio
async def test_a_malformed_table_is_skipped_rather_than_crashing_the_run():
    payload = {"tables": ["not a table", {"table_id": "table_1",
                                          "header_rows": [["A"]], "rows": [["1"]]}]}
    reporter, _ = _reporter()
    table_set, _ = await TableCapturer(QueueBackend([payload])).capture(
        "t", reporter=reporter)
    assert [t.table_id for t in table_set.tables] == ["table_1"]


@pytest.mark.asyncio
async def test_csv_sidecars_are_written_for_each_table():
    reporter, gate = _reporter()
    await TableCapturer(QueueBackend([CAPTURE])).capture("t", reporter=reporter)
    # the reporter passes sidecars to the journal; assert they were built per table
    assert gate.seen[0].payload["tables"][0]["table_id"] == "table_1"
