"""live.json is the in-flight step the desk polls while a stage is still working."""
from __future__ import annotations

import json
import time

import pytest

from react_review.hitl import RunJournal, ScriptedCheckpoint, StepReporter, StepStage


def _reporter(tmp_path):
    journal = RunJournal(tmp_path)
    return StepReporter("r1", gate=ScriptedCheckpoint([]), journal=journal)


def test_progress_writes_live_json_without_a_console_gate(tmp_path):
    reporter = _reporter(tmp_path)
    started = time.monotonic() - 5
    reporter.progress("figure", 2, 4, caption="fig_2", started=started)
    path = tmp_path / "live.json"
    assert path.is_file()
    live = json.loads(path.read_text(encoding="utf-8"))
    assert live["stage"] == "forest_ocr"
    assert live["caption"] == "fig_2"
    assert live["index"] == 2
    assert live["total"] == 4
    assert live["state"] == "running"
    assert live["started_unix"] < time.time()
    log = (tmp_path / "checkpoints.log").read_text(encoding="utf-8")
    assert "progress figure" in log
    assert "fig_2" in log
    assert "2/4" in log


@pytest.mark.asyncio
async def test_step_clears_live_json_before_the_gate(tmp_path):
    reporter = _reporter(tmp_path)
    reporter.progress("review_lens", started=time.monotonic())
    assert (tmp_path / "live.json").is_file()
    await reporter.step(StepStage.REVIEW_LENS, title="Review lens compressed")
    assert not (tmp_path / "live.json").is_file()


def test_null_journal_progress_does_not_write(tmp_path):
    reporter = StepReporter()
    reporter.progress("figure", 1, 1, caption="x")
    assert not (tmp_path / "live.json").is_file()
