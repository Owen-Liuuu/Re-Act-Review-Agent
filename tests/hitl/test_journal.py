"""The run journal: per-step artifacts land on disk before the gate is asked."""
from __future__ import annotations

import json

import pytest

from react_review.core.exceptions import RunStopped
from react_review.hitl import (
    Decision,
    RunJournal,
    ScriptedCheckpoint,
    StepReporter,
    StepStage,
    SubjectKind,
)


def _reporter(tmp_path, decisions=None):
    journal = RunJournal(tmp_path)
    gate = ScriptedCheckpoint(decisions or [])
    return StepReporter("r1", gate=gate, journal=journal), journal, gate


@pytest.mark.asyncio
async def test_step_writes_artifact_and_ndjson(tmp_path):
    reporter, journal, _ = _reporter(tmp_path)
    await reporter.step(
        StepStage.TABLE_CAPTURE, title="Main table", subject="C:/x/review.pdf",
        subject_kind=SubjectKind.REVIEW_PDF, payload={"rows": [["a", "b"]]},
        render_blocks=["| a | b |"], warnings=["ragged row 2"],
    )
    artifact = tmp_path / "steps" / "001_review_table_capture.json"
    assert artifact.is_file()
    data = json.loads(artifact.read_text(encoding="utf-8"))
    assert data["subject"] == "C:/x/review.pdf"          # WHICH file — requirement #1
    assert data["payload"] == {"rows": [["a", "b"]]}     # FULL content — requirement #2
    assert data["warnings"] == ["ragged row 2"]

    line = json.loads((tmp_path / "journal.ndjson").read_text(encoding="utf-8").strip())
    assert line["stage"] == "review_table_capture" and line["warnings"] == 1


@pytest.mark.asyncio
async def test_artifact_survives_a_stop(tmp_path):
    # The artifact is written BEFORE the gate answers, so stopping still leaves it.
    reporter, _, _ = _reporter(tmp_path, [Decision.STOP])
    with pytest.raises(RunStopped) as excinfo:
        await reporter.step_or_stop(StepStage.TABLE_CAPTURE, payload={"x": 1})
    assert excinfo.value.stage == "review_table_capture"
    assert (tmp_path / "steps" / "001_review_table_capture.json").is_file()


@pytest.mark.asyncio
async def test_decision_is_folded_back_into_the_artifact(tmp_path):
    reporter, _, _ = _reporter(tmp_path, [Decision.CONTINUE])
    await reporter.step(StepStage.AUDIT_SUMMARY)
    data = json.loads(
        (tmp_path / "steps" / "001_audit_summary.json").read_text(encoding="utf-8"))
    assert data["decision"] == "continue"


@pytest.mark.asyncio
async def test_sidecars_and_incrementing_index(tmp_path):
    reporter, _, _ = _reporter(tmp_path)
    await reporter.step(StepStage.TABLE_CAPTURE, sidecars={"table_1.csv": "a,b\n1,2\n"})
    await reporter.step(StepStage.LONG_FORMAT_ROWS)
    assert (tmp_path / "steps" / "001_review_table_capture.table_1.csv").is_file()
    assert (tmp_path / "steps" / "002_long_format_rows.json").is_file()
    assert len((tmp_path / "journal.ndjson").read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.asyncio
async def test_default_reporter_never_blocks_and_never_writes(tmp_path):
    # The library default: no journal, no gate — tests and eval runners are unaffected.
    assert await StepReporter().step(StepStage.AUDIT_SUMMARY) is Decision.CONTINUE
    assert list(tmp_path.iterdir()) == []
