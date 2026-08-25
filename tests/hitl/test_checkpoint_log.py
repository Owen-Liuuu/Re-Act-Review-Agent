"""Run-level checkpoints.log: every step, diffable, no second renderer."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from react_review.hitl import (
    AutoContinue,
    CheckpointPolicy,
    ConsoleCheckpoint,
    Decision,
    RunJournal,
    StepEvent,
    StepReporter,
    StepStage,
    checkpoint_log_header,
    render_screen,
)
from react_review.hitl.policy import Mode


def _event(**kw) -> StepEvent:
    base = dict(
        run_id="r1", index=6, screen=6, stage=StepStage.CLAIM_ORIGIN,
        title="Claim origin (source vs review-computed)",
        subject=r"C:\papers\doc05.pdf",
        render_blocks=["  3 origin label(s)"],
        elapsed_ms=1234,
        backend_profile="judge",
        backend_model_id="deepseek-v4-pro",
        backend_reasoning="on",
        backend_reasoning_tokens=27,
        interaction="gate",
        decision="continue",
    )
    base.update(kw)
    return StepEvent(**base)


def test_stable_render_matches_live_except_the_three_volatile_lines():
    """Acceptance 5: same renderer; stable only drops elapsed / path / backend."""
    event = _event()
    live = render_screen([event])
    stable = render_screen([event], stable=True)
    assert "  (1s)" in live
    assert "file: C:\\papers\\doc05.pdf" in live or "file: C:/papers/doc05.pdf" in live
    assert "model_id: deepseek-v4-pro" in live
    assert "  (1s)" not in stable
    assert "file:" not in stable
    assert "model_id:" not in stable
    assert "profile:" not in stable
    assert "interaction: gate" in stable
    assert "decision: continue" in stable
    assert "interaction:" not in live

    stripped = event.model_copy(update={
        "elapsed_ms": 0,
        "subject": "",
        "backend_profile": "",
        "backend_model_id": "",
        "backend_reasoning": "",
        "backend_reasoning_tokens": None,
    })
    live_body = render_screen([stripped])
    stable_body = "\n".join(
        line for line in stable.splitlines()
        if "interaction:" not in line and "decision:" not in line
    )
    assert live_body == stable_body
    assert "  3 origin label(s)" in live_body
    assert "  3 origin label(s)" in stable


def _header(model: str = "glm-a") -> str:
    return checkpoint_log_header(
        run_id="r1", review="doc05.pdf",
        models={"llm": model},
        prompts={"table_capture": "table_capture_v3"},
        config_summary="checkpoints=none extraction=replay",
    )


async def _two_steps(tmp: Path, header: str, *, subject: str, started_offset: float):
    reporter = StepReporter("r1", gate=AutoContinue(), journal=RunJournal(tmp))
    reporter.checkpoint_header = header
    started = time.monotonic() - started_offset
    await reporter.step(
        StepStage.REVIEW_PDF_LOADED, title="Review PDF loaded",
        subject=subject, started=started)
    await reporter.step(
        StepStage.REVIEW_LENS, title="Review lens compressed",
        subject=subject, render_blocks=["  review focus: elderly ESCC"],
        started=started)
    return tmp / "checkpoints.log"


@pytest.mark.asyncio
async def test_two_offline_runs_diff_empty(tmp_path):
    header = _header("glm-a")
    a = await _two_steps(
        tmp_path / "a", header,
        subject=r"C:\runs\a\doc05.pdf", started_offset=1.2)
    b = await _two_steps(
        tmp_path / "b", header,
        subject=r"D:\other\doc05.pdf", started_offset=9.4)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    text = a.read_text(encoding="utf-8")
    assert "file:" not in text
    assert "doc05.pdf" in text.splitlines()[1]
    assert "(1s)" not in text and "(9s)" not in text


@pytest.mark.asyncio
async def test_changing_the_model_diffs_only_the_header(tmp_path):
    body_a = await _two_steps(
        tmp_path / "a", _header("glm-a"),
        subject=r"C:\doc05.pdf", started_offset=1.0)
    body_b = await _two_steps(
        tmp_path / "b", _header("glm-b"),
        subject=r"C:\doc05.pdf", started_offset=1.0)
    lines_a = body_a.read_text(encoding="utf-8").splitlines()
    lines_b = body_b.read_text(encoding="utf-8").splitlines()
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(lines_a, lines_b)) if x != y]
    assert diffs
    assert all("model." in x or "model." in y for _, x, y in diffs)
    header_end = next(i for i, line in enumerate(lines_a) if line.startswith("──") or line.startswith("--"))
    assert all(i < header_end for i, _, _ in diffs)


@pytest.mark.asyncio
async def test_silent_steps_are_in_the_log(tmp_path):
    gate = ConsoleCheckpoint(CheckpointPolicy.key_stages())
    gate._read_key = lambda: "c"  # type: ignore[method-assign]
    reporter = StepReporter("r1", gate=gate, journal=RunJournal(tmp_path))
    reporter.checkpoint_header = _header()
    await reporter.step(
        StepStage.REVIEW_PDF_LOADED, title="Review PDF loaded",
        subject=r"C:\doc05.pdf")
    await reporter.step(
        StepStage.REVIEW_LENS, title="Review lens compressed",
        subject=r"C:\doc05.pdf")
    text = (tmp_path / "checkpoints.log").read_text(encoding="utf-8")
    assert "Review PDF loaded" in text
    assert "interaction: silent" in text
    assert "interaction: gate" in text
    assert "decision: continue" in text
    policy = CheckpointPolicy.key_stages()
    assert policy.mode_for(StepStage.REVIEW_PDF_LOADED) is Mode.SILENT


@pytest.mark.asyncio
async def test_null_journal_writes_no_checkpoint_log(tmp_path):
    await StepReporter("r1").step(StepStage.AUDIT_SUMMARY, title="Audit")
    assert list(tmp_path.iterdir()) == []
