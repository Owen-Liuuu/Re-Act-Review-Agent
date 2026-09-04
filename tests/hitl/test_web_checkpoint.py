"""WebCheckpoint: wait for a posted decision; mark interaction=web."""
from __future__ import annotations

import asyncio

import pytest

from react_review.hitl import Decision, StepEvent, StepStage, WebCheckpoint
from react_review.hitl.web import StaleDecision


def _event(index: int = 2) -> StepEvent:
    return StepEvent(
        run_id="r1", index=index, screen=index,
        stage=StepStage.REVIEW_LENS, title="Review lens compressed",
        render_blocks=["  review focus"],
    )


@pytest.mark.asyncio
async def test_web_checkpoint_records_web_interaction():
    gate = WebCheckpoint()
    event = _event()

    async def decide():
        while gate.waiting is None:
            await asyncio.sleep(0.01)
        gate.submit(event.index, Decision.CONTINUE)

    decision, _ = await asyncio.gather(gate.check(event), decide())
    assert decision is Decision.CONTINUE
    assert event.interaction == "web"
    assert event.decision == "continue"


@pytest.mark.asyncio
async def test_silent_and_show_stages_do_not_wait():
    gate = WebCheckpoint()
    loaded = StepEvent(
        run_id="r1", index=1, stage=StepStage.REVIEW_PDF_LOADED, title="loaded")
    collect = StepEvent(
        run_id="r1", index=13, stage=StepStage.COLLECT_STUDY, title="paper")
    assert await gate.check(loaded) is Decision.CONTINUE
    assert loaded.interaction == "silent"
    assert await gate.check(collect) is Decision.CONTINUE
    assert collect.interaction == "show"


@pytest.mark.asyncio
async def test_stale_index_is_refused_and_does_not_eat_the_next_gate():
    gate = WebCheckpoint()
    event = _event(index=4)

    async def decide():
        while gate.waiting is None:
            await asyncio.sleep(0.01)
        with pytest.raises(StaleDecision):
            gate.submit(3, Decision.CONTINUE)
        gate.submit(4, Decision.STOP)

    decision, _ = await asyncio.gather(gate.check(event), decide())
    assert decision is Decision.STOP
    assert event.interaction == "web"


def test_web_checkpoint_progress_prints_a_line(capsys):
    WebCheckpoint().progress("review_lens", caption="front matter", elapsed_s=1.2)
    out = capsys.readouterr().out
    assert "review_lens" in out
    assert "front matter" in out
