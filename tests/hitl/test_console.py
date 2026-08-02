"""The interactive checkpoint: key mapping, Windows quirks, and never wedging CI."""
from __future__ import annotations

import pytest

from react_review.hitl import (
    CheckpointPolicy,
    ConsoleCheckpoint,
    Decision,
    Mode,
    StepEvent,
    StepStage,
    render_prompt,
)


def _event(stage=StepStage.AUDIT_SUMMARY, **kw) -> StepEvent:
    base = dict(run_id="r", index=1, stage=stage, title="A step",
                subject="C:/pdf/paper.pdf", payload={"n": 1})
    base.update(kw)
    return StepEvent(**base)


def _gate(keys: list[str], **kw) -> ConsoleCheckpoint:
    """A checkpoint whose keyboard is a scripted list of keys."""
    gate = ConsoleCheckpoint(kw.pop("policy", CheckpointPolicy.key_stages()), **kw)
    pending = list(keys)
    gate._read_key = lambda: pending.pop(0)          # type: ignore[method-assign]
    return gate


# --- key mapping ---

@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["c", "\r", "\n", " "])
async def test_continue_keys(key):
    assert await _gate([key]).check(_event()) is Decision.CONTINUE


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["s", "q"])
async def test_stop_keys(key):
    event = _event()
    assert await _gate([key]).check(event) is Decision.STOP
    assert event.decision == "stop"


@pytest.mark.asyncio
async def test_unknown_key_reprompts_rather_than_deciding():
    assert await _gate(["z", "?", "c"]).check(_event()) is Decision.CONTINUE


@pytest.mark.asyncio
async def test_detail_prints_the_full_payload_then_reprompts(capsys):
    await _gate(["d", "c"]).check(_event(payload={"secret": "42"}))
    assert '"secret": "42"' in capsys.readouterr().out      # requirement: full content


@pytest.mark.asyncio
async def test_retry_is_only_accepted_when_the_stage_offers_it():
    # no offer → 'r' is meaningless and must not end the prompt
    assert await _gate(["r", "c"]).check(_event()) is Decision.CONTINUE
    assert await _gate(["r"]).check(_event(offers=["retry"])) is Decision.RETRY
    assert await _gate(["m"]).check(
        _event(offers=["retry_alt"])) is Decision.RETRY_ALT


# --- skip is withheld until explicitly allowed ---

@pytest.mark.asyncio
async def test_skip_is_ignored_without_allow_skip():
    assert await _gate(["a", "c"]).check(_event()) is Decision.CONTINUE


@pytest.mark.asyncio
async def test_skip_short_circuits_every_later_checkpoint():
    gate = _gate(["a"], allow_skip=True)
    assert await gate.check(_event()) is Decision.SKIP_REST
    # the scripted keyboard is now empty: a further prompt would raise IndexError
    assert await gate.check(_event(index=2)) is Decision.CONTINUE


# --- dropping a captured item at the checkpoint ---

def _droppable(**kw) -> StepEvent:
    return _event(
        stage=StepStage.TABLE_CAPTURE, selectable="tables",
        payload={"tables": [
            {"id": "table_1", "label": "table_1  Characteristics"},
            {"id": "table_2", "label": "table_2  Outcomes"},
            {"id": "table_s1", "label": "table_s1  Search strategy"},
        ]}, **kw)


@pytest.mark.asyncio
async def test_dropping_removes_the_item_and_records_it_then_reprompts(capsys):
    event = _droppable()
    assert await _gate(["x", "3", "c"]).check(event) is Decision.CONTINUE
    assert [t["id"] for t in event.selectable_items()] == ["table_1", "table_2"]
    assert event.dropped == ["table_s1"]           # human intervention is recorded
    assert "dropped table_s1 — 2 left" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_drop_is_not_offered_when_the_stage_has_nothing_selectable():
    # 'x' is then just an unknown key: re-prompt, never a decision
    assert await _gate(["x", "c"]).check(_event()) is Decision.CONTINUE
    assert "[X] drop one" not in render_prompt(_event())
    assert "[X] drop one" in render_prompt(_droppable())


@pytest.mark.asyncio
async def test_out_of_range_choice_drops_nothing(capsys):
    event = _droppable()
    await _gate(["x", "7", "c"]).check(event)
    assert len(event.selectable_items()) == 3 and event.dropped == []
    assert "(no such item)" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_non_digit_choice_aborts_the_drop(capsys):
    event = _droppable()
    await _gate(["x", "z", "c"]).check(event)
    assert len(event.selectable_items()) == 3 and event.dropped == []


@pytest.mark.asyncio
async def test_dropping_everything_warns_before_continuing(capsys):
    event = _droppable()
    await _gate(["x", "1", "x", "1", "x", "1", "c"]).check(event)
    assert event.selectable_items() == []
    assert "nothing left to process" in capsys.readouterr().out


# --- policy modes ---

@pytest.mark.asyncio
async def test_show_mode_prints_but_never_asks(capsys):
    gate = _gate([], policy=CheckpointPolicy.key_stages())    # COLLECT_STUDY is SHOW
    assert await gate.check(_event(StepStage.COLLECT_STUDY)) is Decision.CONTINUE
    out = capsys.readouterr().out
    assert "C:/pdf/paper.pdf" in out and "[C]ontinue" not in out


@pytest.mark.asyncio
async def test_silent_mode_prints_nothing(capsys):
    gate = _gate([], policy=CheckpointPolicy.none())
    assert await gate.check(_event()) is Decision.CONTINUE
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_all_stages_policy_gates_each_source_paper():
    gate = _gate(["c"], policy=CheckpointPolicy.all_stages())
    assert await gate.check(_event(StepStage.COLLECT_STUDY)) is Decision.CONTINUE


# --- the real _read_key: Windows quirks and the non-TTY guard ---

def test_non_tty_continues_without_reading(monkeypatch):
    # A piped or scripted invocation must never hang waiting for a keypress.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("react_review.hitl.console.sys.stdin.isatty", lambda: False)
    assert ConsoleCheckpoint._read_key() == "c"


def test_arrow_key_prefix_is_swallowed(monkeypatch):
    import types
    keys = iter(["\xe0", "H"])                       # an arrow key arrives as two reads
    fake = types.SimpleNamespace(getwch=lambda: next(keys))
    monkeypatch.setattr("react_review.hitl.console.sys.stdin.isatty", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "msvcrt", fake)
    assert ConsoleCheckpoint._read_key() == ""       # → re-prompt, not a decision


def test_ctrl_c_raises_keyboard_interrupt(monkeypatch):
    import types
    fake = types.SimpleNamespace(getwch=lambda: "\x03")
    monkeypatch.setattr("react_review.hitl.console.sys.stdin.isatty", lambda: True)
    monkeypatch.setitem(__import__("sys").modules, "msvcrt", fake)
    with pytest.raises(KeyboardInterrupt):           # getwch does not raise it for us
        ConsoleCheckpoint._read_key()
