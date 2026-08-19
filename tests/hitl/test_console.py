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
    assert await _gate(["m", "c"]).check(_event()) is Decision.CONTINUE
    assert await _gate(["m"]).check(
        _event(offers=["retry_alt"])) is Decision.RETRY_ALT
    # no offer → 'm' must not silently continue as if accepted
    assert await _gate(["m", "c"]).check(_event(offers=["retry"])) is Decision.CONTINUE


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
async def test_off_turns_an_item_off_and_records_it_then_reprompts(capsys):
    event = _droppable()
    assert await _gate(["f", "3", "c"]).check(event) is Decision.CONTINUE
    assert [t["id"] for t in event.selectable_items()] == [
        "table_1", "table_2", "table_s1"]
    assert event.dropped == ["table_s1"]
    assert event.selectable_items()[2]["label"].startswith("[off]")
    assert "set table_s1 off" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_t_key_no_longer_toggles(capsys):
    event = _droppable()
    assert await _gate(["t", "c"]).check(event) is Decision.CONTINUE
    assert event.dropped == []
    assert all("[off]" not in t["label"] for t in event.selectable_items())
    assert "set " not in capsys.readouterr().out


@pytest.mark.asyncio
async def test_on_and_off_are_idempotent(capsys):
    event = _droppable()
    assert await _gate(["f", "2", "f", "2", "n", "2", "n", "2", "c"]).check(
        event) is Decision.CONTINUE
    assert event.dropped == []
    assert event.selectable_items()[1]["label"].startswith("[on]")
    out = capsys.readouterr().out
    assert out.count("set table_2 off") == 2
    assert out.count("set table_2 on") == 2


@pytest.mark.asyncio
async def test_on_off_are_not_offered_when_the_stage_has_nothing_selectable():
    assert await _gate(["f", "c"]).check(_event()) is Decision.CONTINUE
    assert "[N]On <n>" not in render_prompt(_event())
    assert "[F]Off <n>" not in render_prompt(_event())
    assert "[T]Toggle one" not in render_prompt(_droppable())
    assert "[N]On <n>" in render_prompt(_droppable())
    assert "[F]Off <n>" in render_prompt(_droppable())
    assert "[U]Undo" not in render_prompt(_droppable())
    assert "[U]Undo" in render_prompt(_droppable(), undo_available=True)


@pytest.mark.asyncio
async def test_out_of_range_choice_sets_nothing(capsys):
    event = _droppable()
    await _gate(["f", "7", "c"]).check(event)
    assert len(event.selectable_items()) == 3 and event.dropped == []
    assert "(no such item)" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_non_digit_choice_aborts_the_set(capsys):
    event = _droppable()
    await _gate(["n", "z", "c"]).check(event)
    assert len(event.selectable_items()) == 3 and event.dropped == []


@pytest.mark.asyncio
async def test_setting_everything_off_warns_before_continuing(capsys):
    event = _droppable()
    await _gate(["f", "1", "f", "2", "f", "3", "c"]).check(event)
    assert [t["id"] for t in event.selectable_items()] == [
        "table_1", "table_2", "table_s1"]
    assert event.dropped == ["table_1", "table_2", "table_s1"]
    assert "nothing left to process" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_undo_restores_the_last_on_off_in_the_same_pause():
    event = _droppable()
    await _gate(["f", "1", "f", "2", "u", "u", "c"]).check(event)
    assert [t["id"] for t in event.selectable_items()] == [
        "table_1", "table_2", "table_s1"]
    assert event.dropped == []


@pytest.mark.asyncio
async def test_force_gate_pauses_a_show_or_silent_stage():
    policy = CheckpointPolicy.key_stages()
    assert policy.mode_for(StepStage.COHORT_REGISTRY) is Mode.SHOW
    assert policy.mode_for(StepStage.REVIEW_PDF_LOADED) is Mode.SILENT
    asked: list[StepStage] = []
    gate = _gate(["c", "c"], policy=policy)
    orig = gate._ask

    async def counting_ask(event):
        asked.append(event.stage)
        return await orig(event)

    gate._ask = counting_ask  # type: ignore[method-assign]
    event = _event(StepStage.COHORT_REGISTRY)
    assert await gate.check(event, force_gate=True) is Decision.CONTINUE
    assert event.decision == "continue"
    silent = _event(StepStage.REVIEW_PDF_LOADED)
    assert await gate.check(silent, force_gate=True) is Decision.CONTINUE
    assert asked == [StepStage.COHORT_REGISTRY, StepStage.REVIEW_PDF_LOADED]


@pytest.mark.asyncio
async def test_silent_pdf_loaded_neither_prints_nor_asks(capsys):
    gate = _gate([], policy=CheckpointPolicy.key_stages())
    assert await gate.check(_event(StepStage.REVIEW_PDF_LOADED)) is Decision.CONTINUE
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_show_and_silent_stages_still_write_journal(tmp_path):
    from react_review.hitl import RunJournal, StepReporter

    journal = RunJournal(tmp_path)
    gate = _gate([], policy=CheckpointPolicy.key_stages())
    reporter = StepReporter("r", gate=gate, journal=journal)
    await reporter.step_or_stop(StepStage.FIELD_RESOLUTION, title="Field concepts")
    await reporter.step_or_stop(StepStage.REVIEW_PDF_LOADED, title="Review PDF loaded")
    names = {path.name for path in (tmp_path / "steps").glob("*.json")}
    assert any("field_resolution" in name for name in names)
    assert any("review_pdf_loaded" in name for name in names)


@pytest.mark.asyncio
async def test_first_visible_screen_is_one_while_journal_keeps_pdf_loaded(tmp_path):
    from react_review.hitl import RunJournal, StepReporter, render_event

    class _PolicyContinue:
        def __init__(self) -> None:
            self._policy = CheckpointPolicy.key_stages()

        async def check(self, event, *, force_gate=False):
            event.decision = "continue"
            return Decision.CONTINUE

    journal = RunJournal(tmp_path)
    reporter = StepReporter("r", gate=_PolicyContinue(), journal=journal)
    await reporter.step(StepStage.REVIEW_PDF_LOADED, title="Review PDF loaded",
                        subject="C:/pdf/review.pdf")
    await reporter.step(StepStage.REVIEW_LENS, title="Review lens compressed",
                        subject="C:/pdf/review.pdf")
    names = {path.name for path in (tmp_path / "steps").glob("*.json")}
    assert any(name.startswith("001_review_pdf_loaded") for name in names)
    assert any(name.startswith("002_review_lens") for name in names)
    lens = reporter.last_event
    assert lens is not None and lens.screen == 1 and lens.index == 2
    out = render_event(lens)
    assert "[1] Review lens compressed" in out
    assert "\n\n  file: C:/pdf/review.pdf" in out



def test_key_stages_gates_nine_decision_points():
    policy = CheckpointPolicy.key_stages()
    gated = {s for s in StepStage if policy.mode_for(s) is Mode.GATE}
    show = {s for s in StepStage if policy.mode_for(s) is Mode.SHOW}
    silent = {s for s in StepStage if policy.mode_for(s) is Mode.SILENT}
    assert StepStage.FIELD_RESOLUTION in show
    assert StepStage.CHECKLIST_REVIEW in show
    assert StepStage.CHECKLIST_STUDY_COVERAGE in show
    assert StepStage.COHORT_REGISTRY in show
    assert StepStage.COLLECT_STUDY in show
    assert StepStage.REVIEW_PDF_LOADED in silent
    assert StepStage.FOREST_OCR in silent
    assert StepStage.CHECKLIST in gated
    used = [
        StepStage.REVIEW_LENS, StepStage.EVIDENCE_LOCALIZE,
        StepStage.TABLE_CAPTURE, StepStage.CLAIM_ORIGIN,
        StepStage.LONG_FORMAT_ROWS, StepStage.REFERENCE_COVERAGE,
        StepStage.COLLECTION_REVIEW, StepStage.AUDIT_SUMMARY,
        StepStage.JUDGE_FLAGS,
    ]
    assert all(s in gated for s in used)
    assert len(used) == 9


# --- policy modes ---

@pytest.mark.asyncio
async def test_show_mode_prints_but_never_asks(capsys):
    gate = _gate([], policy=CheckpointPolicy.key_stages())    # COLLECT_STUDY is SHOW
    assert await gate.check(_event(StepStage.COLLECT_STUDY)) is Decision.CONTINUE
    out = capsys.readouterr().out
    assert "C:/pdf/paper.pdf" in out and "[C]Continue" not in out


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


def test_retry_offers_hide_model_2_when_unwired():
    from react_review.hitl.gate import require_alt_backend, retry_offers

    assert retry_offers(None) == ["retry"]
    assert retry_offers(object()) == ["retry", "retry_alt"]
    with pytest.raises(RuntimeError, match="no alt_backend"):
        require_alt_backend(None, stage="review_table_capture")
    assert require_alt_backend("llm2", stage="review_lens") == "llm2"
