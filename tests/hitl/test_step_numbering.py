"""Visible [N] is counted from printed screens, never a hardcoded stage map."""
from __future__ import annotations

import pytest

from react_review.hitl import (
    CheckpointPolicy,
    Decision,
    Mode,
    RunJournal,
    StepReporter,
    StepStage,
    render_screen,
)
from react_review.hitl.events import StepEvent
from react_review.hitl.policy import CheckpointPolicy as Policy


def visible_screens(events: list[StepEvent]) -> list[int]:
    return [event.screen for event in events if event.screen]


def assert_screens_consecutive(screens: list[int]) -> None:
    """Visible numbers may repeat (folded artifacts) but must not skip or go back."""
    numbered = [n for n in screens if n]
    unique: list[int] = []
    for n in numbered:
        if not unique or unique[-1] != n:
            unique.append(n)
    assert unique == list(range(1, len(unique) + 1)), numbered
    assert numbered == sorted(numbered), numbered


def test_a_forged_gap_turns_the_numbering_check_red():
    """Tamper: a skipped visible number must fail, not be silently accepted."""
    with pytest.raises(AssertionError):
        assert_screens_consecutive([1, 2, 4])


class _PolicyContinue:
    def __init__(self, policy=None) -> None:
        self._policy = policy or CheckpointPolicy.key_stages()

    async def check(self, event, *, force_gate=False, hold_display=False):
        event.decision = Decision.CONTINUE.value
        return Decision.CONTINUE


@pytest.mark.asyncio
async def test_show_gate_and_silent_combination_numbers_without_gaps(tmp_path):
    reporter = StepReporter("r", gate=_PolicyContinue(), journal=RunJournal(tmp_path))
    await reporter.step(StepStage.REVIEW_PDF_LOADED, title="Review PDF loaded")
    await reporter.step(StepStage.REVIEW_LENS, title="Review focus")
    await reporter.step(StepStage.COHORT_REGISTRY, title="Cohorts",
                        hold_display=True)
    await reporter.step(StepStage.FIELD_RESOLUTION, title="Concepts",
                        hold_display=True)
    await reporter.step(StepStage.CHECKLIST_REVIEW, title="Checklist")
    await reporter.step(StepStage.LONG_FORMAT_ROWS, title="Long-format rows")
    await reporter.step(StepStage.REFERENCE_COVERAGE, title="Reference coverage")
    await reporter.step(StepStage.CHECKLIST_STUDY_COVERAGE, title="Study checklist")
    await reporter.step(StepStage.COLLECTION_REVIEW, title="Collection review")

    events = []
    for path in sorted((tmp_path / "steps").glob("*.json")):
        events.append(StepEvent.model_validate_json(path.read_text(encoding="utf-8")))
    screens = visible_screens(events)
    assert_screens_consecutive(screens)
    by_stage = {e.stage: e.screen for e in events}
    assert by_stage[StepStage.REVIEW_PDF_LOADED] == 0
    assert by_stage[StepStage.CHECKLIST_STUDY_COVERAGE] == 0
    assert by_stage[StepStage.COHORT_REGISTRY] == by_stage[StepStage.FIELD_RESOLUTION]
    assert by_stage[StepStage.FIELD_RESOLUTION] == by_stage[StepStage.CHECKLIST_REVIEW]
    assert by_stage[StepStage.REVIEW_LENS] == 1
    assert by_stage[StepStage.COHORT_REGISTRY] == 2
    assert by_stage[StepStage.LONG_FORMAT_ROWS] == 3
    assert by_stage[StepStage.REFERENCE_COVERAGE] == 4
    assert by_stage[StepStage.COLLECTION_REVIEW] == 5


@pytest.mark.asyncio
async def test_empty_cohorts_still_force_a_pause_on_the_merged_screen(capsys):
    from react_review.hitl.console import ConsoleCheckpoint

    asked: list[StepStage] = []
    gate = ConsoleCheckpoint(CheckpointPolicy.key_stages())
    orig = gate._ask

    async def counting_ask(event):
        asked.append(event.stage)
        return Decision.CONTINUE

    gate._ask = counting_ask  # type: ignore[method-assign]
    reporter = StepReporter("r", gate=gate)
    await reporter.step(StepStage.COHORT_REGISTRY, title="Cohorts",
                        payload={"cohorts": [], "unassigned": ["x"]},
                        force_gate=True, hold_display=True)
    await reporter.step(StepStage.FIELD_RESOLUTION, title="Concepts",
                        payload={"knowledge_base": {
                            "fingerprint": "abc", "concept_count": 1}})
    assert asked == [StepStage.FIELD_RESOLUTION]
    out = capsys.readouterr().out
    assert "Review parsed" in out
    assert "cohorts:" in out
    assert "0 found" in out


def test_merged_review_parsed_screen_has_three_lines():
    events = [
        StepEvent(run_id="r", index=5, screen=5,
                  stage=StepStage.COHORT_REGISTRY, title="Cohorts",
                  payload={"cohorts": [
                      {"key": "mie", "display": "MIE", "source": "discovered"},
                      {"key": "oe", "display": "OE", "source": "discovered"}]}),
        StepEvent(run_id="r", index=6, screen=5,
                  stage=StepStage.FIELD_RESOLUTION, title="Concepts",
                  payload={"knowledge_base": {
                      "fingerprint": "a0206b62ffff", "concept_count": 12}}),
        StepEvent(run_id="r", index=7, screen=5,
                  stage=StepStage.CHECKLIST_REVIEW, title="Checklist",
                  payload={"name": "clinical-review-default", "version": "1",
                           "assessments": [{}], "gaps": []}),
    ]
    out = render_screen(events)
    assert "[5] Review parsed" in out
    assert 'cohorts:    2 found — mie "MIE" · oe "OE"   [discovered]' in out
    assert "concepts:   12 in KB · fingerprint a0206b62" in out
    assert "checklist:  clinical-review-default v1 · 1 item · 0 required gap" in out


def test_key_stages_still_has_nine_gates():
    policy = Policy.key_stages()
    gated = [s for s in StepStage if policy.mode_for(s) is Mode.GATE]
    used = [
        StepStage.REVIEW_LENS, StepStage.EVIDENCE_LOCALIZE,
        StepStage.TABLE_CAPTURE, StepStage.CLAIM_ORIGIN,
        StepStage.LONG_FORMAT_ROWS, StepStage.REFERENCE_COVERAGE,
        StepStage.COLLECTION_REVIEW, StepStage.AUDIT_SUMMARY,
        StepStage.JUDGE_FLAGS,
    ]
    assert all(s in gated for s in used)
    assert policy.mode_for(StepStage.CHECKLIST_STUDY_COVERAGE) is Mode.SILENT
