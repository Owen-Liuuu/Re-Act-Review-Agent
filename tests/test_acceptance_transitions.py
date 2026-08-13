"""The gate's verdict is computed, not argued beside the result.

Gate v1 had no classifier. Its transitions were worked out by a throwaway script
written next to the run, and that script called a review-flagged row
`wrong_released` — which v1 explicitly defines as a wrong value released WITHOUT
review. It reported FAIL for a transition the gate does not define, and nothing
could contradict it because there was nothing to contradict it with.
"""
from __future__ import annotations

import json

from react_review.acceptance_transitions import (
    CORRECT,
    REFUSED,
    WRONG_BUT_FLAGGED,
    WRONG_RELEASED,
    grade,
    row_state,
)
from react_review.contracts import repo_root


def _row(audit_id, predicted, expected, review=False, field="hazard_ratio"):
    return {"audit_id": audit_id, "field_type": field,
            "predicted_label": predicted, "expected_label": expected,
            "review_required": review}


def _gate(states, **over):
    body = {
        "baseline_rows": [{"audit_id": a, "baseline_state": s}
                          for a, s in states.items()],
        "transitions": {
            "allowed": {"correct -> correct": "", "correct -> refused": "",
                        "refused -> correct": "", "refused -> refused": "",
                        "correct -> wrong_but_flagged": ""},
            "forbidden": {"correct -> wrong_released": "",
                          "refused -> wrong_released": ""},
        },
    }
    body.update(over)
    return body


# --- the fourth state -------------------------------------------------------

def test_a_wrong_row_that_was_escalated_is_not_a_released_error():
    """"Released" is literal everywhere else in this project."""
    assert row_state(predicted_label="match", expected_label="mismatch",
                     review_required=True) == WRONG_BUT_FLAGGED
    assert row_state(predicted_label="match", expected_label="mismatch",
                     review_required=False) == WRONG_RELEASED


def test_a_refusal_is_a_refusal_whatever_it_was_asked():
    for label in ("not_comparable", "review_required"):
        assert row_state(predicted_label=label, expected_label="match",
                         review_required=False) == REFUSED


def test_agreement_with_the_key_is_correct():
    assert row_state(predicted_label="mismatch", expected_label="mismatch",
                     review_required=True) == CORRECT


# --- a gate that cannot express what happened does not get to judge ---------

def test_a_transition_the_gate_never_defined_is_not_evaluable():
    """Reporting it as FAIL would attribute a fault of the gate to the system."""
    gate = _gate({"MA015": CORRECT})
    gate["transitions"]["allowed"].pop("correct -> wrong_but_flagged")
    result = grade(gate, [_row("MA015", "match", "mismatch", review=True)])
    assert result.verdict == "not_evaluable"
    assert "does not express" in result.reason


def test_a_row_the_gate_never_pre_registered_is_not_evaluable():
    result = grade(_gate({"MA001": CORRECT}), [_row("MA999", "match", "match")])
    assert result.verdict == "not_evaluable"
    assert "baseline does not contain" in result.reason


# --- prohibitions alone are not a pass --------------------------------------

def test_refusing_everything_satisfies_every_prohibition():
    """The hole in v1: a system that answers nothing breaks no rule."""
    gate = _gate({"A": CORRECT, "B": CORRECT})
    result = grade(gate, [_row("A", "not_comparable", "match"),
                          _row("B", "not_comparable", "match")])
    assert not result.violations
    assert result.verdict == "pass_prohibitions_only"
    assert result.capability_judged is False
    assert result.capability["correct"] == 0


def test_a_floor_turns_that_into_a_failure():
    gate = _gate({"A": CORRECT, "B": CORRECT},
                 capability_floor={"min_fraction_of_baseline_correct": 0.5})
    result = grade(gate, [_row("A", "not_comparable", "match"),
                          _row("B", "not_comparable", "match")])
    assert result.verdict == "fail"
    assert any("capability floor" in u for u in result.unmet_hard_conditions)


def test_a_met_floor_is_a_full_pass():
    gate = _gate({"A": CORRECT, "B": CORRECT},
                 capability_floor={"min_fraction_of_baseline_correct": 0.5})
    result = grade(gate, [_row("A", "match", "match"),
                          _row("B", "not_comparable", "match")])
    assert result.verdict == "pass" and result.capability_judged


def test_a_forbidden_transition_fails_whatever_the_capability():
    gate = _gate({"A": CORRECT, "B": CORRECT},
                 capability_floor={"min_fraction_of_baseline_correct": 0.0})
    result = grade(gate, [_row("A", "match", "mismatch"),      # not flagged
                          _row("B", "match", "match")])
    assert result.verdict == "fail"
    assert [t.audit_id for t in result.violations] == ["A"]


# --- the two published gates, on the run they were written for --------------

def _scored_rows():
    path = (repo_root() / "output/baselines/melanoma_checkpoint_2017"
            / "d1_7_scored.json")
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))["rows"]


def _published(version):
    return json.loads((repo_root() / f"configs/gates/d1_batch_{version}.json"
                       ).read_text(encoding="utf-8"))


def test_v1_cannot_judge_the_recording_it_was_written_for():
    """The FAIL reported on the day was never v1's verdict."""
    rows = _scored_rows()
    if rows is None:
        import pytest

        pytest.skip("the recording is local-only and not in this checkout")
    result = grade(_published("v1"), rows)
    assert result.verdict == "not_evaluable"
    assert "MA015" in result.reason


def test_v2_reaches_only_a_prohibitions_only_pass():
    """Its capability floor is deliberately unset, so it cannot say more."""
    rows = _scored_rows()
    if rows is None:
        import pytest

        pytest.skip("the recording is local-only and not in this checkout")
    result = grade(_published("v2"), rows)
    assert result.verdict == "pass_prohibitions_only"
    assert result.capability_judged is False
    assert result.capability["wrong_released"] == 0
    assert result.capability["wrong_but_flagged"] == 1


def test_v2_publishes_no_capability_threshold():
    """A draft set 0.8, and the run keeps exactly 8 of 10. A threshold chosen
    after seeing the number it must clear is not a threshold."""
    floor = _published("v2")["capability_floor"]
    assert floor["min_fraction_of_baseline_correct"] is None
    assert floor["min_correct_rows"] is None
    assert "chosen after seeing the result" in floor["why_unset"]


def test_v1_is_not_edited_to_change_its_own_verdict():
    v1 = _published("v1")
    assert "wrong_but_flagged" not in json.dumps(v1["transitions"])
    assert v1["gate_id"] == "d1_batch_v1"
