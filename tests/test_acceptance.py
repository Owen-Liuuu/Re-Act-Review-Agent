"""A gate that can only pass or fail will be made to pass.

So these tests hold the third outcome in place. "Not estimable" has to survive
contact with a result that looks good, a safety failure has to outrank it, and
an interval over studies has to refuse to exist when there is one study — the
case that made "60%" and "80%" from a fifteen-row checkpoint unrankable in the
first place.
"""
from __future__ import annotations

import json

import pytest

from react_review.acceptance import (
    FAIL,
    NOT_ESTIMABLE,
    PASS,
    AcceptanceGate,
    CapabilityGate,
    HardGate,
    SampleRequirement,
    cluster_bootstrap,
    evaluate_gate,
    load_gate,
)
from react_review.contracts import ContractError, repo_root

GATE_FILE = repo_root() / "configs" / "gates" / "cross_domain_v1.json"


def _row(study, expected, predicted, found=True):
    return {"study_id": study, "expected_label": expected,
            "predicted_label": predicted, "found": found,
            "expected_match_mode": "numeric"}


def _gate(**overrides) -> AcceptanceGate:
    body = dict(
        gate_id="test", version=1, path=GATE_FILE, sha256="ABC",
        bootstrap_resamples=200,
        sample=SampleRequirement(min_domains=1, min_studies_per_domain=1,
                                 min_rows_per_domain=1, min_true_discrepancies=1),
        hard_gates=[HardGate(metric="silent_release_count", must_equal=0)],
        capability_gates=[CapabilityGate(metric="discrepancy_recall",
                                         lower_bound_at_least=0.70)])
    body.update(overrides)
    return AcceptanceGate(**body)


def _clean() -> dict[str, float]:
    return {"silent_release_count": 0}


# --- the third outcome ----------------------------------------------------

def test_one_study_supports_no_interval():
    """Fifteen rows from one paper are not fifteen observations."""
    rows = [_row("s1", "mismatch", "mismatch") for _ in range(15)]
    outcome = evaluate_gate(_gate(), rows, hard_counts=_clean())
    assert outcome.status == NOT_ESTIMABLE
    assert outcome.capability[0]["point"] == 1.0        # a perfect point estimate…
    assert outcome.capability[0]["lower"] is None       # …and no claim from it
    assert any("at least two" in reason for reason in outcome.blocking)


def test_a_missing_minimum_blocks_even_a_flawless_result():
    rows = [_row(f"s{i}", "mismatch", "mismatch") for i in range(3)]
    outcome = evaluate_gate(
        _gate(sample=SampleRequirement(min_domains=3, min_studies_per_domain=1,
                                       min_rows_per_domain=1,
                                       min_true_discrepancies=1)),
        rows, hard_counts=_clean())
    assert outcome.status == NOT_ESTIMABLE
    assert any("domain(s)" in reason for reason in outcome.blocking)


def test_a_safety_failure_outranks_insufficient_evidence():
    """"We cannot estimate accuracy" is no defence for having released one."""
    rows = [_row("s1", "mismatch", "match")]
    outcome = evaluate_gate(_gate(), rows,
                            hard_counts={"silent_release_count": 1})
    assert outcome.status == FAIL
    assert outcome.hard[0]["passed"] is False


def test_a_missing_hard_count_is_a_failure_not_a_pass():
    """An unreported safety number is not a zero."""
    outcome = evaluate_gate(_gate(), [_row("s1", "mismatch", "mismatch")],
                            hard_counts={})
    assert outcome.hard[0]["passed"] is False
    assert outcome.status == FAIL


# --- capability -----------------------------------------------------------

def _many_studies(recall_pattern):
    rows = []
    for index, caught in enumerate(recall_pattern):
        rows.append(_row(f"s{index}", "mismatch",
                         "mismatch" if caught else "match"))
        rows.append(_row(f"s{index}", "match", "match"))
    return rows


def test_a_lower_bound_below_the_bar_fails_even_with_a_good_point_estimate():
    rows = _many_studies([True] * 7 + [False] * 3)     # point 0.70, bound lower
    outcome = evaluate_gate(_gate(), rows, hard_counts=_clean())
    assert outcome.status == FAIL
    assert outcome.capability[0]["point"] == pytest.approx(0.70)
    assert outcome.capability[0]["lower"] < 0.70


def test_a_clean_result_over_enough_studies_passes():
    outcome = evaluate_gate(_gate(), _many_studies([True] * 12),
                            hard_counts=_clean())
    assert outcome.status == PASS


def test_the_interval_is_deterministic():
    rows = _many_studies([True] * 8 + [False] * 2)
    first = evaluate_gate(_gate(), rows, hard_counts=_clean())
    second = evaluate_gate(_gate(), rows, hard_counts=_clean())
    assert first.capability[0]["lower"] == second.capability[0]["lower"]


def test_resampling_is_over_studies_not_rows():
    """One study of 100 rows must not look like 100 studies of one row."""
    clustered = [[_row("s1", "mismatch", "match")] * 50]
    spread = [[_row(f"s{i}", "mismatch", "match")] for i in range(50)]
    _, lower_one, _, note = cluster_bootstrap(clustered, lambda rows: 0.5,
                                              resamples=50)
    assert lower_one is None and "at least two" in note
    _, lower_many, _, note = cluster_bootstrap(spread, lambda rows: 0.5,
                                               resamples=50)
    assert lower_many is not None and note == ""


# --- the shipped gate -----------------------------------------------------

def test_the_shipped_gate_is_loadable_and_pre_registered():
    gate = load_gate(GATE_FILE)
    assert gate.gate_id == "cross_domain_v1"
    assert gate.status == "provisional"        # not yet signed off by a clinician
    assert gate.sample.held_out_domain_required is True
    assert {h.metric for h in gate.hard_gates} >= {
        "silent_release_count", "wrong_target_released_count",
        "wrong_scope_released_count", "review_visibility_rate"}


def test_label_accuracy_is_reported_and_never_gated():
    """It mixes safety, capability and coverage; a system can raise it by refusing less."""
    gate = load_gate(GATE_FILE)
    assert "label_accuracy" in gate.reported_only
    assert "label_accuracy" not in {c.metric for c in gate.capability_gates}


def test_an_unknown_statistic_is_refused():
    with pytest.raises(ContractError, match="unknown statistic"):
        evaluate_gate(_gate(capability_gates=[CapabilityGate(
            metric="vibes", lower_bound_at_least=0.5)]),
            [_row("s1", "mismatch", "mismatch")], hard_counts=_clean())


def test_a_gate_without_an_id_is_refused(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"status": "provisional"}), encoding="utf-8")
    with pytest.raises(ContractError, match="gate_id"):
        load_gate(path)
