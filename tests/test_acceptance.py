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
    STATISTICS,
    FAIL,
    FAILED,
    MISSING,
    NO_DENOMINATOR,
    OK,
    Observation,
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


def _clean() -> dict:
    """A safety number that was actually measured, over something."""
    return {"silent_release_count": Observation(value=0, denominator=12,
                                                source="test")}


# --- the third outcome ----------------------------------------------------

def test_one_study_supports_no_interval():
    """Fifteen rows from one paper are not fifteen observations."""
    rows = [_row("s1", "mismatch", "mismatch") for _ in range(15)]
    outcome = evaluate_gate(_gate(), rows, evidence=_clean())
    assert outcome.status == NOT_ESTIMABLE
    assert outcome.capability[0]["point"] == 1.0        # a perfect point estimate…
    assert outcome.capability[0]["lower"] is None       # …and no claim from it
    assert any("nothing to resample" in reason for reason in outcome.blocking)


def test_a_missing_minimum_blocks_even_a_flawless_result():
    rows = [_row(f"s{i}", "mismatch", "mismatch") for i in range(3)]
    outcome = evaluate_gate(
        _gate(sample=SampleRequirement(min_domains=3, min_studies_per_domain=1,
                                       min_rows_per_domain=1,
                                       min_true_discrepancies=1)),
        rows, evidence=_clean())
    assert outcome.status == NOT_ESTIMABLE
    assert any("domain(s)" in reason for reason in outcome.blocking)


def test_a_safety_failure_outranks_insufficient_evidence():
    """"We cannot estimate accuracy" is no defence for having released one."""
    rows = [_row("s1", "mismatch", "match")]
    outcome = evaluate_gate(_gate(), rows, evidence={
        "silent_release_count": Observation(value=1, denominator=3, source="test")})
    assert outcome.status == FAIL
    assert outcome.hard[0]["passed"] is False


def test_a_missing_hard_count_is_never_a_pass():
    """An unreported safety number is not a zero — and not a failure either.

    Which of the two it is matters: "we did not look" and "we looked and it
    broke" call for different actions, and only the second is evidence about
    the system.
    """
    outcome = evaluate_gate(_gate(), [_row("s1", "mismatch", "mismatch")],
                            evidence={})
    assert outcome.hard[0]["passed"] is False
    assert outcome.status == NOT_ESTIMABLE


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
    outcome = evaluate_gate(_gate(), rows, evidence=_clean())
    assert outcome.status == FAIL
    assert outcome.capability[0]["point"] == pytest.approx(0.70)
    assert outcome.capability[0]["lower"] < 0.70


def test_a_clean_result_over_enough_studies_passes():
    outcome = evaluate_gate(_gate(), _many_studies([True] * 12),
                            evidence=_clean())
    assert outcome.status == PASS


def test_the_interval_is_deterministic():
    rows = _many_studies([True] * 8 + [False] * 2)
    first = evaluate_gate(_gate(), rows, evidence=_clean())
    second = evaluate_gate(_gate(), rows, evidence=_clean())
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
            [_row("s1", "mismatch", "mismatch")], evidence=_clean())


def test_a_gate_without_an_id_is_refused(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"status": "provisional"}), encoding="utf-8")
    with pytest.raises(ContractError, match="gate_id"):
        load_gate(path)


# --- D6-R0: an absence is not a zero, and order is not evidence ---

def _obs(value, denominator=None):
    return Observation(value=value, denominator=denominator, source="test")


def test_a_missing_safety_number_is_not_a_pass():
    """The first checker filled this with 0 and printed [ok]."""
    outcome = evaluate_gate(_gate(), [_row("s1", "mismatch", "mismatch")],
                            evidence={})
    assert outcome.hard[0]["state"] == MISSING
    assert outcome.status == NOT_ESTIMABLE
    assert any("no evidence was supplied" in b for b in outcome.blocking)


def test_zero_errors_over_zero_graded_rows_is_not_a_safety_record():
    gate = _gate(hard_gates=[HardGate(metric="wrong_target_released_count",
                                      must_equal=0,
                                      evidence_denominator="graded")])
    outcome = evaluate_gate(
        gate, [_row("s1", "match", "match")],
        evidence={"wrong_target_released_count": _obs(0), "graded": _obs(0)})
    assert outcome.hard[0]["state"] == NO_DENOMINATOR
    assert outcome.status == NOT_ESTIMABLE


def test_the_same_zero_counts_once_something_was_graded():
    gate = _gate(hard_gates=[HardGate(metric="wrong_target_released_count",
                                      must_equal=0,
                                      evidence_denominator="graded")])
    outcome = evaluate_gate(
        gate, [_row("s1", "match", "match")],
        evidence={"wrong_target_released_count": _obs(0), "graded": _obs(40)})
    assert outcome.hard[0]["state"] == OK


def test_an_observed_failure_outranks_missing_evidence():
    gate = _gate(hard_gates=[HardGate(metric="silent_release_count", must_equal=0),
                             HardGate(metric="never_reported", must_equal=0)])
    outcome = evaluate_gate(gate, [_row("s1", "mismatch", "match")],
                            evidence={"silent_release_count": _obs(2)})
    assert [c["state"] for c in outcome.hard] == [FAILED, MISSING]
    assert outcome.status == FAIL


def test_the_interval_does_not_depend_on_the_order_reports_arrive_in():
    """0.444 one way round and 0.500 the other, either side of the bar."""
    first = [_row("s2", "mismatch", "mismatch"), _row("s2", "match", "mismatch")]
    second = [_row(f"s{i}", "mismatch", "mismatch" if i % 2 else "match")
              for i in range(10)]
    forwards = evaluate_gate(_gate(), first + second,
                             evidence={"silent_release_count": _obs(0)})
    backwards = evaluate_gate(_gate(), second + first,
                              evidence={"silent_release_count": _obs(0)})
    assert forwards.capability[0]["lower"] == backwards.capability[0]["lower"]
    assert forwards.capability[0]["upper"] == backwards.capability[0]["upper"]
    assert forwards.status == backwards.status


def test_row_minimums_are_checked_per_domain_not_in_total():
    """"150 rows and 50 rows" must not pass a 100-per-domain requirement."""
    rows = ([_row(f"big{i}", "match", "match") for i in range(150)]
            + [_row(f"small{i}", "match", "match") for i in range(50)])
    domains = {f"big{i}": "eat" for i in range(150)}
    domains.update({f"small{i}": "melanoma" for i in range(50)})
    outcome = evaluate_gate(
        _gate(sample=SampleRequirement(min_domains=2, min_studies_per_domain=1,
                                       min_rows_per_domain=100,
                                       min_true_discrepancies=0)),
        rows, evidence={"silent_release_count": _obs(0)}, domains=domains)
    assert any("'melanoma' has 50 row(s)" in b for b in outcome.blocking)
    assert not any("'eat'" in b for b in outcome.blocking)


# --- D6-R1: the estimand is domain-weighted, and held-out stays out ---

def _domain_rows(prefix, n_studies, caught):
    rows = []
    for i in range(n_studies):
        rows.append(_row(f"{prefix}{i}", "mismatch",
                         "mismatch" if caught else "match"))
    return rows


def test_a_large_domain_does_not_speak_for_a_small_one():
    """Nine studies at 100% beside one at 0% is not "90% overall"."""
    rows = _domain_rows("eat", 9, True) + _domain_rows("mel", 1, False)
    domains = {f"eat{i}": "eat" for i in range(9)}
    domains.update({"mel0": "melanoma"})
    outcome = evaluate_gate(_gate(), rows, evidence=_clean(), domains=domains)
    check = outcome.capability[0]
    assert check["point"] == pytest.approx(0.5)          # (1.0 + 0.0) / 2
    assert check["per_domain"] == {"eat": 1.0, "melanoma": 0.0}


def test_the_held_out_domain_is_reported_alone_and_not_averaged_in():
    rows = _domain_rows("a", 4, True) + _domain_rows("b", 4, True) + \
        _domain_rows("h", 4, False)
    domains = {f"a{i}": "a" for i in range(4)}
    domains.update({f"b{i}": "b" for i in range(4)})
    domains.update({f"h{i}": "held" for i in range(4)})
    outcome = evaluate_gate(_gate(), rows, evidence=_clean(), domains=domains,
                            held_out_domain="held")
    check = outcome.capability[0]
    assert check["point"] == pytest.approx(1.0)   # pooled = the two dev domains
    assert check["held_out"] == pytest.approx(0.0)


def test_a_held_out_domain_that_is_not_in_the_evidence_is_refused():
    """Passing a name is not holding a domain out."""
    outcome = evaluate_gate(_gate(), _domain_rows("a", 3, True),
                            evidence=_clean(),
                            domains={f"a{i}": "a" for i in range(3)},
                            held_out_domain="banana")
    assert any("not present in the evidence" in b for b in outcome.blocking)


def test_undefined_resamples_are_reported_not_silently_dropped():
    """An interval from the draws that happened to work is not an interval."""
    from react_review.acceptance import stratified_bootstrap

    # Only one study in the domain carries a discrepancy, so most resamples
    # contain none and recall is undefined in them.
    domain = [[_row("s0", "mismatch", "mismatch")]] + \
        [[_row(f"s{i}", "match", "match")] for i in range(1, 8)]
    estimate = stratified_bootstrap([domain], STATISTICS["discrepancy_recall"],
                                    resamples=200)
    assert estimate["undefined_rate"] > 0.05
    assert estimate["lower"] is None
    assert "happened to work" in estimate["note"]


def test_discrepancies_must_be_spread_across_domains_and_studies():
    """Twenty-five discrepancies in one paper is not twenty-five chances to fail."""
    rows = [_row("s0", "mismatch", "mismatch") for _ in range(25)]
    outcome = evaluate_gate(
        _gate(sample=SampleRequirement(
            min_domains=1, min_studies_per_domain=1, min_rows_per_domain=1,
            min_true_discrepancies=25, min_true_discrepancies_per_domain=8,
            min_studies_with_discrepancies=6)),
        rows, evidence=_clean(), domains={"s0": "one"})
    assert any("study/studies carry a true discrepancy" in b
               for b in outcome.blocking)


def test_retrieval_coverage_and_auditable_coverage_are_different_questions():
    from react_review.acceptance import STATISTICS as S

    found_but_wrong_scope = [{"study_id": "s", "expected_label": "match",
                              "predicted_label": "not_comparable", "found": True,
                              "scope_check": "scope_unresolved"}]
    assert S["retrieval_coverage"](found_but_wrong_scope) == 1.0
    assert S["auditable_coverage"](found_but_wrong_scope) == 0.0


def test_the_gate_names_both_coverages_and_reports_the_new_rates():
    gate = load_gate(GATE_FILE)
    metrics = {c.metric for c in gate.capability_gates}
    assert {"retrieval_coverage", "auditable_coverage"} <= metrics
    assert "source_coverage" not in metrics          # renamed, not silently kept
    assert {"scope_assessable_rate", "review_burden"} <= set(gate.reported_only)


# --- D6-R2: passing a gate is not authorisation ---

def test_a_provisional_gate_can_never_authorise_a_release():
    """Meeting a bar we invented ourselves is not fitness for use."""
    outcome = evaluate_gate(_gate(), _many_studies([True] * 12), evidence=_clean())
    assert outcome.status == PASS
    assert outcome.release_eligible is False
    assert any("provisional" in b for b in outcome.release_blockers)


def test_a_signed_off_gate_can():
    outcome = evaluate_gate(_gate(status="signed_off", signed_off_by="a clinician"),
                            _many_studies([True] * 12), evidence=_clean())
    assert outcome.status == PASS and outcome.release_eligible is True


def test_a_failed_evaluation_blocks_release_whatever_the_gate_status():
    outcome = evaluate_gate(_gate(status="signed_off"),
                            _many_studies([True] * 5 + [False] * 5),
                            evidence=_clean())
    assert outcome.status == FAIL and outcome.release_eligible is False


def test_the_shipped_register_holds_nothing_available():
    """The honest current statement: nothing here can serve as held-out."""
    import json

    register = json.loads(
        (repo_root() / "configs" / "gates" / "held_out_register.json")
        .read_text(encoding="utf-8"))
    assert register["available_for_holding_out"] == []
    assert all(entry["role"] == "development"
               for entry in register["domains"].values())
