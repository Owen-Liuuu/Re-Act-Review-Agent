"""Which population a number counts — and the refusal to guess.

MA004 is the case these tests are built around. The review reports the 314
patients allocated to an arm; the extractor returned 313 from a table of an
analysis population; a 1% band called it a match. Both quotes below are the real
ones from the frozen Phase 7 recording.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.audit.scope import (
    MISMATCH,
    NOT_REQUIRED,
    OK,
    UNRESOLVED,
    scope_rates,
    scope_verdict,
)
from react_review.contracts import ContractError, repo_root
from react_review.core.enums import AuditLabel
from react_review.normalize.population import (
    PopulationScope,
    classify_population,
    load_population_contract,
)

ALLOCATION = ("A total of 945 patients underwent randomization: 316 patients were "
              "assigned to the nivolumab group, 314 to the nivolumab-plus-"
              "ipilimumab group, and 315 to the ipilimumab group.")
TABLE_CELL = "Nivolumab plus Ipilimumab (N = 313)"


# --- classification -------------------------------------------------------

def test_the_allocation_sentence_is_read_as_allocated():
    scope = classify_population(ALLOCATION)
    assert scope.basis == "allocated"
    assert scope.randomisation_stated is True       # it says so, in this sentence
    assert scope.analysis_set == "unspecified"
    assert scope.basis_phrase                        # the words that decided it


def test_the_table_cell_states_no_population_at_all():
    """MA004's evidence: a count with no population words is UNKNOWN."""
    scope = classify_population(TABLE_CELL)
    assert scope.basis == "unknown"
    assert scope.stated is False
    assert scope.randomisation_stated is False


def test_assignment_alone_is_not_randomisation():
    """"were assigned to" establishes allocation, never a random one."""
    scope = classify_population("315 patients were assigned to the ipilimumab group")
    assert scope.basis == "allocated"
    assert scope.randomisation_stated is False


@pytest.mark.parametrize("text,basis", [
    ("patients who received at least one dose of study drug", "treated"),
    ("all patients included in the analysis of the primary endpoint", "analysed"),
    ("evaluable for response at week 12", "analysed"),
    ("baseline characteristics of the study population", "unknown"),
])
def test_the_basis_is_read_from_methodology_words_only(text, basis):
    assert classify_population(text).basis == basis


@pytest.mark.parametrize("text,analysis_set", [
    ("the intention-to-treat population", "itt"),
    ("the modified intention-to-treat population", "mitt"),
    ("the per-protocol analysis", "per_protocol"),
    ("the safety population", "safety"),
    ("patients in the trial", "unspecified"),
])
def test_the_analysis_set_is_a_separate_axis(text, analysis_set):
    assert classify_population(text).analysis_set == analysis_set


def test_the_longest_phrase_decides():
    """"modified intention-to-treat" must not be read as plain ITT."""
    scope = classify_population("the modified intention-to-treat population")
    assert scope.analysis_set == "mitt"


def test_the_two_axes_do_not_collapse_into_each_other():
    """A safety population is treated AND an analysis set; both are recorded."""
    scope = classify_population(
        "the safety population, comprising all patients who received at least "
        "one dose")
    assert scope.basis == "treated" and scope.analysis_set == "safety"


def test_no_disease_or_drug_word_appears_in_the_contract():
    body = (repo_root() / "configs" / "population_roles.json").read_text(
        encoding="utf-8-sig").lower()
    for word in ("nivolumab", "ipilimumab", "melanoma", "diabetes", "epicardial"):
        assert word not in body


def test_an_unknown_axis_value_in_the_contract_is_refused(tmp_path):
    path = tmp_path / "roles.json"
    path.write_text(json.dumps({"population_basis": {"enrolled": ["enrolled"]}}),
                    encoding="utf-8")
    with pytest.raises(ContractError, match="unknown population_basis"):
        load_population_contract(path)


# --- the verdict ----------------------------------------------------------

def _scope(basis="unknown", analysis_set="unspecified") -> PopulationScope:
    return PopulationScope(basis=basis, analysis_set=analysis_set)


def test_ma004_is_unresolved_rather_than_a_0_3_percent_match():
    review = classify_population(ALLOCATION)
    source = classify_population(TABLE_CELL)
    outcome = scope_verdict(review, source, required_axes=["population_basis"])
    assert outcome.status == UNRESOLVED
    assert outcome.blocks_comparison and not outcome.assessable
    assert "not stated on the source side" in outcome.reason


def test_two_populations_that_are_both_stated_and_differ_are_a_mismatch():
    outcome = scope_verdict(_scope("allocated"), _scope("analysed"),
                            required_axes=["population_basis"])
    assert outcome.status == MISMATCH
    assert "do not count the same people" in outcome.reason


def test_the_same_population_on_both_sides_passes():
    outcome = scope_verdict(_scope("allocated"), _scope("allocated"),
                            required_axes=["population_basis"])
    assert outcome.status == OK and outcome.blocks_comparison is False


def test_a_field_that_requires_no_axis_is_not_gated():
    outcome = scope_verdict(_scope(), _scope(), required_axes=[])
    assert outcome.status == NOT_REQUIRED and outcome.blocks_comparison is False


def test_an_unrequired_axis_left_unstated_does_not_block():
    """The basis is settled; nobody asked about the analysis set."""
    outcome = scope_verdict(_scope("allocated"), _scope("allocated"),
                            required_axes=["population_basis"])
    assert outcome.status == OK


def test_an_unrequired_axis_that_both_sides_state_in_conflict_does_block():
    """A stated contradiction is evidence, whatever the contract required."""
    outcome = scope_verdict(_scope("analysed", "per_protocol"),
                            _scope("analysed", "safety"),
                            required_axes=["population_basis"])
    assert outcome.status == MISMATCH
    assert "analysis set" in outcome.reason


def test_both_axes_are_checked_when_the_field_requires_both():
    outcome = scope_verdict(_scope("analysed", "itt"), _scope("analysed"),
                            required_axes=["population_basis", "analysis_set"])
    assert outcome.status == UNRESOLVED
    assert outcome.unresolved_axes == ["analysis_set"]


# --- the anti-gaming rates ------------------------------------------------

def test_refusing_everything_cannot_look_like_success():
    outcomes = [scope_verdict(_scope(), _scope(), required_axes=["population_basis"])
                for _ in range(4)]
    rates = scope_rates(outcomes)
    assert rates["scope_required"] == 4
    assert rates["scope_assessable_rate"] == 0.0     # nothing could be judged
    assert rates["scope_resolved_rate"] == 0.0
    assert rates["scope_unresolved"] == 4


def test_the_rates_separate_could_not_judge_from_judged_and_differed():
    outcomes = [
        scope_verdict(_scope("allocated"), _scope("allocated"),
                      required_axes=["population_basis"]),      # ok
        scope_verdict(_scope("allocated"), _scope("analysed"),
                      required_axes=["population_basis"]),      # mismatch
        scope_verdict(_scope("allocated"), _scope(),
                      required_axes=["population_basis"]),      # unresolved
        scope_verdict(_scope(), _scope(), required_axes=[]),    # not gated
    ]
    rates = scope_rates(outcomes)
    assert rates["scope_required"] == 3
    assert rates["scope_assessable_rate"] == pytest.approx(2 / 3)
    assert rates["scope_resolved_rate"] == pytest.approx(1 / 3)
    assert rates["scope_mismatch"] == 1


# --- the scope check as the comparator applies it (P8-0 U8) ---

def test_the_ma004_shape_is_refused_before_the_arithmetic():
    """313 analysed against 314 allocated is not a 0.3% difference."""
    from react_review.audit import compare_values

    result = compare_values(
        field_type="cohort_n", review_value="314", source_value="313",
        review_scope={"basis": "allocated"}, source_scope={"basis": "unknown"},
        required_scope_axes=["population_basis"])
    assert result.label is AuditLabel.NOT_COMPARABLE
    assert result.scope_check == "scope_unresolved"
    assert result.review_required is True


def test_a_field_with_no_required_axis_is_untouched():
    from react_review.audit import compare_values

    result = compare_values(
        field_type="hazard_ratio", review_value="0.42", source_value="0.42",
        review_scope={"basis": "allocated"}, source_scope={"basis": "unknown"},
        required_scope_axes=[])
    assert result.label is AuditLabel.MATCH
    assert result.scope_check == "not_required"


def test_a_stated_contradiction_blocks_even_an_axis_nobody_required():
    """An axis both sides state, in conflict, is evidence — not an omission."""
    from react_review.audit import compare_values

    result = compare_values(
        field_type="cohort_n", review_value="314", source_value="314",
        review_scope={"basis": "allocated"}, source_scope={"basis": "analysed"},
        required_scope_axes=[])
    assert result.label is AuditLabel.NOT_COMPARABLE
    assert result.scope_check == "scope_mismatch"


def test_scope_and_exact_counts_only_bite_together():
    """Exactness alone would trade a silent wrong match for a loud wrong one."""
    from react_review.audit import compare_values

    # scope agreed → the exact band decides, and 313 != 314
    result = compare_values(
        field_type="cohort_n", review_value="314", source_value="313",
        rel_tolerance=0.0,
        review_scope={"basis": "allocated"}, source_scope={"basis": "allocated"},
        required_scope_axes=["population_basis"])
    assert result.label is AuditLabel.MISMATCH
    assert result.scope_check == "ok"
