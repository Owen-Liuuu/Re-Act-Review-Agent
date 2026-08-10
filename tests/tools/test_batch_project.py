"""Two stages, in that order — the point of the whole batch.

MA004 is the test that matters: 314 allocated to the combination arm and 313
analysed in it, and the review reporting the allocated one. Getting it right
needs the arm decided BEFORE the population, because feeding both readings into
the arm assignment as two arms leaves three review arms chasing four paper arms
and nothing resolves at all.
"""
from __future__ import annotations

import pytest

from react_review.normalize.cohorts import parse_comparison
from react_review.normalize.population import PopulationScope
from react_review.schemas.batch import ARM, COMPARISON, STUDY
from react_review.tools.batch_parse import parse_batch
from react_review.tools.batch_project import (
    AMBIGUOUS,
    BATCH_FAILED,
    CONTRADICTORY,
    NOT_REPORTED,
    OK,
    SCOPE_UNRESOLVED,
    TIMEPOINT_UNRESOLVED,
    UNSUPPORTED,
    project_claim,
)

PAPER = (
    "A total of 945 patients underwent randomization: 316 patients were assigned "
    "to the nivolumab group, 314 to the nivolumab-plus-ipilimumab group, and 315 "
    "to the ipilimumab group.\n\n"
    "Efficacy The median progression-free survival was 6.9 months (95% CI, 4.3 to "
    "9.5) in the nivolumab group, 11.5 months (95% CI, 8.9 to 16.7) in the "
    "nivolumab-plus-ipilimumab group, and 2.9 months (95% CI, 2.8 to 3.4) in the "
    "ipilimumab group. The hazard ratio for the comparison between the "
    "nivolumab-plus-ipilimumab group and the nivolumab group was 0.74 (95% CI, "
    "0.60 to 0.92).\n\n"
    "Table 3. Analysis population. Nivolumab plus Ipilimumab (N = 313)")

ALLOCATION = ("A total of 945 patients underwent randomization: 316 patients were "
              "assigned to the nivolumab group, 314 to the nivolumab-plus-ipilimumab "
              "group, and 315 to the ipilimumab group.")

REVIEW = {"nivolumab_plus_ipilimumab": "Nivolumab (1 mg/kg) + ipilimumab (3 mg/kg)",
          "ipilimumab_plus_placebo": "Ipilimumab (3 mg/kg) + placebo",
          "nivolumab_plus_placebo": "Nivolumab (3 mg/kg) + placebo"}

ALLOCATED = PopulationScope(basis="allocated")


def _counts_batch():
    """Every arm's allocated count, plus the combination arm's analysed count."""
    return parse_batch({"readings": [
        {"arm_label": "nivolumab group", "value": "316", "quote": ALLOCATION,
         "population_phrase": "underwent randomization"},
        {"arm_label": "nivolumab-plus-ipilimumab group", "value": "314",
         "quote": ALLOCATION, "population_phrase": "underwent randomization"},
        {"arm_label": "ipilimumab group", "value": "315", "quote": ALLOCATION,
         "population_phrase": "underwent randomization"},
        {"arm_label": "Nivolumab plus Ipilimumab", "value": "313",
         "quote": "Nivolumab plus Ipilimumab (N = 313)",
         "population_phrase": "Analysis population"},
    ]}, PAPER)


# --- the case the phase exists for ---------------------------------------

def test_ma004_takes_the_allocated_reading_not_the_analysed_one():
    projection = project_claim(
        _counts_batch(), review_labels=REVIEW,
        cohort_key="nivolumab_plus_ipilimumab", requested_scope=ALLOCATED,
        required_axes=["population_basis"])
    assert projection.status == OK
    assert projection.value == "314"
    assert projection.entry.identity.population.basis == "allocated"


def test_both_readings_of_that_arm_reach_stage_two():
    """Four readings, three arms: the combination arm carries two of them."""
    projection = project_claim(
        _counts_batch(), review_labels=REVIEW,
        cohort_key="nivolumab_plus_ipilimumab", requested_scope=ALLOCATED,
        required_axes=["population_basis"])
    assert sorted(e.value for e in projection.candidates) == ["313", "314"]
    assert "one-to-one arm assignment" in projection.provenance["stage_one"]


def test_the_arm_assignment_still_sees_three_arms_not_four():
    """Folding the readings is what keeps the assignment solvable."""
    projection = project_claim(
        _counts_batch(), review_labels=REVIEW, cohort_key="nivolumab_plus_placebo",
        requested_scope=ALLOCATED, required_axes=["population_basis"])
    assert projection.status == OK and projection.value == "316"
    assert len(projection.provenance["papers_arms"]) == 3


def test_asking_for_the_analysed_population_gets_the_other_reading():
    projection = project_claim(
        _counts_batch(), review_labels=REVIEW,
        cohort_key="nivolumab_plus_ipilimumab",
        requested_scope=PopulationScope(basis="analysed"),
        required_axes=["population_basis"])
    assert projection.status == OK and projection.value == "313"


def test_a_population_the_batch_never_reports_is_refused_not_approximated():
    projection = project_claim(
        _counts_batch(), review_labels=REVIEW,
        cohort_key="nivolumab_plus_ipilimumab",
        requested_scope=PopulationScope(basis="treated"),
        required_axes=["population_basis"])
    assert projection.status == SCOPE_UNRESOLVED
    assert projection.value is None
    assert "none of its 2 reading(s)" in projection.reason


# --- stage one refusals ---------------------------------------------------

def test_an_arm_the_batch_never_reported_is_not_answered_by_another():
    reading = parse_batch({"readings": [
        {"arm_label": "nivolumab group", "value": "316", "quote": ALLOCATION},
    ]}, PAPER)
    projection = project_claim(
        reading, review_labels=REVIEW, cohort_key="ipilimumab_plus_placebo",
        requested_scope=ALLOCATED, required_axes=["population_basis"])
    assert projection.status in (NOT_REPORTED, AMBIGUOUS)
    assert projection.value is None


def test_a_failed_batch_fails_the_claim_with_the_batch_s_reason():
    projection = project_claim(parse_batch({"nope": 1}, PAPER), review_labels=REVIEW,
                               cohort_key="nivolumab_plus_ipilimumab")
    assert projection.status == BATCH_FAILED
    assert "no `readings` list" in projection.reason


def test_an_empty_batch_reports_the_model_s_own_reason():
    reading = parse_batch({"readings": [],
                           "nothing_reported_reason": "the field is not in this paper"},
                          PAPER)
    projection = project_claim(reading, review_labels=REVIEW,
                               cohort_key="nivolumab_plus_ipilimumab")
    assert projection.status == NOT_REPORTED
    assert projection.reason == "the field is not in this paper"


# --- comparisons ----------------------------------------------------------

def _hr_batch():
    quote = ("The hazard ratio for the comparison between the "
             "nivolumab-plus-ipilimumab group and the nivolumab group was 0.74 "
             "(95% CI, 0.60 to 0.92).")
    return parse_batch({"readings": [
        {"left_label": "nivolumab-plus-ipilimumab group",
         "right_label": "nivolumab group", "value": "0.74 (95% CI, 0.60 to 0.92)",
         "quote": quote},
    ]}, PAPER, target_shape=COMPARISON)


def test_a_comparison_is_matched_as_a_pair():
    projection = project_claim(
        _hr_batch(), target_shape=COMPARISON, review_labels=REVIEW,
        comparison=parse_comparison("nivolumab_plus_ipilimumab_vs_nivolumab_plus_placebo"))
    assert projection.status == OK and projection.value.startswith("0.74")


def test_the_mirror_image_comparison_is_refused():
    projection = project_claim(
        _hr_batch(), target_shape=COMPARISON, review_labels=REVIEW,
        comparison=parse_comparison("nivolumab_plus_placebo_vs_nivolumab_plus_ipilimumab"))
    assert projection.status == UNSUPPORTED
    assert "other way round" in projection.reason


# --- the whole-study total, without arithmetic ---------------------------

def test_ma002_takes_the_total_the_paper_prints():
    reading = parse_batch({"readings": [
        {"scope_label": "all randomised patients", "value": "945",
         "quote": "A total of 945 patients underwent randomization",
         "population_phrase": "underwent randomization"},
    ]}, PAPER, target_shape=STUDY)
    projection = project_claim(reading, target_shape=STUDY, requested_scope=ALLOCATED,
                               required_axes=["population_basis"])
    assert projection.status == OK and projection.value == "945"
    assert projection.entry.identity.population.basis == "allocated"


def test_a_total_the_paper_does_not_print_is_not_derived_from_the_arms():
    """Arithmetic is not reading. An unprinted total stays unresolved."""
    projection = project_claim(_counts_batch(), target_shape=STUDY,
                               requested_scope=ALLOCATED,
                               required_axes=["population_basis"])
    assert projection.status != OK
    assert projection.value is None


# --- contradictions are surfaced, not resolved ---------------------------

def test_two_readings_of_the_same_thing_that_disagree_are_surfaced():
    reading = parse_batch({"readings": [
        {"arm_label": "nivolumab group", "value": "316", "quote": ALLOCATION,
         "population_phrase": "underwent randomization"},
        {"arm_label": "nivolumab group", "value": "315", "quote": ALLOCATION,
         "population_phrase": "underwent randomization"},
    ]}, PAPER)
    projection = project_claim(
        reading, review_labels={"nivolumab_plus_placebo": REVIEW["nivolumab_plus_placebo"]},
        cohort_key="nivolumab_plus_placebo", requested_scope=ALLOCATED,
        required_axes=["population_basis"])
    assert projection.status == CONTRADICTORY
    assert projection.value is None
    assert "315" in projection.reason and "316" in projection.reason


def test_two_readings_that_agree_are_not_a_contradiction():
    reading = parse_batch({"readings": [
        {"arm_label": "nivolumab group", "value": "316", "quote": ALLOCATION,
         "population_phrase": "underwent randomization"},
        {"arm_label": "nivolumab group", "value": "316",
         "quote": "316 patients were assigned to the nivolumab group",
         "population_phrase": "were assigned to"},
    ]}, PAPER)
    projection = project_claim(
        reading, review_labels={"nivolumab_plus_placebo": REVIEW["nivolumab_plus_placebo"]},
        cohort_key="nivolumab_plus_placebo", requested_scope=ALLOCATED,
        required_axes=["population_basis"])
    assert projection.status == OK and projection.value == "316"


# --- timepoint ------------------------------------------------------------

def _two_timepoints():
    pfs = ("The median progression-free survival was 6.9 months (95% CI, 4.3 to "
           "9.5) in the nivolumab group")
    return parse_batch({"readings": [
        {"arm_label": "nivolumab group", "value": "6.9 months (95% CI, 4.3 to 9.5)",
         "quote": pfs, "timepoint_phrase": "median progression-free survival"},
        {"arm_label": "nivolumab group", "value": "2.9 months (95% CI, 2.8 to 3.4)",
         "quote": "2.9 months (95% CI, 2.8 to 3.4) in the ipilimumab group",
         "timepoint_phrase": "median overall survival"},
    ]}, PAPER)


def test_a_timepoint_decides_between_readings_that_offer_a_choice():
    projection = project_claim(
        _two_timepoints(),
        review_labels={"nivolumab_plus_placebo": REVIEW["nivolumab_plus_placebo"]},
        cohort_key="nivolumab_plus_placebo",
        timepoint_label="median progression-free survival")
    assert projection.status == OK and projection.value.startswith("6.9")


def test_a_lone_reading_is_answered_and_its_timepoint_marked_unverified():
    """Confirming "median PFS" is "median progression-free survival" needs an
    abbreviation table. Until there is one the claim is answered and the gap is
    recorded — neither hidden, nor invented as a rejection."""
    quote = ("The median progression-free survival was 6.9 months (95% CI, 4.3 to "
             "9.5) in the nivolumab group")
    reading = parse_batch({"readings": [
        {"arm_label": "nivolumab group", "value": "6.9 months (95% CI, 4.3 to 9.5)",
         "quote": quote, "timepoint_phrase": "median progression-free survival"},
    ]}, PAPER)
    projection = project_claim(
        reading, review_labels={"nivolumab_plus_placebo": REVIEW["nivolumab_plus_placebo"]},
        cohort_key="nivolumab_plus_placebo", timepoint_label="median PFS")
    assert projection.status == OK
    assert any("timepoint not verified" in line
               for line in projection.provenance["stage_two"])
