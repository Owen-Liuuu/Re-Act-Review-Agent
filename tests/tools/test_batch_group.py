"""Which claims share a prompt, and what sharing one is called.

The grouping decides the cost of a run and, more importantly, decides what the
model is asked. Two claims that do not put the same question to the same paper
must not share a reading — and two that do must not be asked twice, or the
batch buys nothing.
"""
from __future__ import annotations

from react_review.schemas.batch import ARM, COMPARISON, STUDY
from react_review.schemas.evidence import ReviewDataItem
from react_review.tools.batch_group import (
    claim_kind,
    group_claims,
    group_key_for,
    plan_batches,
    target_shape,
)


def _claim(**kw) -> ReviewDataItem:
    body = dict(study_id="larkin", group="nivolumab_plus_placebo",
                field_type="cohort_n", raw_field_name="Intervention arm, n",
                column_header="N", timepoint="single", unit="count",
                cohort_label="Nivolumab (3 mg/kg) + placebo", value="316")
    body.update(kw)
    return ReviewDataItem(**body)


# --- what kind of question is this ----------------------------------------

def test_a_cell_repeating_its_own_cohort_label_asks_what_the_arm_IS():
    """Structural: the cell's value is the review's own name for that arm."""
    item = _claim(value="Nivolumab (3 mg/kg) + placebo")
    assert claim_kind(item) == "arm_identity"
    assert claim_kind(_claim(value="316")) == "value"


def test_the_shape_comes_from_the_reviews_own_cohort_field():
    assert target_shape(_claim()) == ARM
    assert target_shape(_claim(group="all")) == STUDY
    assert target_shape(_claim(group="-")) == STUDY
    assert target_shape(_claim(group="a_vs_b")) == COMPARISON


# --- what may share a prompt ----------------------------------------------

def test_two_arms_of_one_field_share_a_reading():
    groups = group_claims([_claim(group="nivolumab_plus_placebo"),
                           _claim(group="ipilimumab_plus_placebo")])
    assert len(groups) == 1 and len(groups[0].claims) == 2


def test_a_different_column_wording_is_a_different_question():
    """`raw_field_name` reaches the prompt, so it cannot be folded away."""
    groups = group_claims([_claim(raw_field_name="Intervention arm, n"),
                           _claim(raw_field_name="N randomised")])
    assert len(groups) == 2


def test_a_header_that_never_reaches_the_prompt_does_not_split_a_group():
    """Splitting on it would ask one paper the same thing twice because a
    header was punctuated differently."""
    groups = group_claims([_claim(column_header="N"),
                           _claim(column_header="N."),])
    assert len(groups) == 1


def test_arms_comparisons_and_totals_are_asked_separately():
    groups = group_claims([_claim(group="a"), _claim(group="a_vs_b"),
                           _claim(group="all")])
    assert {g.shape for g in groups} == {ARM, COMPARISON, STUDY}


def test_asking_what_an_arm_is_and_what_it_reports_are_two_questions():
    groups = group_claims([_claim(value="316"),
                           _claim(value="Nivolumab (3 mg/kg) + placebo")])
    assert len(groups) == 2
    assert {g.kind for g in groups} == {"value", "arm_identity"}


def test_a_different_timepoint_is_a_different_question():
    groups = group_claims([_claim(timepoint="baseline"),
                           _claim(timepoint="median_pfs")])
    assert len(groups) == 2


def test_two_studies_never_share_a_reading():
    groups = group_claims([_claim(study_id="a"), _claim(study_id="b")])
    assert len(groups) == 2


def test_claims_keep_the_order_they_arrived_in():
    """A preflight and a run must enumerate the same batches in the same order."""
    items = [_claim(group=f"arm_{i}") for i in range(5)]
    groups = group_claims(items)
    assert [c.group for c in groups[0].claims] == [i.group for i in items]


def test_the_key_says_what_it_is_about():
    key = group_key_for(_claim(group="all"))
    assert "larkin/cohort_n" in key.describe() and STUDY in key.describe()


# --- the cost argument, checkable before it is paid ------------------------

def test_a_plan_reports_what_a_run_would_ask_without_asking():
    plan = plan_batches([_claim(group="a"), _claim(group="b"),
                         _claim(group="all"),
                         _claim(value="Nivolumab (3 mg/kg) + placebo")])
    assert plan.claim_count == 4
    assert plan.batch_count == 3           # two arms together, total, identity
    assert plan.singletons == 2
    assert round(plan.calls_saved, 2) == 1.33
    assert plan.by_kind() == {"value": 3, "arm_identity": 1}


def test_a_plan_of_singletons_says_the_batch_bought_nothing():
    plan = plan_batches([_claim(study_id=f"s{i}") for i in range(4)])
    assert plan.batch_count == plan.claim_count == 4
    assert plan.calls_saved == 1.0
    assert plan.singletons == 4


def test_an_empty_plan_does_not_divide_by_zero():
    plan = plan_batches([])
    assert plan.batch_count == 0 and plan.calls_saved == 0.0
