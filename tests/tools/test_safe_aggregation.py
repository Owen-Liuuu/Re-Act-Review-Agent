"""When the arms may be added up, and — mostly — when they may not.

MA002 is the case: the paper prints 316, 314 and 315 and never prints 945, and
the review reports 945. The number is recoverable, but only because the paper
also says it randomised patients to one of THREE groups. Take that sentence away
and the same three numbers are just three numbers whose sum happens to look like
a total.

Most of what follows is that sentence being taken away, one way at a time. The
adversarial cases are the ones where something plausible arrives and must still
be refused: a boolean spelled as a string, a partition claim about the wrong
population, two printed totals that disagree with each other and a sum that
would quietly settle the argument.
"""
from __future__ import annotations

import pytest

from react_review.normalize.population import PopulationScope
from react_review.schemas.batch import STUDY
from react_review.tools.batch_parse import parse_batch
from react_review.tools.batch_project import (
    AMBIGUOUS,
    CONTRADICTORY,
    DERIVED,
    NOT_REPORTED,
    OK,
    SCOPE_UNRESOLVED,
    project_claim,
)
from react_review.tools.batch_result import to_source_result
from react_review.tools.safe_aggregation import (
    NOT_APPLICABLE,
    PROTOCOL_ERROR,
    REJECTED,
    load_aggregation_policy,
)

ALLOCATION = ("A total of 945 patients underwent randomization: 316 patients were "
              "assigned to the nivolumab group, 314 to the nivolumab-plus-ipilimumab "
              "group, and 315 to the ipilimumab group.")

PARTITION = ("Patients were randomly assigned in a 1:1:1 ratio to one of three "
             "groups, and no patient received more than one assigned regimen.")

ANALYSIS_ROW = ("Table 3. Analysis population. Nivolumab (N = 311), Nivolumab plus "
                "Ipilimumab (N = 313), Ipilimumab (N = 312)")

ANALYSIS_PARTITION = ("The analysis population comprised the three treatment "
                      "groups, each patient counted once.")

PAPER = "\n\n".join([ALLOCATION, PARTITION, ANALYSIS_ROW, ANALYSIS_PARTITION])

ALLOCATED = PopulationScope(basis="allocated")
ANALYSED = PopulationScope(basis="analysed")
#: What a run contract declares for sample_size. The analysis-set axis is not
#: required here; it is still COMPARED, and a stated conflict on it still blocks
#: (see test_t6/test_t7), which is the distinction scope_verdict draws.
AXES = ["population_basis"]


# --- building responses ---------------------------------------------------

def _count(label, n, quote=ALLOCATION):
    return {"arm_label": label, "count": n, "quote": quote}


ALLOCATED_ARMS = [_count("nivolumab group", 316),
                  _count("nivolumab-plus-ipilimumab group", 314),
                  _count("ipilimumab group", 315)]

ANALYSED_ARMS = [_count("Nivolumab", 311, ANALYSIS_ROW),
                 _count("Nivolumab plus Ipilimumab", 313, ANALYSIS_ROW),
                 _count("Ipilimumab", 312, ANALYSIS_ROW)]

GOOD_PARTITION = {"complete": True, "mutually_exclusive": True,
                  "quote": PARTITION, "declared_arm_count": 3,
                  "reason": "randomised 1:1:1 to one of three groups"}


def _set(*, counts=None, partition=None, phrase="underwent randomization",
         witness=ALLOCATION, timepoint=None, timepoint_quote=None,
         population_type="allocated", **extra):
    body = {"population_type": population_type, "population_phrase": phrase,
            "population_quote": witness,
            "cohort_counts": list(ALLOCATED_ARMS if counts is None else counts),
            "partition": dict(GOOD_PARTITION if partition is None else partition)}
    if timepoint:
        body["timepoint_phrase"] = timepoint
        body["timepoint_quote"] = timepoint_quote or witness
    body.update(extra)
    return body


ANALYSED_SET = _set(counts=ANALYSED_ARMS, phrase="Analysis population",
                    witness=ANALYSIS_ROW, population_type="analysed",
                    partition={"complete": True, "mutually_exclusive": True,
                               "quote": ANALYSIS_PARTITION,
                               "declared_arm_count": 3,
                               "reason": "three groups, each counted once"})


def _batch(*, readings=(), sets=None, document=PAPER):
    body = {"readings": list(readings)}
    if sets is not None:
        body["aggregation_sets"] = sets
    return parse_batch(body, document, target_shape=STUDY, aggregable=True)


def _project(reading, scope=ALLOCATED, field_type="sample_size", **kw):
    return project_claim(reading, target_shape=STUDY, requested_scope=scope,
                         required_axes=AXES, field_type=field_type, **kw)


def _total(value, quote, phrase="underwent randomization"):
    return {"scope_label": "all randomised patients", "value": value,
            "quote": quote, "population_phrase": phrase}


# --- the case this exists for ---------------------------------------------

def test_ma002_is_derived_from_three_anchored_arms():
    projection = _project(_batch(sets=[_set()]))
    assert projection.status == DERIVED
    assert projection.derived_value == 945
    assert projection.derivation == "316 + 314 + 315 = 945"
    assert projection.verified_scope.basis == "allocated"


def test_the_derived_total_reaches_the_evidence_object_as_derived():
    result = to_source_result(_project(_batch(sets=[_set()])))
    assert result.found and result.value == "945"
    assert result.value_origin == "derived_sum"
    assert result.aggregation_status == "derived"
    assert result.derivation == "316 + 314 + 315 = 945"
    assert [c.count for c in result.cohort_counts] == [316, 314, 315]
    assert result.source_scope.basis == "allocated"


def test_a_derived_total_never_carries_a_quote_that_prints_it():
    """No passage of the paper states a number the paper never computed."""
    result = to_source_result(_project(_batch(sets=[_set()])))
    assert result.quote == PARTITION and "945" not in result.quote
    assert all(c.quote for c in result.cohort_counts)


def test_the_policy_that_permitted_it_is_recorded_with_its_hash():
    projection = _project(_batch(sets=[_set()]))
    assert projection.policy_id == "safe_sum_v5"
    assert len(projection.policy_sha256) == 64


# --- T1: a boolean is a boolean -------------------------------------------

def test_t1_the_string_false_is_not_a_boolean():
    """`bool("false")` is True. Coercing here would inverate the model's answer."""
    reading = _batch(sets=[_set(partition={**GOOD_PARTITION, "complete": "false",
                                           "mutually_exclusive": "false"})])
    assert reading.aggregation_sets == []
    assert any("only a JSON true or false" in e for e in reading.aggregation_errors)
    projection = _project(reading)
    assert projection.status != DERIVED
    assert projection.aggregation_status == PROTOCOL_ERROR


def test_t1_the_string_true_is_not_a_boolean_either():
    reading = _batch(sets=[_set(partition={**GOOD_PARTITION, "complete": "true"})])
    assert reading.aggregation_sets == []
    assert _project(reading).aggregation_status == PROTOCOL_ERROR


def test_t1_one_and_zero_are_not_booleans():
    for value in (1, 0, None):
        reading = _batch(sets=[_set(partition={**GOOD_PARTITION,
                                               "mutually_exclusive": value})])
        assert reading.aggregation_sets == [], value
        assert _project(reading).aggregation_status == PROTOCOL_ERROR


# --- T2: a sum may fill a silence, never settle an argument ---------------

def test_t2_two_conflicting_printed_totals_are_not_resolved_by_the_arms():
    """The failure this whole priority order exists for.

    The paper prints 945 in one place and 944 in another, and the arms add to
    945. Returning 945 would look like corroboration and would in fact be a
    decision to disbelieve one of the paper's own sentences — silently, in the
    one place an audit was supposed to raise its hand.
    """
    document = PAPER + "\n\nIn total, 944 patients were enrolled and underwent randomization."
    reading = _batch(readings=[
        _total("945", "A total of 945 patients underwent randomization"),
        _total("944", "In total, 944 patients were enrolled and underwent randomization")],
        sets=[_set()], document=document)
    projection = _project(reading)
    assert projection.status == CONTRADICTORY
    assert projection.derived_value is None and projection.value is None
    assert "944" in projection.reason and "945" in projection.reason
    assert "cannot decide which of the paper" in projection.reason
    assert to_source_result(projection).found is False


def test_t2_the_no_derive_guard_covers_ambiguity_as_well_as_contradiction():
    """Ambiguity cannot arise for a study total today; the guard is deliberate.

    Stage one has no arm to be ambiguous about here, so only CONTRADICTORY is
    reachable. The guard names both because the reason is the same for each — a
    sum may fill a silence and may never settle an argument — and a later shape
    that can be ambiguous must not have to rediscover that.
    """
    from react_review.tools.batch_project import _NEVER_DERIVE
    assert CONTRADICTORY in _NEVER_DERIVE and AMBIGUOUS in _NEVER_DERIVE


# --- T3: malformed components are a broken answer, not a missing one ------

def test_t3_a_malformed_component_is_a_protocol_error_not_an_absence():
    reading = _batch(sets=[_set(counts=[{**ALLOCATED_ARMS[0], "count": "many"},
                                        *ALLOCATED_ARMS[1:]])])
    assert reading.aggregation_malformed
    projection = _project(reading)
    assert projection.aggregation_status == PROTOCOL_ERROR
    assert projection.status != DERIVED
    assert to_source_result(projection).aggregation_status == PROTOCOL_ERROR


def test_t3_a_count_the_quote_does_not_print_is_a_protocol_error():
    reading = _batch(sets=[_set(counts=[_count("nivolumab group", 999),
                                        *ALLOCATED_ARMS[1:]])])
    assert any("does not print 999" in e for e in reading.aggregation_errors)
    assert _project(reading).aggregation_status == PROTOCOL_ERROR


def test_t3_a_quote_that_does_not_name_the_arm_is_refused():
    reading = _batch(sets=[_set(counts=[_count("placebo group", 316),
                                        *ALLOCATED_ARMS[1:]])])
    assert any("does not name the arm" in e for e in reading.aggregation_errors)


def test_t3_one_arm_given_two_different_counts_is_refused():
    reading = _batch(sets=[_set(counts=[_count("nivolumab group", 316),
                                        _count("nivolumab group", 314)])])
    assert reading.aggregation_sets == []
    assert any("twice" in e for e in reading.aggregation_errors)


def test_t3_one_arm_given_the_SAME_count_twice_is_also_refused():
    """The more dangerous case: 316 + 316 is a study of 632 that never existed,
    and unlike a disagreement it leaves the sum looking perfectly plausible."""
    reading = _batch(sets=[_set(counts=[_count("nivolumab group", 316),
                                        _count("nivolumab group", 316)],
                                partition={**GOOD_PARTITION,
                                           "declared_arm_count": 2})])
    assert reading.aggregation_sets == []
    assert any("twice" in e for e in reading.aggregation_errors)
    assert _project(reading).status != DERIVED


def test_t3_an_arm_label_with_no_distinguishing_word_is_refused():
    reading = _batch(sets=[_set(counts=[_count("the", 316), *ALLOCATED_ARMS[1:]])])
    assert reading.aggregation_sets == []


def test_t5_an_analysed_count_smuggled_into_an_allocated_set_is_refused():
    """Listing it beside allocated counts does not make it one of them.

    This is the substitution the sets were meant to prevent, performed one level
    down: 316 + 313 + 315 = 944, a number for a group of people that never met.
    """
    reading = _batch(sets=[_set(counts=[
        _count("nivolumab group", 316),
        _count("Nivolumab plus Ipilimumab", 313, ANALYSIS_ROW),
        _count("ipilimumab group", 315)])])
    assert reading.aggregation_sets == []
    assert any("is not printed with the population" in e
               for e in reading.aggregation_errors)
    assert _project(reading).status != DERIVED


# --- T4: the arm census is what makes "complete" checkable ----------------

def test_t4_a_missing_arm_is_refused_even_though_the_rest_still_sum():
    """Two of three arms add to a number. It is not the study's total."""
    projection = _project(_batch(sets=[_set(counts=ALLOCATED_ARMS[:2])]))
    assert projection.status != DERIVED
    assert projection.derived_value is None
    # Caught at parse time now: the passage says three and two were offered, so
    # the census contradicts its own witness rather than merely falling short.
    assert projection.aggregation_status == PROTOCOL_ERROR
    assert "describes 3 groups" in projection.aggregation_reason


def test_t4_a_partition_with_no_census_at_all_is_refused():
    """`complete: true` about a list the code cannot see is unfalsifiable."""
    partition = {k: v for k, v in GOOD_PARTITION.items()
                 if k != "declared_arm_count"}
    projection = _project(_batch(sets=[_set(partition=partition)]))
    assert projection.status != DERIVED
    assert "cannot be checked" in projection.aggregation_reason


#: A census by NAME needs a passage that names them. The randomisation sentence
#: counts the groups; the allocation sentence names them.
NAMED_PARTITION = {"complete": True, "mutually_exclusive": True,
                   "quote": ALLOCATION, "declared_arm_labels":
                       ["nivolumab group", "nivolumab-plus-ipilimumab group",
                        "ipilimumab group"],
                   "reason": "the three assigned groups are named here"}


def test_t4_a_named_arm_with_no_count_is_refused():
    projection = _project(_batch(sets=[_set(counts=ALLOCATED_ARMS[:2],
                                            partition=NAMED_PARTITION)]))
    assert projection.status != DERIVED
    assert "not the same set of arms" in projection.aggregation_reason


def test_t4_the_census_may_be_satisfied_by_labels_alone():
    projection = _project(_batch(sets=[_set(partition=NAMED_PARTITION)]))
    assert projection.status == DERIVED and projection.derived_value == 945


def test_t4_a_declared_label_the_passage_does_not_name_is_refused():
    """The census must be READ from the passage, not supplied alongside it."""
    reading = _batch(sets=[_set(partition={**NAMED_PARTITION,
                                           "quote": PARTITION})])
    assert reading.aggregation_sets == []
    assert any("its own partition passage does not name" in e
               for e in reading.aggregation_errors)


def test_t4_a_count_the_passage_does_not_state_is_refused():
    """"three groups" in the paper, "2" from the model, and 630 summed."""
    reading = _batch(sets=[_set(counts=ALLOCATED_ARMS[:2],
                                partition={**GOOD_PARTITION,
                                           "declared_arm_count": 2})])
    assert reading.aggregation_sets == []
    assert any("which its own partition passage does not state" in e
               for e in reading.aggregation_errors)


def test_t4_a_census_written_in_words_is_read():
    """Papers write "three groups", not "3 groups"."""
    projection = _project(_batch(sets=[_set()]))
    assert projection.status == DERIVED
    assert "three" in GOOD_PARTITION["quote"] and "3" not in GOOD_PARTITION["quote"]


def test_not_complete_is_refused():
    projection = _project(_batch(sets=[_set(partition={**GOOD_PARTITION,
                                                       "complete": False})]))
    assert projection.status != DERIVED
    assert "cover the whole population" in projection.aggregation_reason


def test_not_mutually_exclusive_is_refused():
    projection = _project(_batch(sets=[_set(partition={
        **GOOD_PARTITION, "mutually_exclusive": False})]))
    assert projection.status != DERIVED
    assert "overlap" in projection.aggregation_reason


def test_a_true_with_no_locatable_quote_counts_as_a_false():
    projection = _project(_batch(sets=[_set(partition={
        **GOOD_PARTITION,
        "quote": "The three groups together comprised the whole trial population."})]))
    assert projection.status != DERIVED
    assert "not evidence" in projection.aggregation_reason


def test_a_single_arm_is_not_a_partition():
    projection = _project(_batch(sets=[_set(counts=ALLOCATED_ARMS[:1],
                                            partition={**GOOD_PARTITION,
                                                       "declared_arm_count": 1})]))
    assert projection.status != DERIVED
    assert projection.aggregation_status == PROTOCOL_ERROR


# --- T5 / T6 / T7: populations are not interchangeable --------------------

def test_t5_a_set_whose_population_witness_does_not_carry_the_phrase_is_refused():
    """The partition sentence is about randomisation; the counts are analysed."""
    reading = _batch(sets=[_set(counts=ANALYSED_ARMS, phrase="Analysis population",
                                witness=ALLOCATION)])
    assert reading.aggregation_sets == []
    assert any("witness passage does not contain" in e
               for e in reading.aggregation_errors)


def test_t5_a_component_has_no_population_of_its_own_to_disagree_with():
    """Mixing has nowhere to be written down: the SET owns the population."""
    reading = _batch(sets=[_set()])
    assert len(reading.aggregation_sets) == 1
    chosen = reading.aggregation_sets[0]
    assert chosen.population.basis == "allocated"
    assert not any(hasattr(c, "population") for c in chosen.cohort_counts)


def test_t6_an_analysed_set_cannot_answer_an_allocated_claim():
    projection = _project(_batch(sets=[ANALYSED_SET]))
    assert projection.status != DERIVED
    assert projection.aggregation_status == REJECTED
    assert "allocated" in projection.aggregation_reason


def test_t6_two_analysis_sets_of_the_same_basis_are_different_people():
    """analysed/ITT and analysed/per-protocol are not interchangeable."""
    itt = PopulationScope(basis="analysed", analysis_set="itt")
    projection = _project(_batch(sets=[ANALYSED_SET]), scope=itt)
    assert projection.status != DERIVED
    assert "analysis set is not the same people" in projection.aggregation_reason


def test_t7_an_unspecified_analysis_set_is_unknown_not_compatible():
    per_protocol = PopulationScope(basis="analysed", analysis_set="per_protocol")
    projection = _project(_batch(sets=[ANALYSED_SET]), scope=per_protocol)
    assert projection.status != DERIVED


# --- T8: the policy, not the prompt, is the permission --------------------

def test_t8_a_field_outside_the_whitelist_is_never_summed():
    """A response in the right shape for a field nobody may sum is still refused."""
    projection = _project(_batch(sets=[_set()]),
                          field_type="progression_free_survival")
    assert projection.status != DERIVED
    assert projection.aggregation_status == NOT_APPLICABLE
    assert "permits deriving a total only for" in projection.aggregation_reason


def test_t8_an_unnamed_field_is_not_summed():
    projection = _project(_batch(sets=[_set()]), field_type="")
    assert projection.status != DERIVED


def test_t8_the_policy_only_permits_whole_study_counts():
    policy = load_aggregation_policy()
    assert policy.applies_to("study", "sample_size")
    assert not policy.applies_to("study", "progression_free_survival")
    assert not policy.applies_to("arm", "sample_size")


# --- T9 / T10: choosing among sets is the code's job ----------------------

def test_t9_the_claim_selects_its_own_population_from_several_sets():
    reading = _batch(sets=[_set(), ANALYSED_SET])
    assert len(reading.aggregation_sets) == 2
    allocated = _project(reading, scope=ALLOCATED)
    assert allocated.status == DERIVED and allocated.derived_value == 945
    analysed = _project(reading, scope=ANALYSED)
    assert analysed.status == DERIVED and analysed.derived_value == 936


def test_t10_two_sets_that_match_equally_well_are_refused():
    """Not one of them chosen; nobody has said which the review means."""
    twin = _set(phrase="underwent randomization")
    projection = _project(_batch(sets=[_set(), twin]))
    assert projection.status != DERIVED
    assert "equally well" in projection.aggregation_reason


def test_a_broken_set_about_other_people_does_not_cost_this_claim():
    """The claim asks about the analysed; the broken set was about the allocated.

    It never held this claim's answer, so refusing the claim would punish it for
    a fault in a part of the response that did not concern it. The fault is
    still reported.
    """
    broken = _set(counts=[{**ALLOCATED_ARMS[0], "count": -1}])
    reading = _batch(sets=[broken, ANALYSED_SET])
    assert len(reading.aggregation_sets) == 1 and reading.aggregation_errors
    projection = _project(reading, scope=ANALYSED)
    assert projection.status == DERIVED and projection.derived_value == 936
    assert projection.unrelated_rejections == ["set 0 (allocated)"]


def test_a_broken_set_about_THESE_people_costs_the_claim():
    """Now nobody can show the surviving set was the only candidate."""
    broken = _set(counts=[{**ANALYSED_ARMS[0], "count": -1}],
                  phrase="Analysis population", witness=ANALYSIS_ROW,
                  partition={**GOOD_PARTITION, "quote": ANALYSIS_PARTITION,
                             "declared_arm_count": 3})
    reading = _batch(sets=[broken, ANALYSED_SET])
    projection = _project(reading, scope=ANALYSED)
    assert projection.status != DERIVED
    assert projection.aggregation_status == PROTOCOL_ERROR


def test_a_broken_set_whose_population_cannot_be_read_costs_the_claim():
    """Unknown means it might have been the one this claim needed."""
    reading = _batch(sets=[{"population_phrase": "", "cohort_counts": []},
                           ANALYSED_SET])
    projection = _project(reading, scope=ANALYSED)
    assert projection.status != DERIVED
    assert projection.aggregation_status == PROTOCOL_ERROR


# --- T11: timepoints ------------------------------------------------------

def test_t11_sets_at_different_timepoints_are_not_added_together():
    document = PAPER + "\n\nAt baseline 316 patients were assigned to the nivolumab group. At week 12, 314 to the nivolumab-plus-ipilimumab group."
    reading = parse_batch({"readings": [], "aggregation_sets": [{
        "population_phrase": "underwent randomization",
        "population_quote": ALLOCATION,
        "timepoint_phrase": "At baseline",
        "timepoint_quote": "At baseline 316 patients were assigned to the nivolumab group.",
        "cohort_counts": [_count("nivolumab group", 316)],
        "partition": {**GOOD_PARTITION, "declared_arm_count": 1}}]},
        document, target_shape=STUDY, aggregable=True)
    projection = _project(reading)
    assert projection.status != DERIVED


def test_t11_a_timepoint_with_no_passage_behind_it_is_refused():
    reading = _batch(sets=[_set(timepoint="at 5 years",
                                timepoint_quote=ALLOCATION)])
    assert reading.aggregation_sets == []
    assert any("witness passage does not contain" in e
               for e in reading.aggregation_errors)


# --- T12: a broken sum never costs a good printed total -------------------

def test_t12_an_explicit_total_survives_a_malformed_aggregation():
    reading = _batch(readings=[
        _total("945", "A total of 945 patients underwent randomization")],
        sets=[_set(counts=[{**ALLOCATED_ARMS[0], "count": "not a number"}])])
    projection = _project(reading)
    assert projection.status == OK and projection.value == "945"
    # But the broken set is still on the record, on the released path.
    assert projection.aggregation_status == PROTOCOL_ERROR
    assert projection.aggregation_errors
    result = to_source_result(projection)
    assert result.found and result.value == "945"
    assert result.aggregation_status == PROTOCOL_ERROR
    assert "not one positive whole number" in result.aggregation_reason
    # …and said once, not once per path that carried it.
    assert result.aggregation_reason.count("aggregation set 0") == 1


# --- explicit and derived, together ---------------------------------------

def test_an_explicit_total_that_agrees_is_released_with_the_sum_as_corroboration():
    reading = _batch(readings=[
        _total("945", "A total of 945 patients underwent randomization")],
        sets=[_set()])
    projection = _project(reading)
    assert projection.status == OK and projection.value == "945"
    assert "independently add to the same total" in projection.aggregation_reason


def test_an_explicit_total_that_disagrees_with_the_sum_releases_nothing():
    document = PAPER.replace("A total of 945", "A total of 944")
    reading = _batch(readings=[
        _total("944", "A total of 944 patients underwent randomization")],
        sets=[_set(witness=document.split("\n\n")[0],
                   counts=[_count("nivolumab group", 316, document.split("\n\n")[0]),
                           _count("nivolumab-plus-ipilimumab group", 314,
                                  document.split("\n\n")[0]),
                           _count("ipilimumab group", 315,
                                  document.split("\n\n")[0])])],
        document=document)
    projection = _project(reading)
    assert projection.status == CONTRADICTORY
    assert "944" in projection.reason and "945" in projection.reason
    assert to_source_result(projection).found is False


def test_an_analysed_total_does_not_block_deriving_the_allocated_one():
    """The printed total is for other people; the components are for these."""
    document = PAPER + "\n\nThe analysis population comprised 936 patients."
    reading = _batch(readings=[{"scope_label": "analysis population",
                                "value": "936",
                                "quote": "The analysis population comprised 936 patients.",
                                "population_phrase": "analysis population"}],
                     sets=[_set()], document=document)
    projection = _project(reading)
    assert projection.status == DERIVED and projection.derived_value == 945


def test_a_scope_that_no_set_matches_exactly_is_not_answered_by_a_sum():
    reading = _batch(readings=[{"scope_label": "analysis population",
                                "value": "936",
                                "quote": ANALYSIS_PARTITION.replace(
                                    "comprised the three treatment groups",
                                    "comprised 936 patients"),
                                "population_phrase": "analysis population"}],
                     sets=[ANALYSED_SET],
                     document=PAPER + "\n\nThe analysis population comprised 936 patients.")
    projection = _project(reading, scope=PopulationScope(basis="treated"))
    assert projection.status in (SCOPE_UNRESOLVED, NOT_REPORTED)
    assert projection.derived_value is None


# --- nothing offered ------------------------------------------------------

def test_nothing_is_derived_when_no_components_were_offered():
    projection = _project(_batch(readings=[]))
    assert projection.status == NOT_REPORTED
    assert projection.aggregation_status == NOT_APPLICABLE
    assert "does not print" in projection.reason


def test_the_policy_is_hashed_so_a_run_can_say_which_one_it_applied():
    policy = load_aggregation_policy()
    assert len(policy.sha256) == 64 and policy.policy_id == "safe_sum_v5"


# --- T11 again: a timepoint is checked even with only one candidate -------

#: One passage carrying everything a timed set needs: the population words, the
#: timepoint words, every arm and its count, and the census. It has to, now that
#: each component must be shown to be counted at that population AND that
#: moment — which is the point of the rule and not an inconvenience of it.
BASELINE = ("At baseline, among those who underwent randomization, 316 patients "
            "were in the nivolumab group, 314 in the nivolumab-plus-ipilimumab "
            "group and 315 in the ipilimumab group, one of three groups.")
TIMED_PAPER = PAPER + "\n\n" + BASELINE


def _timed_set(timepoint):
    return {"population_phrase": "underwent randomization",
            "population_quote": BASELINE,
            "timepoint_phrase": timepoint, "timepoint_quote": BASELINE,
            "cohort_counts": [_count("nivolumab group", 316, BASELINE),
                              _count("nivolumab-plus-ipilimumab group", 314, BASELINE),
                              _count("ipilimumab group", 315, BASELINE)],
            "partition": {**GOOD_PARTITION, "quote": BASELINE,
                          "declared_arm_count": 3}}


def test_t11_the_only_set_is_still_checked_against_the_claims_timepoint():
    """Being the only candidate is not evidence of being the right one."""
    reading = parse_batch({"readings": [],
                           "aggregation_sets": [_timed_set("At baseline")]},
                          TIMED_PAPER, target_shape=STUDY, aggregable=True)
    assert len(reading.aggregation_sets) == 1
    projection = _project(reading, timepoint_label="week 12")
    assert projection.status != DERIVED
    assert projection.derived_value is None
    assert "no set of counts is stated at that timepoint" in projection.aggregation_reason


def test_t11_the_matching_timepoint_is_derived():
    reading = parse_batch({"readings": [],
                           "aggregation_sets": [_timed_set("At baseline")]},
                          TIMED_PAPER, target_shape=STUDY, aggregable=True)
    projection = _project(reading, timepoint_label="at baseline")
    assert projection.status == DERIVED and projection.derived_value == 945


def test_t11_a_set_with_no_timepoint_cannot_answer_a_timed_claim():
    projection = _project(_batch(sets=[_set()]), timepoint_label="week 12")
    assert projection.status != DERIVED
    assert "carry no timepoint of their own" in projection.aggregation_reason


def test_t11_a_shared_word_is_not_a_shared_moment():
    """"overall survival at 5 years" and "median progression-free survival"
    have "survival" in common and nothing else."""
    reading = parse_batch(
        {"readings": [], "aggregation_sets": [_timed_set("At baseline")]},
        TIMED_PAPER, target_shape=STUDY, aggregable=True)
    assert _project(reading, timepoint_label="baseline characteristics").status != DERIVED


# --- A6.2: the account of a derived total reaches the evidence object -----

def test_a_derived_total_carries_its_policy_and_all_four_kinds_of_anchor():
    projection = _project(_batch(sets=[_set()]))
    result = to_source_result(projection)
    provenance = result.aggregation_provenance
    assert provenance.policy_id == "safe_sum_v5"
    assert len(provenance.policy_sha256) == 64
    assert provenance.aggregation_set.startswith("allocated")
    assert provenance.population_quote == ALLOCATION
    assert provenance.partition_quote == PARTITION
    assert len(provenance.component_quotes) == 3
    # Four kinds of passage, none of which substitutes for another: what was
    # added, why adding them is the whole, and whom they count.
    assert len(provenance.anchors) == 5
    assert projection.evidence_anchors == provenance.anchors


def test_a_timed_derived_total_carries_its_timepoint_anchor_too():
    reading = parse_batch({"readings": [],
                           "aggregation_sets": [_timed_set("At baseline")]},
                          TIMED_PAPER, target_shape=STUDY, aggregable=True)
    provenance = to_source_result(
        _project(reading, timepoint_label="at baseline")).aggregation_provenance
    assert provenance.timepoint_quote == BASELINE
    assert len(provenance.anchors) == 6


def test_a_released_printed_total_still_carries_the_broken_sets():
    reading = _batch(readings=[
        _total("945", "A total of 945 patients underwent randomization")],
        sets=[_set(counts=[{**ALLOCATED_ARMS[0], "count": "not a number"}])])
    provenance = to_source_result(_project(reading)).aggregation_provenance
    assert provenance.errors and provenance.policy_id == "safe_sum_v5"


# --- round 3: the sentence that licenses the sum must be about these people ---

TWO_GROUP_ANALYSIS = ("The analysis population comprised two treatment groups, "
                      "each patient counted once.")
RATIO_PARTITION = ("Patients were randomised in a 2:1:1 ratio to one of three "
                   "groups, nivolumab, nivolumab-plus-ipilimumab and ipilimumab.")
#: The ratio sentence sits WITH the allocation sentence, as it does in a paper.
#: A randomisation sentence names no population of its own, so it is accepted
#: only where the paper puts it beside the counts it divides — parked at the end
#: of the document it is refused, which is the same-block rule working.
ROUND3_PAPER = "\n\n".join([ALLOCATION + " " + RATIO_PARTITION, PARTITION,
                            ANALYSIS_ROW, ANALYSIS_PARTITION, TWO_GROUP_ANALYSIS])


def test_a_partition_about_the_analysed_cannot_license_allocated_counts():
    """"the analysis population comprised two groups" says nothing about who
    was randomised, and 316 + 314 is not the trial."""
    reading = _batch(sets=[_set(counts=ALLOCATED_ARMS[:2],
                                partition={"complete": True,
                                           "mutually_exclusive": True,
                                           "quote": TWO_GROUP_ANALYSIS,
                                           "declared_arm_count": 2})],
                     document=ROUND3_PAPER)
    assert reading.aggregation_sets == []
    assert any("describes 'analysed'" in e for e in reading.aggregation_errors)
    assert _project(reading).status != DERIVED


def test_a_census_may_not_be_read_out_of_a_randomisation_ratio():
    """"2:1:1" contains a 2 and declares three groups, not two."""
    reading = _batch(sets=[_set(counts=ALLOCATED_ARMS[:2],
                                partition={"complete": True,
                                           "mutually_exclusive": True,
                                           "quote": RATIO_PARTITION,
                                           "declared_arm_count": 2})],
                     document=ROUND3_PAPER)
    assert reading.aggregation_sets == []
    assert any("does not state" in e for e in reading.aggregation_errors)


def test_the_same_ratio_passage_does_support_its_real_census_of_three():
    projection = _project(_batch(sets=[_set(partition={
        "complete": True, "mutually_exclusive": True,
        "quote": RATIO_PARTITION, "declared_arm_count": 3})],
        document=ROUND3_PAPER))
    assert projection.status == DERIVED and projection.derived_value == 945


ITT_ROW = ("Table 4. ITT analysis population: Nivolumab (N = 311), Nivolumab "
           "plus Ipilimumab (N = 313), Ipilimumab (N = 312)")
ITT_PAPER = PAPER + "\n\n" + ITT_ROW
ITT = PopulationScope(basis="analysed", analysis_set="itt")
BOTH_AXES = ["population_basis", "analysis_set"]


def test_a_broken_set_whose_analysis_set_is_unstated_still_blocks_an_itt_claim():
    """Unresolved is not "confirmed to be about other people".

    An analysed set that never names its analysis set may well have been the ITT
    one; treating it as unrelated assumes the answer to the question nobody
    could answer.
    """
    broken = _set(counts=[{**ANALYSED_ARMS[0], "count": -1}, *ANALYSED_ARMS[1:]],
                  phrase="Analysis population", witness=ANALYSIS_ROW,
                  partition={**GOOD_PARTITION, "quote": ANALYSIS_PARTITION})
    good = _set(counts=[_count("Nivolumab", 311, ITT_ROW),
                        _count("Nivolumab plus Ipilimumab", 313, ITT_ROW),
                        _count("Ipilimumab", 312, ITT_ROW)],
                phrase="ITT analysis population", witness=ITT_ROW,
                partition={**GOOD_PARTITION, "quote": ANALYSIS_PARTITION})
    reading = _batch(sets=[broken, good], document=ITT_PAPER)
    projection = project_claim(reading, target_shape=STUDY, requested_scope=ITT,
                               required_axes=BOTH_AXES, field_type="sample_size")
    assert projection.status != DERIVED
    assert projection.aggregation_status == PROTOCOL_ERROR


def test_a_claim_that_names_no_population_is_not_answered_by_the_only_set():
    """on_unknown: reject, applied to the claim's own side.

    A derived total is the kind of answer nobody can re-check against a printed
    number, so which population it counts has to have been asked for — not
    inferred from there being nothing else on offer.
    """
    projection = _project(_batch(sets=[_set()]), scope=None)
    assert projection.status != DERIVED
    assert projection.aggregation_status == REJECTED
    assert "does not say which population" in projection.aggregation_reason


def test_an_unknown_claim_population_is_refused_just_as_a_missing_one_is():
    projection = _project(_batch(sets=[_set()]), scope=PopulationScope())
    assert projection.status != DERIVED
    assert projection.aggregation_status == REJECTED


# --- an arm projection never pretends a sum was considered ----------------

def test_an_arm_projection_carries_no_aggregation_provenance_at_all():
    """Empty fields would say a sum left no trace. None says it never arose."""
    from react_review.normalize.cohorts import parse_comparison   # noqa: F401
    reading = parse_batch({"readings": [
        {"arm_label": "nivolumab group", "value": "316", "quote": ALLOCATION,
         "population_phrase": "underwent randomization"}]}, PAPER)
    projection = project_claim(
        reading, review_labels={"a": "Nivolumab (3 mg/kg)"}, cohort_key="a")
    assert to_source_result(projection).aggregation_provenance is None


# --- round 4 (D1-4B): every axis, and the axes the run contract demands ---

ITT_TABLE = ("Table 5. ITT analysis population: alfa (N = 40), beta (N = 60), "
             "one of two groups.")
WEEK12 = ("At week 12, among those who underwent randomization, 41 were in the "
          "alfa group and 61 in the beta group, one of two groups.")
BASELINE_PARTITION = ("At baseline the trial comprised two groups, alfa and beta, "
                      "with no patient in both.")
#: The baseline partition sits WITH the week-12 counts, so the population axis
#: is satisfied by proximity and the TIMEPOINT axis is the one under test. That
#: is the realistic shape of the trap: two sentences printed together, about the
#: same people, at different visits.
B_PAPER = "\n\n".join([PAPER, ITT_TABLE, WEEK12 + " " + BASELINE_PARTITION])
BOTH = ["population_basis", "analysis_set"]
ITT = PopulationScope(basis="analysed", analysis_set="itt")


def _itt_set(partition_quote):
    return {"population_phrase": "ITT analysis population",
            "population_quote": ITT_TABLE,
            "cohort_counts": [_count("alfa", 40, ITT_TABLE),
                              _count("beta", 60, ITT_TABLE)],
            "partition": {"complete": True, "mutually_exclusive": True,
                          "quote": partition_quote, "declared_arm_count": 2}}


def test_b1_an_itt_set_needs_a_partition_that_says_itt():
    """Agreeing about the basis is not agreeing.

    "The analysis population comprised three groups" is silent about whether the
    ITT set did, and silence is not a licence to add ITT counts together.
    """
    reading = _batch(sets=[_itt_set(ANALYSIS_PARTITION)], document=B_PAPER)
    assert reading.aggregation_sets == []
    assert any("neither says so nor sits with" in e
               for e in reading.aggregation_errors)


def test_b1_the_same_set_derives_when_its_partition_is_in_the_itt_passage():
    projection = project_claim(
        _batch(sets=[_itt_set(ITT_TABLE)], document=B_PAPER), target_shape=STUDY,
        requested_scope=ITT, required_axes=BOTH, field_type="sample_size")
    assert projection.status == DERIVED and projection.derived_value == 100


def test_b2_week_12_counts_are_not_completed_by_a_baseline_partition():
    """Groups complete at baseline need not be complete twelve weeks later."""
    reading = _batch(sets=[{
        "population_phrase": "underwent randomization", "population_quote": WEEK12,
        "timepoint_phrase": "At week 12", "timepoint_quote": WEEK12,
        "cohort_counts": [_count("alfa", 41, WEEK12), _count("beta", 61, WEEK12)],
        "partition": {"complete": True, "mutually_exclusive": True,
                      "quote": BASELINE_PARTITION, "declared_arm_count": 2}}],
        document=B_PAPER)
    assert reading.aggregation_sets == []
    assert any("does not say it holds at that moment" in e
               for e in reading.aggregation_errors)


def test_b3_a_dose_is_not_a_census():
    """A number two words from a noun is not a count of groups.

    THREE valid components and a declared census of three, so the only thing
    that can refuse this is the grammar: the sole 3 in the partition passage is
    a dose. An earlier version of this test offered two components against a
    census of three and would have stayed green even if "3 mg treatment" were
    read as three groups, because the count check would have caught it instead.
    """
    dosed = ("Patients received 3 mg treatment, with no patient in more than "
             "one of them.")
    reading = _batch(sets=[{
        "population_phrase": "underwent randomization",
        "population_quote": ALLOCATION, "cohort_counts": ALLOCATED_ARMS,
        "partition": {"complete": True, "mutually_exclusive": True,
                      "quote": dosed, "declared_arm_count": 3}}],
        # Beside the allocation sentence, so the population axis is satisfied
        # and the CENSUS is the only thing left to refuse it.
        document=ALLOCATION + " " + dosed + "\n\n" + PARTITION)
    assert dosed.count("3") == 1 and "3 mg" in dosed
    assert reading.aggregation_sets == []
    assert any("does not state" in e for e in reading.aggregation_errors)


def test_b3_the_same_three_components_derive_with_a_real_census():
    """The counterpart: nothing else about that set was wrong."""
    projection = _project(_batch(sets=[_set()]))
    assert projection.status == DERIVED and projection.derived_value == 945


def test_b4_a_profile_requiring_an_analysis_set_refuses_components_without_one():
    """The run contract's axes reach the sum, not only the printed total."""
    projection = project_claim(
        _batch(sets=[ANALYSED_SET]), target_shape=STUDY, requested_scope=ANALYSED,
        required_axes=BOTH, field_type="sample_size")
    assert projection.status != DERIVED
    assert projection.required_axes == sorted(BOTH)


def test_b4_the_same_components_derive_when_the_profile_asks_only_for_a_basis():
    projection = project_claim(
        _batch(sets=[ANALYSED_SET]), target_shape=STUDY, requested_scope=ANALYSED,
        required_axes=["population_basis"], field_type="sample_size")
    assert projection.status == DERIVED and projection.derived_value == 936


def test_b5_a_broken_set_about_other_people_is_unrelated():
    broken = _set(counts=[{**ALLOCATED_ARMS[0], "count": -1}])
    projection = _project(_batch(sets=[broken, ANALYSED_SET]), scope=ANALYSED)
    assert projection.status == DERIVED
    assert projection.unrelated_rejections == ["set 0 (allocated)"]


def test_b6_a_broken_set_of_the_same_people_at_another_time_is_unrelated():
    """One axis definitely different is enough: it cannot have held this answer."""
    broken = {"population_phrase": "underwent randomization",
              "population_quote": WEEK12, "timepoint_phrase": "At week 12",
              "timepoint_quote": WEEK12,
              "cohort_counts": [{**_count("alfa", 41, WEEK12), "count": -1}],
              "partition": {"complete": True, "mutually_exclusive": True,
                            "quote": WEEK12, "declared_arm_count": 2}}
    reading = parse_batch(
        {"readings": [], "aggregation_sets": [broken, _timed_set("At baseline")]},
        TIMED_PAPER + "\n\n" + WEEK12, target_shape=STUDY, aggregable=True)
    projection = _project(reading, timepoint_label="At baseline")
    assert projection.status == DERIVED and projection.derived_value == 945
    assert projection.unrelated_rejections


def test_b6_a_broken_set_nobody_can_place_still_blocks():
    reading = _batch(sets=[{"population_phrase": "", "cohort_counts": []},
                           ANALYSED_SET])
    assert _project(reading, scope=ANALYSED).status != DERIVED


# --- the evaluator travels with every outcome ----------------------------

def _runtime():
    from react_review.tools.safe_aggregation import AggregationRuntime
    return AggregationRuntime.resolve(policy_id="safe_sum_v5",
                                      evaluator_version="1.6.0")


@pytest.mark.parametrize("sets,scope,expected", [
    ([_set()], ALLOCATED, DERIVED),
    ([ANALYSED_SET], ALLOCATED, NOT_REPORTED),
    ([_set(partition={**GOOD_PARTITION, "complete": "false"})], ALLOCATED,
     NOT_REPORTED),
])
def test_every_aggregation_outcome_names_the_code_that_decided(sets, scope, expected):
    """A refusal is a decision too, and a reader must be able to ask whose."""
    projection = project_claim(_batch(sets=sets), target_shape=STUDY,
                               requested_scope=scope, required_axes=AXES,
                               field_type="sample_size", runtime=_runtime())
    assert projection.status == expected
    provenance = to_source_result(projection).aggregation_provenance
    assert provenance.evaluator_id == "safe_aggregation"
    assert provenance.evaluator_version == "1.6.0"
    assert provenance.evaluator_hash.startswith("sha256:")
    assert provenance.policy_id == "safe_sum_v5"


def test_a_printed_total_that_won_still_names_the_evaluator_that_checked_it():
    reading = _batch(readings=[_total(
        "945", "A total of 945 patients underwent randomization")], sets=[_set()])
    projection = project_claim(reading, target_shape=STUDY,
                               requested_scope=ALLOCATED, required_axes=AXES,
                               field_type="sample_size", runtime=_runtime())
    assert projection.status == OK
    provenance = to_source_result(projection).aggregation_provenance
    assert provenance.evaluator_version == "1.6.0"


def test_without_an_identity_nothing_is_release_eligible():
    provenance = to_source_result(
        _project(_batch(sets=[_set()]))).aggregation_provenance
    assert provenance.release_eligible is False
    assert provenance.evaluator_status == "unavailable"


# --- round 5 (C2): one set of axes, both routes ---------------------------

ITT_ONLY = PopulationScope(basis="analysed", analysis_set="itt")


def _printed_analysis_total():
    row = ("Table 9. Analysis population: alfa (N = 11), beta (N = 22), 33 in "
           "total.")
    return parse_batch({"readings": [{
        "scope_label": "analysis population", "value": "33", "quote": row,
        "population_phrase": "Analysis population"}]}, row,
        target_shape=STUDY, aggregable=True)


def test_a_printed_total_is_held_to_the_axes_the_result_claims_to_have_used():
    """The claim names ITT; the paper's total names no analysis set.

    This was released as `ok` while the provenance recorded that analysis_set
    had been among the axes — an answer carrying an attestation to a check that
    never ran, which is worse than an answer that is merely wrong.
    """
    projection = project_claim(
        _printed_analysis_total(), target_shape=STUDY, requested_scope=ITT_ONLY,
        required_axes=["population_basis"], field_type="sample_size")
    assert projection.status == SCOPE_UNRESOLVED
    assert to_source_result(projection).found is False
    assert "analysis_set" in projection.required_axes


def test_the_axes_a_result_records_are_the_axes_that_were_applied():
    projection = project_claim(
        _printed_analysis_total(), target_shape=STUDY, requested_scope=ANALYSED,
        required_axes=["population_basis"], field_type="sample_size")
    assert projection.status == OK and projection.value == "33"
    assert projection.required_axes == ["population_basis"]


def test_a_field_nobody_may_sum_does_not_inherit_the_aggregation_floor():
    """The policy floor is the policy's business only where the policy applies."""
    projection = project_claim(
        _printed_analysis_total(), target_shape=STUDY, requested_scope=None,
        required_axes=[], field_type="progression_free_survival")
    assert projection.required_axes == []


def test_a_policy_readiness_never_saw_cannot_be_vouched_for_by_a_real_identity():
    """The projector takes ONE object, so the two cannot be mismatched at all.

    Before this, `aggregation_policy=` and `evaluator=` were separate arguments
    and a result could name a rogue policy while reporting release_eligible.
    """
    import inspect

    from dataclasses import replace

    signature = inspect.signature(project_claim).parameters
    assert "aggregation_policy" not in signature and "evaluator" not in signature

    good = _runtime()
    rogue = replace(good, policy=replace(good.policy, policy_id="rogue_policy",
                                         sha256="0" * 64))
    projection = project_claim(_batch(sets=[_set()]), target_shape=STUDY,
                               requested_scope=ALLOCATED, required_axes=AXES,
                               field_type="sample_size", runtime=rogue)
    provenance = to_source_result(projection).aggregation_provenance
    assert provenance.policy_id == "rogue_policy"
    assert provenance.release_eligible is False


def test_the_scope_axes_cannot_be_declared_pre_computed_by_a_caller():
    """The flag turned a rejected ITT claim into a derived 945."""
    import inspect

    from react_review.tools.safe_aggregation import derive_partitioned_total

    assert "axes_already_effective" not in inspect.signature(
        derive_partitioned_total).parameters
    itt = PopulationScope(basis="allocated", analysis_set="itt")
    outcome = derive_partitioned_total(
        _batch(sets=[_set()]).aggregation_sets, itt, target_shape=STUDY,
        field_type="sample_size", required_axes=["population_basis"])
    assert outcome.required_axes == ["analysis_set", "population_basis"]
    assert outcome.status != "derived"
