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
    assert projection.policy_id == "safe_sum_v1"
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
    reading = _batch(sets=[_set(counts=[*ALLOCATED_ARMS,
                                        _count("Nivolumab", 311, ANALYSIS_ROW)])])
    assert any("two different counts" in e for e in reading.aggregation_errors)


# --- T4: the arm census is what makes "complete" checkable ----------------

def test_t4_a_missing_arm_is_refused_even_though_the_rest_still_sum():
    """Two of three arms add to a number. It is not the study's total."""
    projection = _project(_batch(sets=[_set(counts=ALLOCATED_ARMS[:2])]))
    assert projection.status != DERIVED
    assert projection.derived_value is None
    assert projection.aggregation_status == REJECTED
    assert "3 groups" in projection.aggregation_reason


def test_t4_a_partition_with_no_census_at_all_is_refused():
    """`complete: true` about a list the code cannot see is unfalsifiable."""
    partition = {k: v for k, v in GOOD_PARTITION.items()
                 if k != "declared_arm_count"}
    projection = _project(_batch(sets=[_set(partition=partition)]))
    assert projection.status != DERIVED
    assert "cannot be checked" in projection.aggregation_reason


def test_t4_a_named_arm_with_no_count_is_refused():
    partition = {**GOOD_PARTITION, "declared_arm_count": None,
                 "declared_arm_labels": ["nivolumab group",
                                         "nivolumab-plus-ipilimumab group",
                                         "ipilimumab group"]}
    projection = _project(_batch(sets=[_set(counts=ALLOCATED_ARMS[:2],
                                            partition=partition)]))
    assert projection.status != DERIVED
    assert "ipilimumab group" in projection.aggregation_reason


def test_t4_the_census_may_be_satisfied_by_labels_alone():
    partition = {**GOOD_PARTITION, "declared_arm_count": None,
                 "declared_arm_labels": ["nivolumab group",
                                         "nivolumab-plus-ipilimumab group",
                                         "ipilimumab group"]}
    projection = _project(_batch(sets=[_set(partition=partition)]))
    assert projection.status == DERIVED and projection.derived_value == 945


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
    assert "at least 2" in projection.aggregation_reason


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


def test_a_set_that_fails_to_parse_does_not_cost_the_others():
    broken = _set(counts=[{**ALLOCATED_ARMS[0], "count": -1}])
    reading = _batch(sets=[broken, ANALYSED_SET])
    assert len(reading.aggregation_sets) == 1
    assert reading.aggregation_errors
    # The surviving set is still usable for the claim it answers, and the
    # broken one is still reported.
    projection = _project(reading, scope=ANALYSED)
    assert projection.aggregation_status == PROTOCOL_ERROR
    assert projection.aggregation_errors


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
    assert len(policy.sha256) == 64 and policy.policy_id == "safe_sum_v1"
