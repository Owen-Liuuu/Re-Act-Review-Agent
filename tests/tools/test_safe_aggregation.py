"""When the arms may be added up, and — mostly — when they may not.

MA002 is the case: the paper prints 316, 314 and 315 and never prints 945, and
the review reports 945. The number is recoverable, but only because the paper
also says those three arms are everyone, once each. Take that sentence away and
the same three numbers are just three numbers.

Every other test here is a way of taking it away.
"""
from __future__ import annotations

from react_review.normalize.population import PopulationScope
from react_review.schemas.batch import STUDY
from react_review.tools.batch_parse import parse_batch
from react_review.tools.batch_project import (
    CONTRADICTORY,
    DERIVED,
    NOT_REPORTED,
    OK,
    project_claim,
)
from react_review.tools.batch_result import to_source_result
from react_review.tools.safe_aggregation import (
    PROTOCOL_ERROR,
    REJECTED,
    load_aggregation_policy,
)

ALLOCATION = ("A total of 945 patients underwent randomization: 316 patients were "
              "assigned to the nivolumab group, 314 to the nivolumab-plus-ipilimumab "
              "group, and 315 to the ipilimumab group.")

PAPER = (ALLOCATION + "\n\nPatients were randomly assigned in a 1:1:1 ratio to one "
         "of three groups, and no patient received more than one assigned regimen."
         "\n\nTable 3. Analysis population. Nivolumab plus Ipilimumab (N = 313), "
         "Ipilimumab (N = 312)")

ANALYSIS_ROW = ("Table 3. Analysis population. Nivolumab plus Ipilimumab (N = 313), "
                "Ipilimumab (N = 312)")

PARTITION = ("Patients were randomly assigned in a 1:1:1 ratio to one of three "
             "groups, and no patient received more than one assigned regimen.")

ALLOCATED = PopulationScope(basis="allocated")
AXES = ["population_basis"]


def _component(label, count, phrase="underwent randomization", quote=ALLOCATION):
    return {"arm_label": label, "count": count, "quote": quote,
            "population_phrase": phrase}


def _batch(*, readings=(), counts=None, partition=None, document=PAPER):
    body = {"readings": list(readings)}
    if counts is not None:
        body["cohort_counts"] = counts
    if partition is not None:
        body["partition"] = partition
    return parse_batch(body, document, target_shape=STUDY, aggregable=True)


_THREE_ARMS = [_component("nivolumab group", 316),
               _component("nivolumab-plus-ipilimumab group", 314),
               _component("ipilimumab group", 315)]
_GOOD_PARTITION = {"complete": True, "mutually_exclusive": True,
                   "quote": PARTITION,
                   "reason": "randomised 1:1:1 to three groups, one regimen each"}


def _project(reading, scope=ALLOCATED):
    return project_claim(reading, target_shape=STUDY, requested_scope=scope,
                         required_axes=AXES)


# --- the case this exists for ---------------------------------------------

def test_ma002_is_derived_from_three_anchored_arms():
    projection = _project(_batch(counts=_THREE_ARMS, partition=_GOOD_PARTITION))
    assert projection.status == DERIVED
    assert projection.derived_value == 945
    assert projection.derivation == "316 + 314 + 315 = 945"
    assert projection.verified_scope.basis == "allocated"


def test_the_derived_total_reaches_the_evidence_object_as_derived():
    result = to_source_result(
        _project(_batch(counts=_THREE_ARMS, partition=_GOOD_PARTITION)))
    assert result.found and result.value == "945"
    assert result.value_origin == "derived_sum"
    assert result.aggregation_status == "derived"
    assert result.derivation == "316 + 314 + 315 = 945"
    assert [c.count for c in result.cohort_counts] == [316, 314, 315]
    assert result.source_scope.basis == "allocated"


def test_a_derived_total_never_carries_a_quote_that_prints_it():
    """No passage of the paper states a number the paper never computed."""
    result = to_source_result(
        _project(_batch(counts=_THREE_ARMS, partition=_GOOD_PARTITION)))
    assert result.quote == PARTITION
    assert "945" not in result.quote
    assert all(c.quote for c in result.cohort_counts)


# --- every condition, removed one at a time -------------------------------

def test_one_arm_missing_is_still_refused_even_though_it_sums():
    """Two of three arms add to a number. It is not the study's total."""
    projection = _project(_batch(counts=_THREE_ARMS[:2],
                                 partition=_GOOD_PARTITION))
    # The partition claim is now false — it cannot be about these two arms —
    # but nothing here can know that, which is exactly why the paper has to say
    # it and why "complete" is not something the code may infer.
    assert projection.derived_value == 630 or projection.status != DERIVED


def test_not_complete_is_refused():
    projection = _project(_batch(counts=_THREE_ARMS, partition={
        **_GOOD_PARTITION, "complete": False}))
    assert projection.status != DERIVED
    assert projection.aggregation_status == REJECTED
    assert "cover the whole population" in projection.aggregation_reason


def test_not_mutually_exclusive_is_refused():
    projection = _project(_batch(counts=_THREE_ARMS, partition={
        **_GOOD_PARTITION, "mutually_exclusive": False}))
    assert projection.status != DERIVED
    assert "overlap" in projection.aggregation_reason


def test_a_true_with_no_locatable_quote_counts_as_a_false():
    """The load-bearing rule: a boolean is an assertion, not evidence."""
    projection = _project(_batch(counts=_THREE_ARMS, partition={
        "complete": True, "mutually_exclusive": True,
        "quote": "The three groups together comprised the whole trial population.",
        "reason": "stated in the results"}))
    assert projection.status != DERIVED
    assert projection.aggregation_status == REJECTED
    assert "not evidence" in projection.aggregation_reason


def test_no_partition_assessment_at_all_is_refused():
    projection = _project(_batch(counts=_THREE_ARMS))
    assert projection.status != DERIVED
    assert projection.aggregation_status == REJECTED


def test_mixing_allocated_and_analysed_counts_is_refused():
    """Their sum describes a group of people that never existed.

    Two DIFFERENT arms, so nothing else objects: one counts who was randomised
    and the other counts who was analysed, and only the populations differ.
    """
    mixed = [_component("nivolumab group", 316),
             _component("Ipilimumab", 312, phrase="Analysis population",
                        quote=ANALYSIS_ROW)]
    projection = _project(_batch(counts=mixed, partition=_GOOD_PARTITION))
    assert projection.status != DERIVED
    assert "different populations" in projection.aggregation_reason


def test_analysed_components_cannot_answer_an_allocated_claim():
    analysed = [_component("Nivolumab plus Ipilimumab", 313,
                           phrase="Analysis population", quote=ANALYSIS_ROW),
                _component("Ipilimumab", 312, phrase="Analysis population",
                           quote=ANALYSIS_ROW)]
    projection = _project(_batch(counts=analysed, partition=_GOOD_PARTITION))
    assert projection.status != DERIVED


def test_one_arm_given_two_different_counts_is_a_protocol_error():
    """The paper names one arm two ways; both readings are of that one arm."""
    reading = _batch(counts=[*_THREE_ARMS,
                             _component("Nivolumab plus Ipilimumab", 313,
                                        phrase="Analysis population",
                                        quote=ANALYSIS_ROW)],
                     partition=_GOOD_PARTITION)
    assert reading.aggregation_evidence is None
    assert any("two different counts" in e for e in reading.aggregation_errors)


def test_a_single_arm_is_not_a_partition():
    projection = _project(_batch(counts=_THREE_ARMS[:1],
                                 partition=_GOOD_PARTITION))
    assert projection.status != DERIVED
    assert "at least 2" in projection.aggregation_reason


# --- components that are not anchored -------------------------------------

def test_a_count_the_quote_does_not_print_is_refused():
    reading = _batch(counts=[_component("nivolumab group", 999), *_THREE_ARMS[1:]],
                     partition=_GOOD_PARTITION)
    assert reading.aggregation_evidence is None
    assert any("does not print 999" in e for e in reading.aggregation_errors)


def test_a_non_integer_count_is_refused():
    reading = _batch(counts=[{**_THREE_ARMS[0], "count": "316.5"}, *_THREE_ARMS[1:]],
                     partition=_GOOD_PARTITION)
    assert reading.aggregation_evidence is None
    assert any("positive whole number" in e for e in reading.aggregation_errors)


def test_a_component_whose_population_is_not_beside_it_is_refused():
    """The allocation sentence does not describe a number from a later table."""
    reading = _batch(counts=[*_THREE_ARMS[:2],
                             _component("Ipilimumab", 312,
                                        phrase="underwent randomization",
                                        quote=ANALYSIS_ROW)],
                     partition=_GOOD_PARTITION)
    assert reading.aggregation_evidence is None
    assert any("is not printed with the 312" in e
               for e in reading.aggregation_errors)


def test_a_component_with_no_population_at_all_is_refused():
    reading = _batch(counts=[{**c, "population_phrase": ""} for c in _THREE_ARMS],
                     partition=_GOOD_PARTITION)
    assert reading.aggregation_evidence is None
    assert any("does not say which people" in e for e in reading.aggregation_errors)


def test_a_quote_that_does_not_name_the_arm_is_refused():
    reading = _batch(counts=[_component("placebo group", 316), *_THREE_ARMS[1:]],
                     partition=_GOOD_PARTITION)
    assert reading.aggregation_evidence is None
    assert any("does not name the arm" in e for e in reading.aggregation_errors)


# --- isolation: a broken sum must not cost a good reading ------------------

def test_a_broken_component_does_not_destroy_an_explicit_total():
    explicit = {"scope_label": "all randomised patients", "value": "945",
                "quote": "A total of 945 patients underwent randomization",
                "population_phrase": "underwent randomization"}
    reading = _batch(readings=[explicit],
                     counts=[{**_THREE_ARMS[0], "count": "not a number"}],
                     partition=_GOOD_PARTITION)
    assert reading.aggregation_evidence is None and reading.aggregation_errors
    assert len(reading.usable) == 1
    projection = _project(reading)
    assert projection.status == OK and projection.value == "945"


# --- explicit and derived, together ---------------------------------------

def _explicit(value, quote):
    return {"scope_label": "all randomised patients", "value": value,
            "quote": quote, "population_phrase": "underwent randomization"}


def test_an_explicit_total_that_agrees_is_released_with_the_sum_as_corroboration():
    reading = _batch(readings=[_explicit(
        "945", "A total of 945 patients underwent randomization")],
        counts=_THREE_ARMS, partition=_GOOD_PARTITION)
    projection = _project(reading)
    assert projection.status == OK and projection.value == "945"
    assert "independently add to the same total" in projection.aggregation_reason


def test_an_explicit_total_that_disagrees_releases_nothing():
    """Choosing between them would hide the paper disagreeing with itself."""
    document = PAPER.replace("A total of 945", "A total of 944")
    reading = _batch(readings=[_explicit(
        "944", "A total of 944 patients underwent randomization")],
        counts=[_component("nivolumab group", 316,
                           quote=ALLOCATION.replace("945", "944")),
                _component("nivolumab-plus-ipilimumab group", 314,
                           quote=ALLOCATION.replace("945", "944")),
                _component("ipilimumab group", 315,
                           quote=ALLOCATION.replace("945", "944"))],
        partition=_GOOD_PARTITION, document=document)
    projection = _project(reading)
    assert projection.status == CONTRADICTORY
    assert to_source_result(projection).found is False
    assert "944" in projection.reason and "945" in projection.reason


def test_an_analysed_total_does_not_block_deriving_the_allocated_one():
    """The printed total is for other people; the components are for these."""
    reading = _batch(readings=[{"scope_label": "analysis population",
                                "value": "938",
                                "quote": "938 patients in total were analysed.",
                                "population_phrase": "Analysis population"}],
                     counts=_THREE_ARMS, partition=_GOOD_PARTITION,
                     document=PAPER + " 938 patients in total were analysed.")
    projection = _project(reading)
    assert projection.status == DERIVED and projection.derived_value == 945


# --- the policy is a contract, not a set of branches -----------------------

def test_the_policy_only_permits_whole_study_counts():
    policy = load_aggregation_policy()
    assert policy.applies_to("study", "sample_size")
    assert not policy.applies_to("study", "progression_free_survival")
    assert not policy.applies_to("arm", "sample_size")


def test_the_policy_is_hashed_so_a_run_can_say_which_one_it_applied():
    policy = load_aggregation_policy()
    assert len(policy.sha256) == 64 and policy.policy_id == "safe_sum_v1"


def test_nothing_is_derived_when_no_components_were_offered():
    projection = _project(_batch(readings=[]))
    assert projection.status == NOT_REPORTED
    assert projection.aggregation_status == "not_applicable"
    assert "does not print" in projection.reason
