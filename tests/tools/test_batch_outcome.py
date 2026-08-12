"""What a reviewer is told, and what the audit is entitled to claim.

`MISSING_SOURCE` reads as "the paper was retrieved and does not say this —
possible fabrication". It is an accusation, and the audit may only make it when
the paper really was read and really is silent. Every other failure of a batch
is a fact about the RESPONSE, and saying so is the difference between reporting
an extractor fault and accusing a reviewer of inventing a number.
"""
from __future__ import annotations

from react_review.core.enums import CollectionOutcome
from react_review.normalize.population import PopulationScope
from react_review.schemas.batch import STUDY
from react_review.tools.batch_outcome import outcome_for
from react_review.tools.batch_parse import parse_batch
from react_review.tools.batch_project import project_claim

REVIEW = {"nivolumab_plus_placebo": "Nivolumab (3 mg/kg) + placebo"}
DOCUMENT = ("A total of 945 patients underwent randomization: 316 patients were "
            "assigned to the nivolumab group, 314 to the nivolumab-plus-"
            "ipilimumab group, and 315 to the ipilimumab group.")


def _read(payload, *, shape="arm", aggregable=False):
    return parse_batch(payload, DOCUMENT, target_shape=shape, aggregable=aggregable)


def _project(reading, **kw):
    body = dict(target_shape="arm", review_labels=REVIEW,
                cohort_key="nivolumab_plus_placebo")
    body.update(kw)
    return project_claim(reading, **body)


def _outcome(payload, **kw):
    reading = _read(payload, **{k: v for k, v in kw.items()
                                if k in {"shape", "aggregable"}})
    projection = _project(reading, **{k: v for k, v in kw.items()
                                      if k not in {"shape", "aggregable"}})
    return outcome_for(projection, reading), projection


# --- 1. released first ------------------------------------------------------

def test_a_released_reading_is_found():
    outcome, projection = _outcome({"readings": [
        {"arm_label": "nivolumab group", "value": "316", "quote": DOCUMENT,
         "population_phrase": "underwent randomization"}]})
    assert projection.released and outcome is CollectionOutcome.FOUND


def test_a_derived_total_is_found_too():
    """A computed number is an answer. Whether it is a PRINTED one is a
    different question, and `value_origin` is where that is recorded."""
    partition = ("Patients were randomly assigned in a 1:1:1 ratio to one of "
                 "three groups.")
    document = DOCUMENT + "\n\n" + partition
    reading = parse_batch({"readings": [], "aggregation_sets": [{
        "population_phrase": "underwent randomization",
        "population_quote": DOCUMENT,
        "cohort_counts": [
            {"arm_label": "nivolumab group", "count": 316, "quote": DOCUMENT},
            {"arm_label": "nivolumab-plus-ipilimumab group", "count": 314,
             "quote": DOCUMENT},
            {"arm_label": "ipilimumab group", "count": 315, "quote": DOCUMENT}],
        "partition": {"complete": True, "mutually_exclusive": True,
                      "quote": partition, "declared_arm_count": 3}}]},
        document, target_shape=STUDY, aggregable=True)
    projection = project_claim(
        reading, target_shape=STUDY, field_type="sample_size",
        requested_scope=PopulationScope(basis="allocated"),
        required_axes=["population_basis"])
    assert outcome_for(projection, reading) is CollectionOutcome.FOUND


# --- 2 and 3. a broken response is never an accusation ---------------------

def test_a_response_that_is_not_a_batch_is_an_extraction_failure():
    outcome, _ = _outcome({"nope": 1})
    assert outcome is CollectionOutcome.EXTRACTION_FAILED


def test_entries_that_all_fail_their_checks_are_an_extraction_failure():
    """The load-bearing row.

    The model returned readings and every one failed a deterministic check.
    Reporting that as MISSING_SOURCE would have the audit assert, with no
    evidence at all, that the paper is silent.
    """
    outcome, _ = _outcome({"readings": [
        {"arm_label": "nivolumab group", "value": "316",
         "quote": "a sentence this paper does not contain"},
        {"arm_label": "ipilimumab group", "value": "315", "quote": ""}]})
    assert outcome is CollectionOutcome.EXTRACTION_FAILED
    assert outcome is not CollectionOutcome.MISSING_SOURCE


# --- 4. the one state that may say the paper is silent ---------------------

def test_an_empty_response_with_a_stated_reason_is_a_missing_source():
    outcome, _ = _outcome({
        "readings": [],
        "nothing_reported_reason": "the paper reports no cohort sizes"})
    assert outcome is CollectionOutcome.MISSING_SOURCE


def test_an_empty_response_with_no_reason_is_not():
    """A response that failed to answer is not a paper that stays silent."""
    outcome, _ = _outcome({"readings": []})
    assert outcome is CollectionOutcome.EXTRACTION_UNRESOLVED


def test_an_empty_list_beside_a_rejected_entry_is_not_silence_either():
    outcome, _ = _outcome({
        "readings": [{"arm_label": "x", "value": "1", "quote": "not in the paper"}],
        "nothing_reported_reason": "nothing else is reported"})
    assert outcome is CollectionOutcome.EXTRACTION_FAILED


# --- 5. everything else ----------------------------------------------------

def test_an_arm_the_batch_never_reported_is_unresolved_not_missing():
    """The paper may well state it; this reading did not pin it down."""
    outcome, _ = _outcome(
        {"readings": [{"arm_label": "ipilimumab group", "value": "315",
                       "quote": DOCUMENT,
                       "population_phrase": "underwent randomization"}]},
        review_labels={"nivolumab_plus_placebo": "Nivolumab (3 mg/kg) + placebo",
                       "ipilimumab_plus_placebo": "Ipilimumab (3 mg/kg) + placebo"},
        cohort_key="nivolumab_plus_placebo")
    assert outcome is CollectionOutcome.EXTRACTION_UNRESOLVED


def test_a_contradiction_is_unresolved():
    outcome, projection = _outcome({"readings": [
        {"arm_label": "nivolumab group", "value": "316", "quote": DOCUMENT,
         "population_phrase": "underwent randomization"},
        {"arm_label": "nivolumab group", "value": "315", "quote": DOCUMENT,
         "population_phrase": "underwent randomization"}]})
    assert projection.status == "contradictory"
    assert outcome is CollectionOutcome.EXTRACTION_UNRESOLVED


# --- the reviewer is told which of these it is -----------------------------

def test_the_judge_has_words_for_both_new_outcomes():
    """Without them a batch failure would arrive as the generic unmatched flag."""
    from react_review.orchestrator.judge import _OUTCOME_FLAG

    for outcome in (CollectionOutcome.EXTRACTION_FAILED,
                    CollectionOutcome.EXTRACTION_UNRESOLVED):
        label, reason = _OUTCOME_FLAG[outcome]
        assert label and reason
    failed = _OUTCOME_FLAG[CollectionOutcome.EXTRACTION_FAILED][1]
    assert "says nothing about whether the paper states the value" in failed
