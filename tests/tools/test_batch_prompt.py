"""The v5 batch contract asks for every reading, and only its own shape.

Two failures are cheap to make and expensive to find later: a prompt that lets
the model pick one reading (which is the Phase 7 behaviour under a new name),
and a prompt whose JSON example is malformed (which teaches the model to answer
malformed JSON). Both are checked here, offline.
"""
from __future__ import annotations

import json

import pytest

from react_review.schemas.batch import ARM, COMPARISON, STUDY
from react_review.tools.batch_prompt import BATCH_PROFILE, build_batch_prompt
from react_review.tools.extraction_profile import BATCH_V5, prompt_version


def _flat(text: str) -> str:
    """Line wrapping is not contract: compare the words, not the layout."""
    return " ".join(text.split())


def _prompt(shape=ARM, **kw):
    body = dict(target_shape=shape, context="melanoma trials",
                concept="cohort size", raw_label="Intervention arm, n",
                field_type="cohort_n", concept_variants="n, number of patients",
                unit_hint="count", paper_text="PAPER")
    body.update(kw)
    return build_batch_prompt(**body)


def test_the_contract_is_registered_as_its_own_profile():
    assert prompt_version(BATCH_PROFILE) == BATCH_V5
    assert BATCH_V5 not in ("extract-source-v3-scoped-cohort-counts",
                            "extract-source-v4-targeted-components")


def test_the_model_is_told_not_to_choose():
    """Choosing is what the deterministic assignment does, from checkable evidence."""
    text = _prompt()
    assert "Do NOT choose one" in text
    assert "EVERY reading" in text


def test_the_same_arm_twice_is_explicitly_required():
    """A contract that reads a repeated arm as a duplicate cannot express MA004."""
    text = _prompt(ARM)
    assert "TWO readings of ONE arm" in _flat(text)
    assert "never drop one because it looks like a duplicate" in _flat(text)


@pytest.mark.parametrize("shape,wanted,unwanted", [
    (ARM, "arm_label", "left_label"),
    (COMPARISON, "left_label", "arm_label"),
    (STUDY, "scope_label", "left_label"),
])
def test_each_shape_asks_only_for_its_own_shape(shape, wanted, unwanted):
    """Asking every batch for arms AND comparisons grows with the square of the arms."""
    text = _prompt(shape)
    assert wanted in text and unwanted not in text


def test_the_output_skeleton_is_valid_json():
    text = _prompt()
    body = text[text.index('{"readings"'):]
    parsed = json.loads(body)
    reading = parsed["readings"][0]
    assert {"arm_label", "value", "quote", "population_phrase",
            "timepoint_phrase", "timepoint_quote", "effect_definition",
            "value_components"} <= set(reading)
    # There is deliberately no `population_quote`. A separate passage could never
    # bind a population to a value that the document search had not already
    # bound, so asking for one only invited a phrase from somewhere else.
    assert "population_quote" not in reading


def test_evidence_for_a_population_must_be_a_passage_not_an_assertion():
    text = _prompt()
    assert "the paper's OWN words" in _flat(text)
    assert "one contiguous verbatim passage" in _flat(text)
    # The population must come from beside the value, and an honest empty answer
    # is named as acceptable, so the model is not pushed into inventing one.
    assert "taken from beside the value itself" in _flat(text)
    assert "an empty field is an honest answer" in _flat(text)


def test_the_timepoint_is_asked_for_when_the_review_declares_one():
    without = _prompt()
    with_timepoint = _prompt(timepoint_label="median PFS")
    assert "median PFS" in with_timepoint and "median PFS" not in without
    assert "a different number" in with_timepoint


def test_arithmetic_is_refused_in_the_study_shape():
    """A total the paper does not print is not a total it reported."""
    text = _prompt(STUDY)
    assert "rather than adding the arms up" in _flat(text)
    assert "Arithmetic is not reading" in _flat(text)


def test_an_unknown_shape_is_refused():
    with pytest.raises(ValueError, match="unknown target shape"):
        _prompt("paragraph")


# --- the aggregation ask is narrow, and forbids the model to do the sum ----

def _study(field_type: str) -> str:
    return build_batch_prompt(
        target_shape=STUDY, context="melanoma trials", concept="sample size",
        raw_label="N", field_type=field_type, concept_variants="N, total",
        unit_hint="patients", paper_text="PAPER")


def test_a_whole_study_count_is_asked_for_its_arms_as_components():
    text = _flat(_study("sample_size"))
    assert "cohort_counts" in text and "partition" in text
    assert "NOT in ``readings``" in text


def test_no_other_study_field_may_offer_components():
    """Means, rates and hazard ratios do not partition, so nothing may add them."""
    text = _study("progression_free_survival")
    assert "cohort_counts" not in text and "partition" not in text
    assert "Arithmetic is not reading" in text


def test_the_model_is_forbidden_to_compute_the_total_itself():
    text = _flat(_study("sample_size"))
    assert "You must not add them up" in text
    assert "Do not report a sum" in text


def test_an_unanchored_partition_claim_is_named_as_worthless_in_the_prompt():
    """The rule the parser enforces is stated where the model can act on it."""
    text = _flat(_study("sample_size"))
    assert "A true without a locatable quote counts as a false" in text
    assert "If you are not sure, answer false" in text


def test_each_population_is_asked_for_as_its_own_set():
    """Mixing is prevented by the shape of the answer, not by an instruction."""
    text = _flat(_study("sample_size"))
    assert "One set per population, per timepoint" in text
    assert "aggregation_sets" in text and "NOT in ``readings``" in text
    assert "three populations, return three sets" in text


def test_the_arm_census_is_asked_for_and_its_purpose_stated():
    """`complete` about a list the code cannot see is worth nothing without it."""
    text = _flat(_study("sample_size"))
    assert "declared_arm_count" in text and "declared_arm_labels" in text
    assert "what make ``complete`` worth anything" in text
    assert "never from your own count of the arms" in text


def test_the_booleans_are_asked_for_as_booleans():
    text = _flat(_study("sample_size"))
    assert 'not the strings "true"/"false", not 0 or 1' in text


def test_the_aggregation_skeleton_is_valid_json():
    text = _study("sample_size")
    body = text[text.index('{"readings"'):]
    parsed = json.loads(body)
    one = parsed["aggregation_sets"][0]
    assert {"population_phrase", "population_quote", "timepoint_phrase",
            "cohort_counts", "partition"} <= set(one)
    assert {"complete", "mutually_exclusive", "quote", "declared_arm_count",
            "declared_arm_labels"} <= set(one["partition"])
    assert one["partition"]["complete"] is False
    # A component names an arm and a number, and nothing about people: the set
    # owns the population, so a component cannot contradict the set it is in.
    assert set(one["cohort_counts"][0]) == {"arm_label", "count", "quote"}


def test_the_prompt_states_the_binding_the_parser_enforces():
    """A prompt that omits a rule the parser applies buys nothing but refusals."""
    text = _flat(_study("sample_size"))
    assert "must itself carry this set's population words" in text
    assert "cannot go in a set whose population is" in text


def test_the_prompt_says_a_ratio_is_not_a_count_of_groups():
    text = _flat(_study("sample_size"))
    assert "2:1:1 ratio to one of three groups" in text
    assert "declares three, not two" in text
    assert "not a number that merely appears in it" in text


def test_the_prompt_requires_the_partition_to_be_about_this_population():
    text = _flat(_study("sample_size"))
    assert "establishes nothing about the randomised one" in text
