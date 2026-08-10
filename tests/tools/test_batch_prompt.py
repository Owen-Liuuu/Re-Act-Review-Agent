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
    assert {"arm_label", "value", "quote", "population_phrase", "population_quote",
            "timepoint_phrase", "timepoint_quote", "effect_definition",
            "value_components"} <= set(reading)


def test_evidence_for_a_population_must_be_a_passage_not_an_assertion():
    text = _prompt()
    assert "the paper's OWN words" in _flat(text)
    assert "one contiguous verbatim passage" in _flat(text)
    # An honest empty answer is named as acceptable, so the model is not pushed
    # into inventing a population it did not read.
    assert "an empty field\n  is an honest answer" in text


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
