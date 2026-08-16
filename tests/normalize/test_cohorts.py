"""Cohorts discovered from the review — and never quietly merged.

The cases below are the failure this module replaces: on a review with Treatment
and Placebo arms the predecessor mapped BOTH to ``all``, their claims collided on
the audit's join key, and one arm's value was compared against the other's
evidence with no error and no flag.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.normalize.cohorts import (
    build_cohort_registry,
    load_aliases,
    slug,
)

ALIASES = load_aliases(Path(__file__).resolve().parents[2] / "configs" /
                       "cohort_aliases.json")


def _keys(labels: list[str], aliases=None) -> list[str]:
    return [c.key for c in build_cohort_registry(labels, aliases=aliases).labels]


# --- the failures this module exists for ---

def test_treatment_and_placebo_are_two_cohorts_not_one():
    assert _keys(["Treatment", "Placebo"], ALIASES) == ["treatment", "placebo"]


def test_long_arm_names_are_two_cohorts_not_one():
    # The old keyword list matched "patient" inside both of these and made them
    # the same cohort.
    keys = _keys(["Patients on nivolumab", "Patients on chemotherapy"], ALIASES)
    assert len(set(keys)) == 2


@pytest.mark.parametrize("labels", [
    ["Survivors", "Non-survivors"],
    ["Arm A", "Arm B", "Arm C"],
    ["Sepsis patients", "Standard care"],
    ["Intervention", "Usual care"],
])
def test_arms_of_other_domains_stay_distinct(labels):
    assert len(set(_keys(labels, ALIASES))) == len(labels)


def test_an_unplaceable_label_is_unknown_and_never_all():
    registry = build_cohort_registry(["Treatment", "Placebo"], aliases=ALIASES)
    res = registry.resolve("Some other arm")
    assert res.status == "unknown"
    assert res.key == "" and res.key != "all"     # the old code returned "all"
    assert res.reason


# --- the three group values must stay distinct ---

def test_blank_combined_and_unknown_are_not_the_same_thing():
    registry = build_cohort_registry(["T1DM", "Control"], aliases=ALIASES)
    blank = registry.resolve("")            # the table did not split this value
    combined = registry.resolve("Overall")  # the review says "all participants"
    unknown = registry.resolve("Cohort Q")  # a label that fits nothing

    assert blank.status == "combined" and blank.key == "all"
    assert combined.status == "combined" and combined.key == "all"
    assert unknown.status == "unknown" and unknown.key == ""
    assert not unknown.known                # must never pass as a cohort


# --- benchmark compatibility comes from data, not from code ---

def test_benchmark_labels_keep_their_answer_key_ids():
    assert _keys(["T1DM", "Control"], ALIASES) == ["t1dm", "control"]
    assert _keys(["Diabetic children", "Healthy controls"], ALIASES) == \
        ["t1dm", "control"]


def test_without_the_alias_file_the_same_labels_are_still_two_cohorts():
    # The alias file only RE-KEYS; remove it and the cohorts are still distinct.
    assert len(set(_keys(["T1DM", "Control"]))) == 2


def test_aliases_never_introduce_a_cohort_the_review_did_not_mention():
    registry = build_cohort_registry(["Treatment", "Placebo"], aliases=ALIASES)
    assert registry.by_key("t1dm") is None and registry.by_key("control") is None


def test_documentation_keys_in_the_alias_file_are_not_cohorts():
    assert not any(k.startswith("_") for k in ALIASES)


# --- mechanics ---

def test_variants_of_one_cohort_merge_without_losing_the_reviews_words():
    registry = build_cohort_registry(
        ["Diabetic children", "T1DM", "Diabetic children"], aliases=ALIASES)
    assert len(registry.labels) == 1
    cohort = registry.labels[0]
    assert cohort.key == "t1dm"
    assert set(cohort.raw_variants) == {"Diabetic children", "T1DM"}


def test_word_boundaries_stop_short_tokens_matching_inside_words():
    # "dm" must not match inside "Admission"; that class of accident is why the
    # old substring keyword lists misfired.
    assert _keys(["Admission group", "Discharge group"], ALIASES) == \
        ["admission_group", "discharge_group"]


def test_slug_is_stable_and_readable():
    assert slug("Patients on nivolumab") == "patients_on_nivolumab"
    assert slug("Non-survivors") == "non_survivors"


# --- the fixed-vocabulary mapper must not come back ---

def test_the_parser_no_longer_uses_the_legacy_group_mapper():
    """The removed mapper folded unplaceable labels into a catch-all cohort.

    The module is gone, so an import would already fail; this pins the name so a
    reintroduction has to be deliberate rather than a copy-paste back into the
    parser.
    """
    source = (Path(__file__).resolve().parents[2] / "src" / "react_review" /
              "parser" / "review_parser.py").read_text(encoding="utf-8")
    assert "normalize_group" not in source
