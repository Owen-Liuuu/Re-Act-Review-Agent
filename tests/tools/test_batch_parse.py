"""One bad reading must not destroy the good ones.

The Phase 7 parser returned an empty list the moment any entry failed. For one
question that was fine — the single answer was unusable. For a batch it means a
single malformed line fails every claim in the group, which is a failure mode
that grows with the batch it was meant to make safe.

The MA004 case is here too, in the form it will actually arrive: the same arm,
read twice, once allocated and once analysed.
"""
from __future__ import annotations

import pytest

from react_review.schemas.batch import ARM, COMPARISON, STUDY
from react_review.tools.batch_parse import parse_batch, parse_one_entry

PAPER = (
    "A total of 945 patients underwent randomization: 316 patients were assigned "
    "to the nivolumab group, 314 to the nivolumab-plus-ipilimumab group, and 315 "
    "to the ipilimumab group.\n\n"
    "Efficacy The median progression-free survival was 6.9 months (95% CI, 4.3 to "
    "9.5) in the nivolumab group, 11.5 months (95% CI, 8.9 to 16.7) in the "
    "nivolumab-plus-ipilimumab group.\n\n"
    "Table 3. Analysis population. Nivolumab plus Ipilimumab (N = 313)")

ALLOCATION = ("314 to the nivolumab-plus-ipilimumab group")
ANALYSIS = "Nivolumab plus Ipilimumab (N = 313)"


def _reading(**kw):
    body = {"arm_label": "nivolumab-plus-ipilimumab group", "value": "314",
            "unit": "count", "quote": ALLOCATION}
    body.update(kw)
    return body


# --- the case the whole phase is for -------------------------------------

def test_one_arm_read_twice_survives_as_two_readings():
    """314 allocated and 313 analysed: one arm, two readings, both kept."""
    batch = parse_batch({"readings": [
        _reading(value="314", quote=ALLOCATION,
                 population_phrase="underwent randomization",
                 population_quote="A total of 945 patients underwent randomization: "
                                  "316 patients were assigned to the nivolumab group, "
                                  "314 to the nivolumab-plus-ipilimumab group"),
        _reading(arm_label="Nivolumab plus Ipilimumab", value="313", quote=ANALYSIS,
                 population_phrase="Analysis population"),
    ]}, PAPER)

    assert batch.batch_error == "" and len(batch.usable) == 2
    first, second = batch.entries
    assert first.identity.target() == second.identity.target()      # one arm
    assert first.selection_key() != second.selection_key()          # two readings
    assert first.identity.population.basis == "allocated"
    assert second.identity.population.basis == "analysed"


def test_the_allocation_sentence_cannot_be_borrowed_for_the_analysis_number():
    """Both halves are real text; only their relationship would be invented."""
    batch = parse_batch({"readings": [
        _reading(arm_label="Nivolumab plus Ipilimumab", value="313", quote=ANALYSIS,
                 population_phrase="underwent randomization"),
    ]}, PAPER)
    entry = batch.entries[0]
    # The reading survives — the number and its quote are sound — but the
    # population it tried to borrow does not stick to it.
    assert entry.usable
    assert entry.identity.population.basis != "allocated"
    assert entry.population_anchor is None


# --- isolation levels -----------------------------------------------------

def test_a_response_that_is_not_a_batch_fails_as_one_thing():
    for raw in ({"arms": []}, {"readings": "none"}, ["a list"]):
        batch = parse_batch(raw, PAPER)
        assert batch.batch_error and batch.entries == []


def test_one_unusable_reading_does_not_take_the_others_with_it():
    """The Phase 7 parser returned [] here and failed every claim in the group."""
    batch = parse_batch({"readings": [
        _reading(value="316", arm_label="nivolumab group",
                 quote="316 patients were assigned to the nivolumab group"),
        _reading(value="999", quote="a sentence the paper does not contain"),
        _reading(value="315", arm_label="ipilimumab group",
                 quote="315 to the ipilimumab group"),
    ]}, PAPER)

    assert [e.value for e in batch.usable] == ["316", "315"]
    assert len(batch.rejected) == 1
    assert batch.rejected[0]["index"] == 1
    assert "contiguous passage" in batch.rejected[0]["reason"]


@pytest.mark.parametrize("bad,expected", [
    ({"value": None}, "no value"),
    ({"quote": ""}, "no supporting quote"),
    ({"value": "777"}, "does not contain the value"),
    ({"arm_label": ""}, "names no arm"),
    ({"arm_label": "ipilimumab group"}, "does not name the arm"),
    ({"value_components": {"ci_level": "high"}}, "not a number"),
])
def test_each_way_a_single_reading_can_be_refused(bad, expected):
    entry, reason = parse_one_entry(_reading(**bad), 0, PAPER, target_shape=ARM)
    assert entry is None and expected in reason


def test_a_reading_that_is_not_an_object_is_refused_alone():
    batch = parse_batch({"readings": ["not an object", _reading()]}, PAPER)
    assert len(batch.usable) == 1 and len(batch.rejected) == 1


# --- shapes ---------------------------------------------------------------

def test_a_comparison_reading_must_name_both_sides_in_its_own_quote():
    quote = ("The median progression-free survival was 6.9 months (95% CI, 4.3 to "
             "9.5) in the nivolumab group, 11.5 months (95% CI, 8.9 to 16.7) in "
             "the nivolumab-plus-ipilimumab group")
    good, reason = parse_one_entry(
        {"left_label": "nivolumab-plus-ipilimumab group",
         "right_label": "nivolumab group", "value": "11.5 months (95% CI, 8.9 to 16.7)",
         "quote": quote}, 0, PAPER, target_shape=COMPARISON)
    assert good is not None and reason == ""
    assert good.identity.comparison_pair == ("nivolumab-plus-ipilimumab group",
                                             "nivolumab group")

    missing, reason = parse_one_entry(
        {"left_label": "nivolumab group", "right_label": "ipilimumab group",
         "value": "6.9 months (95% CI, 4.3 to 9.5)", "quote": quote},
        0, PAPER, target_shape=COMPARISON)
    assert missing is None and "ipilimumab group" in reason


def test_a_study_reading_needs_no_arm():
    entry, reason = parse_one_entry(
        {"scope_label": "all randomised patients", "value": "945",
         "quote": "A total of 945 patients underwent randomization",
         "population_phrase": "underwent randomization"},
        0, PAPER, target_shape=STUDY)
    assert entry is not None and reason == ""
    assert entry.identity.target_shape == STUDY
    assert entry.identity.population.basis == "allocated"


# --- axes that do not bind stay unstated ---------------------------------

def test_a_timepoint_the_paper_did_not_print_here_is_dropped_not_kept():
    entry, _ = parse_one_entry(
        _reading(timepoint_phrase="at 5 years"), 0, PAPER, target_shape=ARM)
    assert entry is not None
    assert entry.identity.timepoint_phrase == ""      # not borrowed, not invented
    assert entry.timepoint_anchor is None


def test_a_timepoint_inside_the_quote_is_kept_with_its_anchor():
    quote = ("The median progression-free survival was 6.9 months (95% CI, 4.3 to "
             "9.5) in the nivolumab group")
    entry, _ = parse_one_entry(
        {"arm_label": "nivolumab group", "value": "6.9 months (95% CI, 4.3 to 9.5)",
         "quote": quote, "timepoint_phrase": "median progression-free survival"},
        0, PAPER, target_shape=ARM)
    assert entry.identity.timepoint_phrase == "median progression-free survival"
    assert entry.timepoint_anchor is not None


def test_components_travel_with_the_reading_that_owns_them():
    quote = ("The median progression-free survival was 6.9 months (95% CI, 4.3 to "
             "9.5) in the nivolumab group")
    entry, _ = parse_one_entry(
        {"arm_label": "nivolumab group", "value": "6.9 months (95% CI, 4.3 to 9.5)",
         "quote": quote,
         "value_components": {"point_estimate": 6.9, "ci_level": 95,
                              "ci_lower": 4.3, "ci_upper": 9.5}},
        0, PAPER, target_shape=ARM)
    assert entry.components["ci_level"] == 95.0
