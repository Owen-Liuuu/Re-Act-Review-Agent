"""Deterministic arm discovery from column-header geometry."""
from __future__ import annotations

from react_review.parser.cohort_headers import arm_labels_from_headers


def test_forest_and_n_columns_yield_mie_and_oe():
    labels = arm_labels_from_headers([
        "Study",
        "Country",
        "Age",
        "N MIE",
        "N OE",
        "Minimally Invasive Esophagectomy (MIE) Events",
        "Minimally Invasive Esophagectomy (MIE) Total",
        "Open Esophagectomy (OE) Events",
        "Open Esophagectomy (OE) Total",
    ])
    assert labels == ["MIE", "OE"]


def test_no_shared_measure_discovers_nothing():
    assert arm_labels_from_headers([
        "Study", "Country", "Year", "Design", "Age",
    ]) == []


def test_bare_events_and_total_are_not_arms():
    """Forest 2b trap: measure words themselves must not become cohorts."""
    assert arm_labels_from_headers([
        "Study or Subgroup", "Events", "Total", "Events", "Total",
    ]) == []


def test_one_arm_with_two_measures_is_not_a_contrast():
    assert arm_labels_from_headers([
        "MIE Events", "MIE Total",
    ]) == []


def test_does_not_use_disease_vocabulary():
    labels = arm_labels_from_headers([
        "T1DM Events", "Control Events", "T1DM Total", "Control Total",
    ])
    assert labels == ["T1DM", "Control"]
