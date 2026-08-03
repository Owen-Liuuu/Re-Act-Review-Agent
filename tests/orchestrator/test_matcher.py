"""Tests for the review↔source match-table join."""
from __future__ import annotations

from react_review.orchestrator.matcher import build_pairs, match_key
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem


def _rv(study, group, ft, value="1", *, table_id="", cell_ref=None):
    return ReviewDataItem(study_id=study, group=group, field_type=ft, value=value,
                          table_id=table_id, cell_ref=cell_ref)


def _sv(study, group, ft, value="1", *, table_id="", cell_ref=None):
    return SourceEvidenceItem(study_id=study, group=group, field_type=ft,
                              source_value=value, table_id=table_id, cell_ref=cell_ref)


def test_key_is_normalized():
    assert match_key("Ahmad_2022", " T1DM ", "Single", "EAT_thickness") == (
        "ahmad_2022", "t1dm", "single", "eat_thickness"
    )


def test_exact_pairing():
    review = [_rv("s1", "t1dm", "bmi"), _rv("s1", "control", "bmi")]
    source = [_sv("s1", "control", "bmi"), _sv("s1", "t1dm", "bmi")]
    pairs, ur, us = build_pairs(review, source)
    assert len(pairs) == 2 and not ur and not us
    # each review paired with the SAME group's source
    for r, s in pairs:
        assert r.group == s.group


def test_unmatched_review_and_source():
    review = [_rv("s1", "t1dm", "bmi"), _rv("s1", "t1dm", "age")]
    source = [_sv("s1", "t1dm", "bmi"), _sv("s1", "t1dm", "eat_thickness")]
    pairs, ur, us = build_pairs(review, source)
    assert len(pairs) == 1
    assert [r.field_type for r in ur] == ["age"]
    assert [s.field_type for s in us] == ["eat_thickness"]


def test_duplicate_key_without_cells_is_refused_not_guessed():
    # Two claims share a key and neither says which cell it came from. Pairing
    # one of them by arrival order would attribute evidence to the wrong row and
    # nothing downstream could tell. Refusing is the finding.
    review = [_rv("s1", "t1dm", "bmi"), _rv("s1", "t1dm", "bmi")]
    source = [_sv("s1", "t1dm", "bmi")]
    pairs, ur, us = build_pairs(review, source)

    assert pairs == []
    assert [u.reason_code for u in ur] == ["ambiguous_match_key"] * 2
    assert "refusing to guess" in ur[0].message
    assert [u.reason_code for u in us] == ["unclaimed_source"]


def test_duplicate_key_is_paired_when_both_sides_name_the_same_cell():
    review = [_rv("s1", "t1dm", "bmi", "24", table_id="t1", cell_ref=(0, 3)),
              _rv("s1", "t1dm", "bmi", "25", table_id="t1", cell_ref=(1, 3))]
    source = [_sv("s1", "t1dm", "bmi", "25", table_id="t1", cell_ref=(1, 3)),
              _sv("s1", "t1dm", "bmi", "24", table_id="t1", cell_ref=(0, 3))]
    pairs, ur, us = build_pairs(review, source)

    assert len(pairs) == 2 and not ur and not us
    for r, s in pairs:                       # paired by cell, not by order
        assert r.value == s.source_value


def test_same_coordinates_in_different_tables_do_not_collide():
    # (0,3) exists in every table, so the table id has to be part of the cell id.
    review = [_rv("s1", "t1dm", "bmi", "24", table_id="t1", cell_ref=(0, 3)),
              _rv("s1", "t1dm", "bmi", "99", table_id="t2", cell_ref=(0, 3))]
    source = [_sv("s1", "t1dm", "bmi", "99", table_id="t2", cell_ref=(0, 3)),
              _sv("s1", "t1dm", "bmi", "24", table_id="t1", cell_ref=(0, 3))]
    pairs, ur, us = build_pairs(review, source)

    assert len(pairs) == 2 and not ur and not us
    assert {(r.table_id, r.value) for r, _ in pairs} == {("t1", "24"), ("t2", "99")}


def test_checklist_id_disambiguates_claims_without_table_cells():
    review = [
        ReviewDataItem(study_id="s1", group="-", field_type="effect_size",
                       value="0.4", origin="checklist", checklist_id="primary_effect"),
        ReviewDataItem(study_id="s1", group="-", field_type="effect_size",
                       value="0.2", origin="checklist", checklist_id="secondary_effect"),
    ]
    source = [
        SourceEvidenceItem(study_id="s1", group="-", field_type="effect_size",
                           source_value="0.2", checklist_id="secondary_effect"),
        SourceEvidenceItem(study_id="s1", group="-", field_type="effect_size",
                           source_value="0.4", checklist_id="primary_effect"),
    ]
    pairs, ur, us = build_pairs(review, source)
    assert len(pairs) == 2 and not ur and not us
    assert all(r.checklist_id == s.checklist_id for r, s in pairs)
    assert all(r.value == s.source_value for r, s in pairs)


def test_partial_cell_information_still_refuses():
    # One side knows its cell, the other does not: not enough to pair safely.
    review = [_rv("s1", "t1dm", "bmi", table_id="t1", cell_ref=(0, 3)),
              _rv("s1", "t1dm", "bmi", table_id="t1", cell_ref=(1, 3))]
    source = [_sv("s1", "t1dm", "bmi"), _sv("s1", "t1dm", "bmi")]
    pairs, ur, us = build_pairs(review, source)

    assert pairs == []
    assert all(u.reason_code == "ambiguous_match_key" for u in ur)


def test_timepoint_separates_otherwise_identical_claims():
    review = [ReviewDataItem(study_id="s1", group="t1dm", field_type="bmi",
                             timepoint="baseline", value="24"),
              ReviewDataItem(study_id="s1", group="t1dm", field_type="bmi",
                             timepoint="12m", value="26")]
    source = [SourceEvidenceItem(study_id="s1", group="t1dm", field_type="bmi",
                                 timepoint="12m", source_value="26"),
              SourceEvidenceItem(study_id="s1", group="t1dm", field_type="bmi",
                                 timepoint="baseline", source_value="24")]
    pairs, ur, us = build_pairs(review, source)

    assert len(pairs) == 2 and not ur and not us
    for r, s in pairs:
        assert r.timepoint == s.timepoint


def test_unmatched_claim_carries_a_reason():
    pairs, ur, us = build_pairs([_rv("s1", "t1dm", "age")], [])
    assert ur[0].reason_code == "no_source_evidence"
    assert ur[0].key_text == "s1/t1dm/single/age"
