from __future__ import annotations

import pytest

from react_review.claim_ids import (
    assign_claim_ids,
    claim_index,
    global_claim_index,
    study_letter,
    validate_claim_ids,
)
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.table import CapturedTable, CapturedTableSet


def _item(study: str, table: str, row: int, col: int, *, existing: str = ""):
    return ReviewDataItem(
        review_data_id=existing,
        study_id=study,
        field_type="measure",
        value=str(row * 10 + col),
        table_id=table,
        cell_ref=(row, col),
        column_header=f"column_{col}",
    )


def test_study_letters_continue_after_z():
    assert study_letter(0) == "A"
    assert study_letter(25) == "Z"
    assert study_letter(26) == "AA"
    assert study_letter(51) == "AZ"


def test_claim_ids_follow_table_coordinates_not_input_order():
    tables = CapturedTableSet(tables=[
        CapturedTable(table_id="table_1"),
        CapturedTable(table_id="table_2"),
    ])
    items = [
        _item("second_2020", "table_1", 1, 2),
        _item("first_2020", "table_2", 0, 1),
        _item("first_2020", "table_1", 0, 2),
        _item("first_2020", "table_1", 0, 1),
    ]

    assigned = assign_claim_ids(items, tables)

    assert [(item.study_id, item.review_data_id) for item in assigned] == [
        ("first_2020", "A_01"),
        ("first_2020", "A_02"),
        ("second_2020", "B_01"),
        ("first_2020", "A_03"),
    ]
    assert claim_index(assigned)["A_01"]["cell_ref"] == [0, 1]


def test_twenty_seventh_study_uses_aa():
    tables = CapturedTableSet(tables=[CapturedTable(table_id="table_1")])
    items = [_item(f"study_{row}", "table_1", row, 1) for row in range(27)]
    assigned = assign_claim_ids(items, tables)
    assert assigned[-1].review_data_id == "AA_01"


def test_existing_identity_is_preserved_and_duplicate_is_rejected():
    tables = CapturedTableSet(tables=[CapturedTable(table_id="table_1")])
    kept = assign_claim_ids(
        [_item("s1", "table_1", 0, 1, existing="CUSTOM")], tables)
    assert kept[0].review_data_id == "CUSTOM"

    with pytest.raises(ValueError, match="duplicate claim id"):
        assign_claim_ids([
            _item("s1", "table_1", 0, 1, existing="SAME"),
            _item("s1", "table_1", 0, 2, existing="SAME"),
        ], tables)


def test_explicit_ids_are_globally_unique_even_across_different_keys():
    items = [
        _item("study_a", "table_1", 0, 1, existing="A_01"),
        _item("study_b", "table_2", 8, 4, existing="A_01"),
    ]

    with pytest.raises(ValueError, match="duplicate claim id 'A_01'"):
        validate_claim_ids(items, allow_legacy=True)
    with pytest.raises(ValueError, match="duplicate claim id 'A_01'"):
        global_claim_index(items)
    with pytest.raises(ValueError, match="duplicate claim id 'A_01'"):
        claim_index(items)


def test_empty_identity_is_allowed_only_when_legacy_is_explicitly_enabled():
    legacy = [_item("study_a", "table_1", 0, 1)]
    with pytest.raises(ValueError, match="missing an explicit claim id"):
        validate_claim_ids(legacy)
    validate_claim_ids(legacy, allow_legacy=True)
