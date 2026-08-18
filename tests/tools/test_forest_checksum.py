"""Forest count-column checksum: set compare, not column position.

Fixtures are copies of saved vision-probe JSON (glm-4.6v-flashx / glm-4v-flash).
These tests never call a model.
"""
from __future__ import annotations

import json
from pathlib import Path

from react_review.schemas.table import CapturedTable, CapturedTableSet
from react_review.tools.forest_ocr import (
    _apply_forest_integrity,
    _forest_checksum,
    _is_summary_label,
    _missing_count_columns,
    _parse_int_cell,
    _split_forest_rows,
    _table_from_named_values,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "forest_checksum"
OE_TOTAL = "Open Esophagectomy (OE) Total"
OE_EVENTS = "Open Esophagectomy (OE) Events"


def _load(name: str) -> CapturedTable:
    body = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return _table_from_named_values(body["parsed"], table_id=body.get("id") or "")


def _study_count_cells(table: CapturedTable) -> int:
    study_rows, _summary = _split_forest_rows(table)
    n = 0
    for row in study_rows:
        for cell in row[1:]:
            if _parse_int_cell(cell) is not None:
                n += 1
    return n


def test_summary_label_is_structural_not_domain():
    assert _is_summary_label("Total (Wald)a")
    assert _is_summary_label("Total events:")
    assert _is_summary_label("Subtotal")
    assert _is_summary_label("Heterogeneity: Tau² = 0.00")
    assert _is_summary_label("Test for overall effect")
    assert not _is_summary_label("Li J 2015")
    assert not _is_summary_label("Capovilla 2023")


def test_paid_forest_1_detects_oe_total_and_keeps_nine_cells():
    table = _load("vision_probe_v2_flashx__forest_1.json")
    failed = _forest_checksum(table)
    assert failed == [OE_TOTAL]
    assert _missing_count_columns(table) == []
    _apply_forest_integrity(table)
    assert table.checksum_failures == [OE_TOTAL]
    assert _study_count_cells(table) == 9
    assert any("not released" in note for note in table.difficulties)
    assert 130 in table.checksum_column_sums.values()
    assert 208 in table.checksum_printed_values


def test_paid_forests_2_3_4_have_zero_false_positives():
    for name in (
        "vision_probe_v2_flashx__forest_2.json",
        "vision_probe_v2_flashx__forest_3.json",
        "vision_probe_v2_flashx__forest_4.json",
    ):
        table = _load(name)
        assert _forest_checksum(table) == [], name
        assert _missing_count_columns(table) == [], name


def test_free_forest_1_missing_column_hits_oe_events():
    """Checksum alone would pass: OE Total sums to the printed OE Events total."""
    table = _load("vision_probe_v2_flash__forest_1.json")
    assert OE_EVENTS in _missing_count_columns(table)
    assert OE_TOTAL not in _forest_checksum(table)


def test_no_summary_row_cannot_fail():
    table = CapturedTable(
        table_id="hand",
        header_rows=[["Study or Subgroup", "Events", "Total"]],
        rows=[["Li J 2015", "23", "58"], ["Capovilla 2023", "12", "40"]],
        row_axis_columns=["Study or Subgroup"],
    )
    assert _forest_checksum(table) == []
    _apply_forest_integrity(table)
    assert table.checksum_failures == []


def test_misplaced_summary_value_still_passes_because_compare_is_a_set():
    """forest_2 wrote 208 under OE Events on the total row. Position compare
    would fail MIE Total (study sum 208, that column's summary cells empty).
    """
    table = _load("vision_probe_v2_flashx__forest_2.json")
    assert _forest_checksum(table) == []
    paths = table.column_paths()
    oe_events = paths.index(OE_EVENTS)
    mie_total = paths.index("Minimally Invasive Esophagectomy (MIE) Total")
    summary = [row for row in table.rows if _is_summary_label(row[0])]
    assert any(
        oe_events < len(row) and _parse_int_cell(row[oe_events]) == 208
        for row in summary
    )
    assert all(
        mie_total >= len(row) or _parse_int_cell(row[mie_total]) is None
        for row in summary
    )


def test_legacy_captured_table_json_loads_without_checksum_fields():
    body = {
        "table_id": "table_1",
        "header_rows": [["Study", "N"]],
        "rows": [["Li J 2015", "58"]],
    }
    table = CapturedTable.model_validate(body)
    assert table.checksum_failures == []
    dumped = table.model_dump(mode="json")
    assert "checksum_failures" not in dumped
    assert "row_kinds" not in dumped
    nested = CapturedTableSet.model_validate({"tables": [body]})
    assert nested.tables[0].checksum_failures == []
    assert "checksum_failures" not in nested.model_dump(mode="json")["tables"][0]
