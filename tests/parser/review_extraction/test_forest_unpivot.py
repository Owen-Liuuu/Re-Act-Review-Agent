"""Deterministic forest unpivot: study count cells only, no model."""
from __future__ import annotations

import json
from pathlib import Path

from react_review.parser.review_extraction.forest_unpivot import (
    forest_count_field,
    unpivot_forest,
)
from react_review.schemas.table import CapturedTable
from react_review.tools.forest_ocr import (
    _apply_forest_integrity,
    _split_forest_rows,
    _table_from_named_values,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "forest_checksum"
OE_TOTAL = "Open Esophagectomy (OE) Total"


def _load_parsed(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_duplicate_study_header_is_not_prepended_again():
    table = _table_from_named_values({
        "column_headers": ["Study or Subgroup", "Events", "Total"],
        "rows": [{
            "label": "Li J 2015", "kind": "study",
            "values": [
                {"column": "Events", "value": "23"},
                {"column": "Total", "value": "58"},
            ],
        }],
    })
    assert table.header_rows == [["Study or Subgroup", "Events", "Total"]]
    assert table.rows == [["Li J 2015", "23", "58"]]
    assert table.row_kinds == ["study"]
    assert table.column_paths() == ["Study or Subgroup", "Events", "Total"]


def test_row_kinds_prefer_model_kind_and_note_mismatch():
    table = _table_from_named_values({
        "column_headers": ["Events", "Total"],
        "rows": [
            {"label": "Li J 2015", "kind": "summary",
             "values": [{"column": "Events", "value": "23"}]},
            {"label": "Total", "kind": "study",
             "values": [{"column": "Events", "value": "1"}]},
        ],
        "difficulties": [],
    })
    assert table.row_kinds == ["summary", "study"]
    study, summary = _split_forest_rows(table)
    assert study == [["Total", "1", ""]]
    assert summary[0][0] == "Li J 2015"
    assert any("kind=summary" in note for note in table.difficulties)
    assert any("kind=study" in note for note in table.difficulties)


def test_unpivot_forest_skips_summary_and_checksum_failed_columns():
    body = _load_parsed("vision_probe_v2_flashx__forest_1.json")
    table = _table_from_named_values(body["parsed"], table_id="forest_1")
    _apply_forest_integrity(table)
    assert OE_TOTAL in table.checksum_failures
    rows = unpivot_forest(table)
    studies = {r["row_key"]["study"] for r in rows}
    assert "Total (Wald)a" not in studies
    assert "Total events:" not in studies
    assert all(r["cohort_label"] == "" for r in rows)
    assert all(OE_TOTAL not in r["column_header"] for r in rows)
    assert len(rows) == 9


def test_unpivot_forest_emits_one_cell_per_count_column():
    table = CapturedTable(
        table_id="figure_3",
        header_rows=[[
            "Study or Subgroup",
            "Minimally Invasive Esophagectomy (MIE) Events",
            "Minimally Invasive Esophagectomy (MIE) Total",
            "Open Esophagectomy (OE) Events",
            "Open Esophagectomy (OE) Total",
        ]],
        rows=[
            ["Capovilla 2023", "19", "58", "58", "102"],
            ["Total (Wald)a", "53", "208", "102", "215"],
        ],
        row_kinds=["study", "summary"],
        row_axis_columns=["Study or Subgroup"],
        display_kind="forest_plot",
    )
    rows = unpivot_forest(table)
    assert len(rows) == 4
    assert {r["column_header"] for r in rows} == {
        "Minimally Invasive Esophagectomy (MIE) Events",
        "Minimally Invasive Esophagectomy (MIE) Total",
        "Open Esophagectomy (OE) Events",
        "Open Esophagectomy (OE) Total",
    }
    assert all(r["row"] == 0 for r in rows)
    events = next(
        r for r in rows
        if r["column_header"].endswith("Events") and "MIE" in r["column_header"])
    assert events["value"] == "19"
    assert events["field_type"] == "events"
    assert events["scope"] == "cohort"
    totals = [r for r in rows if r["column_header"].endswith("Total")]
    assert {r["field_type"] for r in totals} == {"subgroup_n"}
    assert {r["field_type"] for r in rows} <= {"events", "subgroup_n"}


def test_forest_count_field_maps_arm_headers_without_minting_arm_concepts():
    assert forest_count_field(
        "Minimally Invasive Esophagectomy (MIE) Events") == "events"
    assert forest_count_field(
        "Minimally Invasive Esophagectomy (MIE) Total") == "subgroup_n"
    assert forest_count_field("Open Esophagectomy (OE) Events") == "events"
    assert forest_count_field("Open Esophagectomy (OE) Total") == "subgroup_n"
    assert forest_count_field("Events") == "events"
    assert forest_count_field("Total") == "subgroup_n"
    assert forest_count_field("Weight") is None
    assert forest_count_field("Odds ratio") is None


def test_unpivot_forest_skips_weight_and_odds_ratio():
    table = CapturedTable(
        table_id="figure_2",
        header_rows=[[
            "Study or Subgroup", "Events", "Total", "Odds ratio", "Weight",
        ]],
        rows=[["Li J 2015", "23", "58", "0.40", "12%"]],
        row_kinds=["study"],
        row_axis_columns=["Study or Subgroup"],
        display_kind="forest_plot",
    )
    rows = unpivot_forest(table)
    assert {r["column_header"] for r in rows} == {"Events", "Total"}
    by_ft = {r["field_type"]: r["value"] for r in rows}
    assert by_ft == {"events": "23", "subgroup_n": "58"}


def test_unpivot_forest_row_kinds_length_mismatch_falls_back_to_label():
    table = CapturedTable(
        table_id="figure_2",
        header_rows=[["Study or Subgroup", "MIE Events", "MIE Total"]],
        rows=[
            ["Li J 2015", "22", "58"],
            ["Total (Wald)a", "22", "58"],
        ],
        row_kinds=["study"],
        row_axis_columns=["Study or Subgroup"],
        display_kind="forest_plot",
    )
    rows = unpivot_forest(table)
    studies = {r["row_key"]["study"] for r in rows}
    assert studies == {"Li J 2015"}
    assert len(rows) == 2
