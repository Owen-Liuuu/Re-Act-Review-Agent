"""Rendering a captured table so a human can check it against the PDF."""
from __future__ import annotations

from react_review.parser.table_render import (
    render_captured_table,
    render_shape_report,
    render_table_set,
    to_csv,
)
from react_review.schemas.table import CapturedTable

TABLE = CapturedTable(
    table_id="table_1",
    caption="Table 1. Characteristics of included studies",
    role="characteristics",
    # a spanned header: "EAT (mm)" covers the last two columns
    header_rows=[["Study", "Country", "EAT (mm)", ""],
                 ["", "", "T1DM", "Control"]],
    rows=[["Ahmad 2022", "Egypt", "6.60 ± 0.71", "3.83 ± 0.35"],
          ["Keles 2016", "Turkey", "NR", "—"]],
    footnotes=["values are mean ± SD unless stated"],
    row_axis_columns=["Study"],
)


def test_column_paths_join_header_levels():
    assert TABLE.column_paths() == [
        "Study", "Country", "EAT (mm) / T1DM", "EAT (mm) / Control"]


def test_single_header_row_does_not_invent_names_for_blank_columns():
    t = CapturedTable(table_id="t", header_rows=[["Study", "N", ""]],
                      rows=[["Ahmad 2022", "100", "x"]])
    assert t.column_paths() == ["Study", "N", ""]      # blank stays blank


def test_render_keeps_placeholder_cells_visible():
    out = render_captured_table(TABLE)
    assert "NR" in out and "—" in out                  # never silently smoothed away
    assert "EAT (mm) / T1DM" in out
    assert "2 row(s) x 4 column(s)" in out
    assert "role: characteristics" in out


def test_render_elides_long_cells_but_keeps_the_row():
    t = CapturedTable(table_id="t", header_rows=[["Notes"]],
                      rows=[["x" * 200]])
    out = render_captured_table(t)
    assert "…" in out and len(max(out.splitlines(), key=len)) < 120


def test_render_table_set_numbers_the_tables_for_dropping():
    out = render_table_set([TABLE, CapturedTable(table_id="table_2", caption="Outcomes")])
    assert "(1)" in out and "(2)" in out
    assert "[table_1]" in out and "[table_2]" in out


def test_shape_report_surfaces_ragged_rows_and_model_difficulties():
    t = CapturedTable(
        table_id="t", header_rows=[["A", "B", "C"]],
        rows=[["1", "2", "3"], ["1", "2"]],            # ragged
        difficulties=["the last column was cut off"],
        extraction_confidence=0.4,
    )
    report = render_shape_report(t)
    assert any("row 2 has 2 cells, expected 3" in r for r in report)
    assert any("could not read" in r for r in report)
    assert any("low extraction confidence" in r for r in report)


def test_empty_table_is_reported_not_hidden():
    assert any("no data rows" in p
               for p in CapturedTable(table_id="t", header_rows=[["A"]]).validate_shape())


def test_row_labels_fill_down_merged_study_cells():
    # A review table merges the study cell across its cohort rows; a blank
    # identifier means "same study as above", not "unknown study".
    t = CapturedTable(
        table_id="t", header_rows=[["Author", "Group", "Age"]],
        rows=[["Ahmad 2022", "T1DM", "12.9"],
              ["", "Control", "13.0"],
              ["Keles 2016", "T1DM", "34"]],
        row_axis_columns=["Author"])
    assert t.row_labels() == ["Ahmad 2022", "Ahmad 2022", "Keles 2016"]


def test_row_labels_fall_back_to_the_first_column():
    t = CapturedTable(table_id="t", header_rows=[["Study", "N"]],
                      rows=[["Ahmad 2022", "100"]])       # no row_axis declared
    assert t.row_labels() == ["Ahmad 2022"]


def test_row_years_read_a_year_column_and_fill_down_merged_cells():
    t = CapturedTable(
        table_id="t", header_rows=[["Study", "Year", "N"]],
        rows=[["Li J et al.", "2015", "80"],
              ["", "", "40"],
              ["Li K et al.", "2025", "90"]],
        row_axis_columns=["Study"])
    assert t.row_labels() == ["Li J et al.", "Li J et al.", "Li K et al."]
    assert t.row_years() == ["2015", "2015", "2025"]


def test_row_years_are_empty_when_the_table_has_no_year_column():
    t = CapturedTable(table_id="t", header_rows=[["Study", "N"]],
                      rows=[["Ahmad et al. [2022]", "100"]],
                      row_axis_columns=["Study"])
    assert t.row_years() == [""]


def test_rows_name_studies_uses_axis_before_role():
    studies = CapturedTable(
        table_id="t1", role="outcomes",
        header_rows=[["Study", "EAT"]],
        rows=[["Ahmad 2022", "6.6"]],
        row_axis_columns=["Study"])
    pooled = CapturedTable(
        table_id="t2", role="outcomes",
        header_rows=[["Outcome", "OR (95% CI)"]],
        rows=[["Overall Complications", "0.40 (0.27-0.60)"]],
        row_axis_columns=["Outcome"])
    tagged_only = CapturedTable(
        table_id="t3", role="outcomes",
        header_rows=[["A", "B"]],
        rows=[["x", "1"]])
    assert studies.rows_name_studies() is True
    assert pooled.rows_name_studies() is False
    assert tagged_only.rows_name_studies() is False
    revman = CapturedTable(
        table_id="fig", role="outcomes",
        header_rows=[["Study or Subgroup", "Events", "Total"]],
        rows=[["Li J 2015", "23", "58"]],
        row_axis_columns=["Study or Subgroup"])
    assert revman.rows_name_studies() is True


def test_csv_round_trip_preserves_cells_verbatim():
    csv_text = to_csv(TABLE)
    assert "EAT (mm) / T1DM" in csv_text
    assert "6.60 ± 0.71" in csv_text and "NR" in csv_text
    assert len(csv_text.strip().splitlines()) == 3         # header + 2 rows
