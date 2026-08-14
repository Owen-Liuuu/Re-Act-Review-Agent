"""Gold scoring makes TableCapture regressions measurable, not anecdotal."""
from __future__ import annotations

import json
from pathlib import Path

from eval.table_capture_score import load_gold, score_table_capture


def _record(row_id: str, col: int, raw: str, blank_kind: str = "value") -> dict:
    return {
        "document_id": "fixture_review",
        "table_id": "table_1",
        "row_id": row_id,
        "column_id": f"c{col:02d}",
        "raw_value": raw,
        "normalized_value": raw.replace("inter-\nvention", "intervention"),
        "blank_kind": blank_kind,
        "source_locator": "page 3, Table 1",
    }


def _gold(tmp_path: Path) -> Path:
    rows = [
        _record("header_01", 1, "Study"),
        _record("header_01", 2, "Outcome"),
        _record("header_01", 3, "Outcome"),  # duplicate header: position matters
        _record("data_01", 1, "Alpha 2020"),
        _record("data_01", 2, "inter-\nvention"),
        _record("data_01", 3, "NR"),          # NR is data, never a blank
        _record("data_02", 1, "", "merged"),
        _record("data_02", 2, "Control"),
        _record("data_02", 3, "", "true_blank"),
    ]
    path = tmp_path / "gold.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _capture() -> dict:
    return {"tables": [{
        "table_id": "table_1",
        "caption": "Fixture table",
        "header_rows": [["Study", "Outcome", "Outcome"]],
        "rows": [
            ["Alpha 2020", "inter-\nvention", "NR"],
            ["", "Control", ""],
        ],
        "footnotes": ["A footnote is not a data cell"],
    }]}


def test_perfect_transcription_scores_every_expected_cell(tmp_path):
    result = score_table_capture(_gold(tmp_path), _capture())

    assert result["json_success_rate"] == 1.0
    assert result["schema_success_rate"] == 1.0
    assert result["table_recall"] == 1.0
    assert result["row_recall"] == 1.0
    assert result["exact_cell_accuracy"] == 1.0
    assert result["normalized_cell_accuracy"] == 1.0
    assert result["cell_precision"] == 1.0
    assert result["cell_recall"] == 1.0
    assert result["unanchored_cells"] == 0
    assert result["hallucinated_cells"] == 0


def test_tables_outside_the_declared_gold_scope_are_not_hallucinations(tmp_path):
    capture = _capture()
    capture["tables"].append({
        "table_id": "table_2", "caption": "Outcome table",
        "header_rows": [["Study", "Effect"]],
        "rows": [["Alpha 2020", "1.25"]],
    })
    result = score_table_capture(_gold(tmp_path), capture)
    assert result["normalized_cell_accuracy"] == 1.0
    assert result["unanchored_cells"] == 0
    assert result["hallucinated_cells"] == 0


def test_pdf_line_end_hyphenation_only_relaxes_normalized_accuracy(tmp_path):
    capture = _capture()
    capture["tables"][0]["rows"][0][1] = "intervention"
    result = score_table_capture(_gold(tmp_path), capture)

    assert result["exact_cell_accuracy"] < 1.0
    assert result["normalized_cell_accuracy"] == 1.0


def test_deleting_a_table_degrades_recall(tmp_path):
    result = score_table_capture(_gold(tmp_path), {"tables": []})
    assert result["table_recall"] == 0.0
    assert result["row_recall"] == 0.0
    assert result["cell_recall"] == 0.0


def test_deleting_a_row_degrades_row_and_cell_recall(tmp_path):
    capture = _capture()
    capture["tables"][0]["rows"].pop()
    result = score_table_capture(_gold(tmp_path), capture)
    assert result["row_recall"] < 1.0
    assert result["cell_recall"] < 1.0


def test_changing_a_value_degrades_both_cell_accuracies(tmp_path):
    capture = _capture()
    capture["tables"][0]["rows"][0][2] = "42"
    result = score_table_capture(_gold(tmp_path), capture)
    assert result["exact_cell_accuracy"] < 1.0
    assert result["normalized_cell_accuracy"] < 1.0
    assert result["cell_precision"] < 1.0
    assert result["cell_recall"] < 1.0


def test_value_in_a_true_or_merged_blank_is_hallucinated(tmp_path):
    capture = _capture()
    capture["tables"][0]["rows"][1][0] = "Alpha 2020"
    capture["tables"][0]["rows"][1][2] = "0"
    result = score_table_capture(_gold(tmp_path), capture)
    assert result["unanchored_cells"] == 2
    assert result["hallucinated_cells"] == 2


def test_ragged_capture_fails_schema_even_when_json_is_valid(tmp_path):
    capture = _capture()
    capture["tables"][0]["rows"][0].pop()
    result = score_table_capture(_gold(tmp_path), capture)
    assert result["json_success_rate"] == 1.0
    assert result["schema_success_rate"] == 0.0


def test_invalid_json_fails_json_and_schema_without_crashing(tmp_path):
    result = score_table_capture(_gold(tmp_path), "not JSON")
    assert result["json_success_rate"] == 0.0
    assert result["schema_success_rate"] == 0.0


def test_raw_json_text_is_scored_without_being_treated_as_a_filename(tmp_path):
    result = score_table_capture(_gold(tmp_path), json.dumps(_capture()))
    assert result["json_success_rate"] == 1.0
    assert result["normalized_cell_accuracy"] == 1.0


def test_gold_loader_rejects_an_unknown_blank_kind(tmp_path):
    path = _gold(tmp_path)
    rows = path.read_text(encoding="utf-8").splitlines()
    body = json.loads(rows[0])
    body["blank_kind"] = "looks_empty"
    rows[0] = json.dumps(body)
    path.write_text("\n".join(rows), encoding="utf-8")

    try:
        load_gold(path)
    except ValueError as exc:
        assert "blank_kind" in str(exc)
    else:
        raise AssertionError("unknown blank_kind was accepted")
