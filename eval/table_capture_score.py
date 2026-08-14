"""Score a TableCapture JSON response against private, cell-level gold.

Gold is JSONL so each annotation has one auditable source locator.  The scorer
is deliberately position-aware: duplicate headers are distinct columns, merged
cells stay blank, and footnotes never enter the data-cell denominator.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


BLANK_KINDS = {"value", "true_blank", "merged", "not_applicable"}
GOLD_FIELDS = {
    "document_id", "table_id", "row_id", "column_id", "raw_value",
    "normalized_value", "blank_kind", "source_locator",
}


def normalize_cell(value: object) -> str:
    """Relax PDF layout artifacts only; never reinterpret a value."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_gold(path: Path | str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        body = json.loads(line)
        if not isinstance(body, dict) or set(body) != GOLD_FIELDS:
            raise ValueError(
                f"gold line {line_no} must contain exactly {sorted(GOLD_FIELDS)}")
        kind = str(body["blank_kind"])
        if kind not in BLANK_KINDS:
            raise ValueError(f"gold line {line_no} has unknown blank_kind {kind!r}")
        record = {field: str(body[field]) for field in GOLD_FIELDS}
        if not all(record[field] for field in (
                "document_id", "table_id", "row_id", "column_id", "source_locator")):
            raise ValueError(f"gold line {line_no} has an empty identity/source field")
        if kind != "value" and (record["raw_value"] or record["normalized_value"]):
            raise ValueError(
                f"gold line {line_no} marks {kind} but contains a value")
        if kind == "value" and not record["raw_value"]:
            raise ValueError(f"gold line {line_no} marks value but is empty")
        key = tuple(record[field] for field in (
            "document_id", "table_id", "row_id", "column_id"))
        if key in seen:
            raise ValueError(f"gold line {line_no} duplicates cell {key}")
        seen.add(key)
        records.append(record)
    if not records:
        raise ValueError(f"gold file {path} contains no cells")
    return records


def _capture_object(capture: object) -> tuple[dict[str, Any], bool]:
    if isinstance(capture, Path):
        capture = capture.read_text(encoding="utf-8")
    elif isinstance(capture, str) and Path(capture).is_file():
        capture = Path(capture).read_text(encoding="utf-8")
    if isinstance(capture, str):
        try:
            capture = json.loads(capture)
        except (json.JSONDecodeError, TypeError):
            return {}, False
    return (capture, True) if isinstance(capture, dict) else ({}, False)


def _schema_error_counts(body: dict[str, Any]) -> dict[str, int]:
    errors: defaultdict[str, int] = defaultdict(int)
    tables = body.get("tables")
    if not isinstance(tables, list):
        errors["tables_not_list"] += 1
        return dict(errors)
    ids: set[str] = set()
    for table in tables:
        if not isinstance(table, dict):
            errors["table_not_object"] += 1
            continue
        table_id = table.get("table_id")
        headers = table.get("header_rows")
        rows = table.get("rows")
        if not isinstance(table_id, str) or not table_id or table_id in ids:
            errors["invalid_or_duplicate_table_id"] += 1
        else:
            ids.add(table_id)
        if not isinstance(headers, list) or not isinstance(rows, list):
            errors["rows_not_lists"] += 1
            continue
        if any(not isinstance(row, list) for row in [*headers, *rows]):
            errors["row_not_list"] += 1
            continue
        widths = [len(row) for row in headers]
        if rows and not widths:
            errors["missing_header"] += 1
            continue
        width = max(widths, default=0)
        if any(len(row) != width for row in [*headers, *rows]):
            errors["ragged_table"] += 1
    return dict(errors)


def _valid_schema(body: dict[str, Any]) -> bool:
    return not _schema_error_counts(body)


def _column_index(column_id: str) -> int:
    match = re.search(r"(\d+)$", column_id)
    if not match:
        raise ValueError(f"column_id {column_id!r} has no numeric position")
    return int(match.group(1)) - 1


def _gold_layout(records: list[dict[str, str]]):
    tables: dict[str, dict[str, Any]] = {}
    for record in records:
        table = tables.setdefault(record["table_id"], {"rows": [], "cells": {}})
        if record["row_id"] not in table["rows"]:
            table["rows"].append(record["row_id"])
        table["cells"][(record["row_id"], _column_index(record["column_id"]))] = record
    return tables


def _actual_layout(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for table in body.get("tables") or []:
        if not isinstance(table, dict) or not isinstance(table.get("table_id"), str):
            continue
        rows: dict[str, list[Any]] = {}
        for prefix, values in (("header", table.get("header_rows")),
                               ("data", table.get("rows"))):
            if not isinstance(values, list):
                continue
            for index, row in enumerate(values, 1):
                if isinstance(row, list):
                    rows[f"{prefix}_{index:02d}"] = row
        out[table["table_id"]] = {"rows": rows}
    return out


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def score_table_capture(gold_path: Path | str, capture: object) -> dict[str, Any]:
    gold = load_gold(gold_path)
    body, json_ok = _capture_object(capture)
    errors: defaultdict[str, int] = defaultdict(int)
    if not json_ok:
        errors["invalid_json"] += 1
    else:
        for key, value in _schema_error_counts(body).items():
            errors[key] += value
    schema_ok = json_ok and not errors
    expected = _gold_layout(gold)
    actual = _actual_layout(body) if json_ok else {}

    expected_tables = len(expected)
    found_tables = sum(table_id in actual for table_id in expected)
    expected_rows = sum(len(table["rows"]) for table in expected.values())
    found_rows = 0
    exact = normalized = 0
    true_positive = 0
    expected_values = sum(r["blank_kind"] == "value" for r in gold)
    actual_values = 0
    unanchored = hallucinated = 0

    for table_id, gold_table in expected.items():
        actual_rows = actual.get(table_id, {}).get("rows", {})
        if table_id not in actual:
            errors["missing_table"] += 1
        for row_id in gold_table["rows"]:
            row = actual_rows.get(row_id)
            if row is not None:
                found_rows += 1
            else:
                errors["missing_row"] += 1
            for (cell_row, col), record in gold_table["cells"].items():
                if cell_row != row_id:
                    continue
                observed = str(row[col] if row is not None and col < len(row) else "")
                if observed == record["raw_value"]:
                    exact += 1
                observed_normalized = normalize_cell(observed)
                expected_normalized = record["normalized_value"]
                if observed_normalized == expected_normalized:
                    normalized += 1
                    if observed != record["raw_value"]:
                        errors["layout_normalization_only"] += 1
                if record["blank_kind"] == "value":
                    if not observed:
                        errors["missing_value"] += 1
                    elif observed_normalized != expected_normalized:
                        errors["value_mismatch"] += 1
                if observed:
                    actual_values += 1
                    if record["blank_kind"] == "value" and (
                            observed_normalized == expected_normalized):
                        true_positive += 1
                    else:
                        unanchored += 1
                        if record["blank_kind"] != "value":
                            hallucinated += 1
                            errors[f"{record['blank_kind']}_filled"] += 1

    # Count values outside the annotated geometry as unanchored hallucinations.
    for table_id, table in actual.items():
        gold_table = expected.get(table_id)
        # This gate intentionally annotates only the first characteristics
        # table. Other captured tables are outside scope, not hallucinations.
        if gold_table is None:
            continue
        for row_id, row in table["rows"].items():
            for col, observed in enumerate(row):
                if not str(observed or ""):
                    continue
                if (row_id, col) not in gold_table["cells"]:
                    actual_values += 1
                    unanchored += 1
                    hallucinated += 1
                    errors["extra_geometry"] += 1

    total_cells = len(gold)
    return {
        "json_success_rate": float(json_ok),
        "schema_success_rate": float(schema_ok),
        "table_recall": _ratio(found_tables, expected_tables),
        "row_recall": _ratio(found_rows, expected_rows),
        "exact_cell_accuracy": _ratio(exact, total_cells),
        "normalized_cell_accuracy": _ratio(normalized, total_cells),
        "cell_precision": _ratio(true_positive, actual_values),
        "cell_recall": _ratio(true_positive, expected_values),
        "unanchored_cells": unanchored,
        "hallucinated_cells": hallucinated,
        "error_counts": dict(sorted(errors.items())),
        "counts": {
            "gold_tables": expected_tables,
            "gold_rows": expected_rows,
            "gold_cells": total_cells,
            "gold_value_cells": expected_values,
            "actual_nonempty_cells": actual_values,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(score_table_capture(args.gold, args.capture), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
