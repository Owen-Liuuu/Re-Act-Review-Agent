"""Long rows from a forest plot — no model call, no registry.

A forest plot's shape is known: one label column plus count columns whose
headers carry the arm name. Asking an LLM which column is the label is
inventing a question the layout already answers, and it has answered it
wrong (duplicate header, group='-', dropped first column).
"""
from __future__ import annotations

import re
from typing import Any

from react_review.schemas.table import CapturedTable
from react_review.tools.forest_ocr import _count_columns, _is_count_header, _row_is_summary


def forest_count_field(header: str) -> str | None:
    """Deterministic field_type for a forest count column, or None if it is not one.

    Arm identity stays in the header for cohort resolution. This mapping must
    not invent ``mie_events`` / ``oe_total`` concepts.
    """
    if not _is_count_header(header):
        return None
    text = re.sub(r"\s+", " ", (header or "").strip().lower())
    if text == "events" or text.endswith(" events"):
        return "events"
    return "subgroup_n"


def unpivot_forest(table: CapturedTable) -> list[dict[str, Any]]:
    """Emit one long row per study-row count cell. Skip checksum-failed columns."""
    failed = set(table.checksum_failures or [])
    columns = [(j, path) for j, path in _count_columns(table) if path not in failed]
    labels = table.row_labels()
    out: list[dict[str, Any]] = []
    for index, row in enumerate(table.rows):
        if _row_is_summary(table, index, row):
            continue
        study = (labels[index] if index < len(labels) else "") or (
            row[0].strip() if row else "")
        if not study:
            continue
        for col, header in columns:
            field_type = forest_count_field(header)
            if field_type is None:
                continue
            value = row[col] if col < len(row) else ""
            out.append({
                "row": index,
                "row_key": {"study": study},
                "table_id": table.table_id,
                "column_header": header,
                "value": value,
                "cohort_label": "",
                "unit": "",
                "timepoint_label": "",
                "field_type": field_type,
                "scope": "cohort",
            })
    return out
