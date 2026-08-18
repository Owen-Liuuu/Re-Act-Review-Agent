"""Steps 3–4 — origin labels, then keep only source-paper cells."""
from __future__ import annotations

import json

import structlog

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.parser.review_extraction.prompts import render_extraction_prompt
from react_review.parser.review_extraction.schemas import OriginLabel, ReviewLens
from react_review.schemas.table import CapturedTable

logger = structlog.get_logger(__name__)

_SOURCES: set[str] = {"source_paper", "review_computed", "bibliographic"}
NON_SOURCE: frozenset[str] = frozenset({"review_computed", "bibliographic"})


def drop_non_source(value_source: str) -> bool:
    """True when this cell must not become an audit claim."""
    return (value_source or "").strip() in NON_SOURCE


def _norm(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _column_match(label_path: str, actual: str) -> bool:
    want, have = _norm(label_path), _norm(actual)
    if not want:
        return False
    if want == have:
        return True
    last = have.split(" / ")[-1].strip()
    return want == last or want in have or have in want


def match_origin(
    labels: list[OriginLabel],
    table_id: str,
    column_path: str,
    row: int | None = None,
) -> OriginLabel | None:
    """Prefer a row-specific label, then a whole-column label for this table."""
    column_hits = [
        label for label in labels
        if label.table_id == table_id and _column_match(label.column_path, column_path)
    ]
    if row is not None:
        for label in column_hits:
            if label.row == row:
                return label
    sources = {item.value_source for item in column_hits if item.row is None}
    if len(sources) > 1:
        logger.warning(
            "origin_column_label_conflict",
            table=table_id, column=column_path[:60],
            labels=sorted(sources),
        )
    for label in column_hits:
        if label.row is None:
            return label
    table_wide = [
        label for label in labels
        if label.table_id == table_id and not label.column_path
    ]
    return table_wide[0] if table_wide else None


def fields_for_cell(
    labels: list[OriginLabel] | list,
    table: CapturedTable,
    row: dict,
) -> dict[str, str]:
    """Attach origin / outcome / display_kind onto an unpivoted row."""
    parsed: list[OriginLabel] = []
    for item in labels or []:
        if isinstance(item, OriginLabel):
            parsed.append(item)
        elif isinstance(item, dict):
            try:
                parsed.append(OriginLabel.model_validate(item))
            except Exception:  # noqa: BLE001
                continue
    header = str(row.get("column_header") or "")
    try:
        row_index = int(row["row"])
    except (KeyError, TypeError, ValueError):
        row_index = None
    label = match_origin(parsed, table.table_id, header, row_index)
    outcome = ""
    source = ""
    if label is not None:
        source = label.value_source
        outcome = label.outcome
    return {
        "value_source": source,
        "outcome": outcome or table.outcome or table.caption,
        "display_kind": table.display_kind,
    }


def _parse_row(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _label(raw: object, table_id: str) -> OriginLabel | None:
    if not isinstance(raw, dict):
        return None
    source = str(raw.get("value_source") or "").strip()
    if source not in _SOURCES:
        return None
    path = str(raw.get("column_path") or raw.get("field_name") or "").strip()
    return OriginLabel(
        table_id=str(raw.get("table_id") or table_id),
        column_path=path,
        row=_parse_row(raw.get("row")),
        value_source=source,  # type: ignore[arg-type]
        outcome=str(raw.get("outcome") or "").strip(),
        reason=str(raw.get("reason") or "").strip(),
    )


def _sample_rows(table: CapturedTable, *, limit: int = 2) -> str:
    rows = table.rows[:limit]
    return json.dumps(rows, ensure_ascii=False) if rows else "[]"


def _pooled_rows(table: CapturedTable) -> str:
    """Rows whose study label looks like a forest footer, for the origin prompt."""
    labels = table.row_labels()
    found: list[list[str]] = []
    for i, (label, row) in enumerate(zip(labels, table.rows)):
        key = _norm(label)
        if key.startswith("total") or "wald" in key or key in {"pooled", "overall", "subtotal"}:
            found.append([str(i), label, *row[:6]])
    return json.dumps(found, ensure_ascii=False) if found else "[]"


async def label_table(
    backend: LLMBackend, lens: ReviewLens, table: CapturedTable,
) -> list[OriginLabel]:
    prompt = render_extraction_prompt(
        "claim_origin_v1",
        lens=lens.as_ruler() or "(empty lens)",
        table_id=table.table_id,
        caption=table.caption or table.outcome or "(no caption)",
        column_paths=json.dumps(table.column_paths(), ensure_ascii=False),
        sample_rows=_sample_rows(table),
        pooled_rows=_pooled_rows(table),
        footnotes=json.dumps(table.footnotes, ensure_ascii=False),
    )
    try:
        raw = parse_llm_response(await backend.complete(prompt), backend.model_id)
    except Exception:  # noqa: BLE001
        return []
    bodies = raw.get("labels") if isinstance(raw, dict) else None
    if not isinstance(bodies, list):
        return []
    out: list[OriginLabel] = []
    for body in bodies:
        label = _label(body, table.table_id)
        if label is not None:
            out.append(label)
    return out


async def label_origins(
    backend: LLMBackend, lens: ReviewLens, tables: list[CapturedTable],
) -> list[OriginLabel]:
    labels: list[OriginLabel] = []
    for table in tables:
        labels.extend(await label_table(backend, lens, table))
    return labels


def dropped_notes(labels: list[OriginLabel]) -> list[str]:
    notes: list[str] = []
    for label in labels:
        if label.value_source not in NON_SOURCE:
            continue
        where = label.column_path or "(all columns)"
        if label.row is not None:
            where = f"{where} row {label.row}"
        notes.append(
            f"{label.table_id} / {where}: {label.value_source}"
            + (f" ({label.reason})" if label.reason else ""))
    return notes
