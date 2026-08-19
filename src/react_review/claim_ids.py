"""Readable, deterministic identities for review-table claims.

The captured table stays verbatim.  Identities are assigned only after it has
been converted into :class:`ReviewDataItem` objects, so a study-level value that
was repeated by the table layout is de-duplicated before it consumes a number.
"""
from __future__ import annotations

from collections.abc import Iterable

from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.table import CapturedTableSet


def study_letter(index: int) -> str:
    """Return an Excel-style label: 0 -> A, 25 -> Z, 26 -> AA."""

    if index < 0:
        raise ValueError("study index must be non-negative")
    value = index + 1
    out: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        out.append(chr(ord("A") + remainder))
    return "".join(reversed(out))


def declared_claim_id(item: object) -> str:
    """Return an explicitly carried identity, without inventing a fallback."""

    for name in ("review_data_id", "claim_id", "audit_id"):
        value = str(getattr(item, name, "") or "").strip()
        if value:
            return value
    batch = getattr(item, "batch_provenance", None)
    value = str(getattr(batch, "claim_id", "") or "").strip()
    if value:
        return value
    return ""


def claim_id_of(item: object) -> str:
    """Return one claim identity, with a locator fallback for legacy rows."""

    declared = declared_claim_id(item)
    if declared:
        return declared
    return "|".join(str(part) for part in (
        getattr(item, "study_id", ""),
        getattr(item, "group", "-"),
        getattr(item, "timepoint", "single"),
        getattr(item, "field_type", ""),
        getattr(item, "table_id", "") or "",
        getattr(item, "cell_ref", None) or "",
        getattr(item, "checklist_id", "") or "",
    ))


def validate_claim_ids(
    items: Iterable[object],
    *,
    allow_legacy: bool = False,
) -> None:
    """Validate explicit identities across one whole side of a run.

    A review claim and its matching source evidence legitimately share an ID,
    so callers validate each side separately.  Within either side an explicit
    ID is globally unique: grouping by study or match key must not hide a
    collision elsewhere in the run.
    """

    seen: dict[str, int] = {}
    for position, item in enumerate(items):
        claim_id = declared_claim_id(item)
        if not claim_id:
            if allow_legacy:
                continue
            raise ValueError(
                f"claim at position {position} is missing an explicit claim id; "
                "only a declared legacy path may omit it")
        if claim_id in seen:
            raise ValueError(
                f"duplicate claim id {claim_id!r} at positions "
                f"{seen[claim_id]} and {position}")
        seen[claim_id] = position


def global_claim_index(items: Iterable[object]) -> dict[str, object]:
    """Index every explicit identity after enforcing run-wide uniqueness."""

    materialized = list(items)
    validate_claim_ids(materialized, allow_legacy=True)
    return {
        claim_id: item
        for item in materialized
        if (claim_id := declared_claim_id(item))
    }


def _source_order(
    item: ReviewDataItem,
    *,
    position: int,
    table_order: dict[str, int],
) -> tuple[int, int, int, int]:
    """Order a claim by approved table and cell, never by LLM return order."""

    unknown_table = len(table_order)
    table_index = table_order.get(item.table_id, unknown_table)
    if item.cell_ref is None:
        return table_index, 2**31 - 1, 2**31 - 1, position
    row, column = item.cell_ref
    return table_index, int(row), int(column), position


def assign_claim_ids(
    items: Iterable[ReviewDataItem],
    tables: CapturedTableSet,
) -> list[ReviewDataItem]:
    """Assign ``A_01``, ``A_02``, ``B_01`` identities in source-table order.

    Existing non-empty identities are preserved for compatibility with imported
    or hand-authored claims.  They still consume their position in the study's
    sequence.  The returned list is ordered by the approved source coordinates,
    which makes terminal and report output follow the same order as the table.
    """

    materialized = list(items)
    table_order = {table.table_id: i for i, table in enumerate(tables.tables)}
    ordered = sorted(
        enumerate(materialized),
        key=lambda pair: _source_order(
            pair[1], position=pair[0], table_order=table_order),
    )

    study_labels: dict[str, str] = {}
    study_counts: dict[str, int] = {}
    assigned: list[ReviewDataItem] = []
    seen_ids: set[str] = set()

    for _, item in ordered:
        study_id = (item.study_id or "").strip()
        if not study_id:
            raise ValueError("cannot assign a claim id to an item without study_id")
        if study_id not in study_labels:
            study_labels[study_id] = study_letter(len(study_labels))
            study_counts[study_id] = 0

        study_counts[study_id] += 1
        generated = f"{study_labels[study_id]}_{study_counts[study_id]:02d}"
        claim_id = (item.review_data_id or "").strip() or generated
        if claim_id in seen_ids:
            raise ValueError(f"duplicate claim id {claim_id!r}")
        seen_ids.add(claim_id)
        assigned.append(item.model_copy(update={"review_data_id": claim_id}))

    return assigned


def claim_index(items: Iterable[ReviewDataItem]) -> dict[str, dict[str, object]]:
    """Return the compact ID-to-source mapping stored in parser checkpoints."""

    materialized = list(items)
    validate_claim_ids(materialized, allow_legacy=True)
    return {
        item.review_data_id: {
            "study_id": item.study_id,
            "study_label_raw": str(getattr(item, "study_label_raw", "") or ""),
            "table_id": item.table_id,
            "cell_ref": list(item.cell_ref) if item.cell_ref is not None else None,
            "column_header": item.column_header or item.raw_field_name,
        }
        for item in materialized
        if item.review_data_id
    }
