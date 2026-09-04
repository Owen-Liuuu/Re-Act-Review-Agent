"""Deterministic source-table locate for PMC ``CapturedTable`` grids.

The model must not pick the table. A near-match that we "almost" take is the
Treatment/Placebo accident: a self-consistent wrong cell that no later check
can see. This module either uniquely hits (cohort in headers AND field in row
labels) or it refuses. It never ranks closeness.

Classification of a miss is independent of whether the table was actually
sent to a prompt. That lets P4 diagnose last-run failures before P2 changes
what the model sees.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from react_review.normalize.anchors import normalise, normalised_contains
from react_review.normalize.cohorts import distinguishing_tokens, label_affinity, slug
from react_review.schemas.table import CapturedTable

# Same membership as ``normalize.cohorts._COMBINED``. Duplicated so this
# module does not import a private name, and so that file (inside the
# evidence-adequacy hash) does not have to change for a lookup helper.
_COMBINED_GROUPS = {
    "all", "-", "total", "overall", "pooled", "combined", "whole cohort",
    "entire cohort", "both groups", "all participants",
}


NO_TABLES = "no_tables"
NOT_IN_TABLE = "not_in_table"
LOCATED_NOT_READ = "located_not_read"
FALLBACK_TEXT = "fallback_text"

TABLE_LOOKUP_CLASSES = frozenset({
    NO_TABLES, NOT_IN_TABLE, LOCATED_NOT_READ, FALLBACK_TEXT,
})

UNIQUE = "unique"
NONE = "none"
AMBIGUOUS = "ambiguous"
NO_TABLES_STATUS = "no_tables"

_LOCATION_ONLY = {
    "table", "table 1", "table 2", "table 3", "table 4", "table i",
    "table ii", "table iii", "fig", "figure", "forest",
}


@dataclass(frozen=True)
class TableLocateResult:
    """One deterministic locate attempt. ``table`` is set only on a unique hit."""

    status: str
    reason: str
    table: CapturedTable | None = None
    matched_table_ids: tuple[str, ...] = ()
    field_needles: tuple[str, ...] = ()
    cohort_needles: tuple[str, ...] = ()

    @property
    def unique(self) -> bool:
        return self.status == UNIQUE and self.table is not None


@dataclass(frozen=True)
class TableLookupClassification:
    klass: str
    reason: str
    locate: TableLocateResult
    extra: dict = field(default_factory=dict)


def classify_missing_source(
    tables: list[CapturedTable] | None,
    *,
    group: str = "",
    cohort_display: str = "",
    cohorts: dict[str, list[str]] | None = None,
    raw_field_name: str = "",
    concept_variants: list[str] | None = None,
    field_type: str = "",
    column_header: str = "",
    outcome: str = "",
    table_sent: bool = False,
) -> TableLookupClassification:
    """Assign exactly one of the four miss classes. Never ``unknown``."""
    locate = locate_source_table(
        tables,
        group=group,
        cohort_display=cohort_display,
        cohorts=cohorts,
        raw_field_name=raw_field_name,
        concept_variants=concept_variants,
        field_type=field_type,
        column_header=column_header,
        outcome=outcome,
    )
    if locate.status == NO_TABLES_STATUS:
        klass = NO_TABLES
    elif locate.status == UNIQUE:
        klass = LOCATED_NOT_READ
    elif locate.status == AMBIGUOUS:
        klass = FALLBACK_TEXT
    else:
        klass = NOT_IN_TABLE
    sent_note = (
        "the uniquely located table was sent to the prompt"
        if table_sent and locate.unique else
        "the uniquely located table was not sent; extraction used the text windows"
        if locate.unique else
        "extraction used the text windows"
    )
    return TableLookupClassification(
        klass=klass,
        reason=f"{locate.reason}; {sent_note}",
        locate=locate,
        extra={
            "table_sent": table_sent,
            "locate_status": locate.status,
            "matched_table_ids": list(locate.matched_table_ids),
        },
    )


def locate_source_table(
    tables: list[CapturedTable] | None,
    *,
    group: str = "",
    cohort_display: str = "",
    cohorts: dict[str, list[str]] | None = None,
    raw_field_name: str = "",
    concept_variants: list[str] | None = None,
    field_type: str = "",
    column_header: str = "",
    outcome: str = "",
) -> TableLocateResult:
    """Return a unique table only when both axes hit; never a nearest miss."""
    captured = list(tables or [])
    field_needles = _needles(
        raw_field_name,
        *(concept_variants or []),
        column_header,
        outcome,
    )
    if not field_needles and field_type:
        field_needles = _needles(field_type.replace("_", " "))
    combined = _combined_group(group)
    cohort_needles: list[str] = []
    if not combined:
        variants = []
        if cohorts:
            variants.extend(cohorts.get(group, []) or [])
            if cohort_display:
                variants.extend(cohorts.get(slug(cohort_display), []) or [])
        cohort_needles = _needles(cohort_display, group, *variants)

    if not captured:
        return TableLocateResult(
            status=NO_TABLES_STATUS,
            reason="this paper has no parsed tables",
            field_needles=tuple(field_needles),
            cohort_needles=tuple(cohort_needles),
        )

    hits: list[CapturedTable] = []
    for table in captured:
        header_text = _header_text(table)
        row_text = _row_label_text(table)
        field_hit = any(_field_needle_in(needle, row_text) for needle in field_needles)
        cohort_hit = (
            True if combined
            else any(_needle_in(needle, header_text) for needle in cohort_needles)
        )
        if field_hit and cohort_hit:
            hits.append(table)

    ids = tuple(t.table_id for t in hits)
    if len(hits) == 1:
        why = (
            f"field {field_needles!r} uniquely matched row labels of "
            f"{hits[0].table_id!r}"
            if combined else
            f"cohort {cohort_needles!r} in headers and field {field_needles!r} "
            f"in row labels uniquely matched {hits[0].table_id!r}"
        )
        return TableLocateResult(
            status=UNIQUE, reason=why, table=hits[0],
            matched_table_ids=ids,
            field_needles=tuple(field_needles),
            cohort_needles=tuple(cohort_needles),
        )
    if len(hits) > 1:
        return TableLocateResult(
            status=AMBIGUOUS,
            reason=(
                "more than one table matched both axes "
                f"({', '.join(ids)}); refusing to guess"
            ),
            matched_table_ids=ids,
            field_needles=tuple(field_needles),
            cohort_needles=tuple(cohort_needles),
        )
    if combined:
        reason = (
            f"no table's row labels contain field {field_needles!r}"
            if field_needles else
            "no field needles were available to match row labels"
        )
    else:
        reason = (
            f"no table has both cohort {cohort_needles!r} in its headers and "
            f"field {field_needles!r} in its row labels"
        )
    return TableLocateResult(
        status=NONE, reason=reason,
        field_needles=tuple(field_needles),
        cohort_needles=tuple(cohort_needles),
    )


def render_table_prompt(table: CapturedTable) -> str:
    """TSV of one table, including caption. Used by P2 when a unique hit exists."""
    lines = []
    caption = (table.caption or table.table_id or "").strip()
    if caption:
        lines.append(f"TABLE: {caption}")
    for row in [*table.header_rows, *table.rows]:
        lines.append("\t".join(cell.strip() for cell in row))
    return "\n".join(lines)


def _combined_group(group: str) -> bool:
    return (group or "").strip().lower() in _COMBINED_GROUPS or not (group or "").strip()


def _needles(*parts: str) -> list[str]:
    seen: list[str] = []
    for part in parts:
        text = (part or "").strip()
        if not text or _location_only(text):
            continue
        key = normalise(text)
        if not key or any(normalise(existing) == key for existing in seen):
            continue
        seen.append(text)
    return seen


def _location_only(text: str) -> bool:
    folded = normalise(text)
    return folded in _LOCATION_ONLY or folded.startswith("table ") or (
        folded.startswith("forest")
    )


def _header_text(table: CapturedTable) -> str:
    parts = [table.caption or "", table.table_id or ""]
    parts.extend(table.column_paths())
    for row in table.header_rows:
        parts.extend(row)
    return " | ".join(p for p in parts if p)


def _row_label_text(table: CapturedTable) -> str:
    labels = table.row_labels()
    first_cells = [
        (row[0] if row else "") for row in table.rows
    ]
    return " | ".join(p for p in [*labels, *first_cells] if p)


def _needle_in(needle: str, text: str) -> bool:
    """True when every distinguishing token of ``needle`` appears in ``text``.

    Subset, not score: a header that shares some but not all tokens is a miss,
    not a close second. The cohort axis uses this helper and must stay exact.
    """
    if not needle or not text:
        return False
    tokens = distinguishing_tokens(needle)
    if not tokens:
        return normalised_contains(text, needle)
    haystack = set(normalise(text).split())
    return tokens <= haystack


# Field-axis prefix: a needle token may be a prefix of a haystack token
# (``leak`` ⊂ ``leakage``). Reverse is not allowed. Tokens shorter than this
# must match exactly, so ``n`` / ``or`` / ``inf`` do not prefix-match everything.
_MIN_FIELD_PREFIX = 4


def _field_needle_in(needle: str, text: str) -> bool:
    """Token subset for the field axis, with one-way prefix matching."""
    if not needle or not text:
        return False
    tokens = distinguishing_tokens(needle)
    if not tokens:
        return normalised_contains(text, needle)
    haystack = set(normalise(text).split())
    if tokens <= haystack:
        return True
    for tok in tokens:
        if tok in haystack:
            continue
        if len(tok) < _MIN_FIELD_PREFIX:
            return False
        if not any(h.startswith(tok) for h in haystack):
            return False
    return True


# Continuation-row labels: statistic descriptors, not a new field name.
_STATISTIC_WORDS = frozenset({
    "median", "mean", "range", "iqr", "sd", "n", "percent", "percentage",
})
_TOTAL_WORDS = frozenset({"total", "overall", "all"})


@dataclass(frozen=True)
class CellLocateResult:
    """One unique table cell, or a refusal. Never a nearest miss."""

    status: str
    reason: str
    table: CapturedTable | None = None
    row_index: int | None = None
    column_index: int | None = None
    value: str = ""
    quote: str = ""
    row_label: str = ""
    column_header: str = ""
    table_caption: str = ""

    @property
    def unique(self) -> bool:
        return self.status == UNIQUE and bool(self.quote)


def locate_source_cell(
    tables: list[CapturedTable] | None,
    *,
    group: str = "",
    cohort_display: str = "",
    cohorts: dict[str, list[str]] | None = None,
    raw_field_name: str = "",
    concept_variants: list[str] | None = None,
    field_type: str = "",
    column_header: str = "",
    outcome: str = "",
    document_text: str = "",
    mapped_row: str = "",
) -> CellLocateResult:
    """Unique (table, column, row) or refuse. Empty tables → caller keeps the text path.

    Continuation rows (``median (range)`` under ``Age, years``) are taken only
    when the field row's target cell is empty, the follower is a statistic
    label, and exactly one such follower has a value. Two followers with
    values is a guess — refuse.

    ``subgroup_n`` may be read from a unique column-header ``n = N`` without a
    field-axis row hit. ``mapped_row`` is a previously validated row label
    (L3); this function still does not pick the table.
    """
    kind = (field_type or "").strip().lower()
    if kind == "subgroup_n" and not _combined_group(group):
        header_hit = _subgroup_n_from_headers(
            tables, group=group, cohort_display=cohort_display,
            cohorts=cohorts, document_text=document_text)
        if header_hit is not None:
            return header_hit

    table_hit = locate_source_table(
        tables,
        group=group, cohort_display=cohort_display, cohorts=cohorts,
        raw_field_name=raw_field_name, concept_variants=concept_variants,
        field_type=field_type, column_header=column_header, outcome=outcome,
    )
    if table_hit.status != UNIQUE or table_hit.table is None:
        return CellLocateResult(
            status=table_hit.status, reason=table_hit.reason,
            table=table_hit.table)

    table = table_hit.table
    paths = table.column_paths()
    if not paths or not table.rows:
        return CellLocateResult(
            status=NONE, reason="the located table has no grid", table=table)

    combined = _combined_group(group)
    after_psm = kind in {"events", "subgroup_n"}
    col = _unique_column(
        paths, combined=combined, group=group,
        cohort_display=cohort_display, cohorts=cohorts,
        after_psm_tiebreak=after_psm)
    if col is None:
        return CellLocateResult(
            status=AMBIGUOUS if not combined else NONE,
            reason="no unique column matches the requested cohort; refusing to guess",
            table=table)

    field_needles = _needles(
        raw_field_name, *(concept_variants or []), column_header, outcome)
    if not field_needles and field_type:
        field_needles = _needles(field_type.replace("_", " "))
    row_hit = _unique_row(table.rows, col, field_needles)
    if row_hit is None:
        labels = [(row[0] or "").strip() for row in table.rows if row]
        mapped = interpret_row_map_response(mapped_row, labels)
        if mapped:
            row_hit = _row_from_mapped_label(table.rows, col, mapped)
    if row_hit is None:
        return CellLocateResult(
            status=NONE,
            reason="no unique field row (with a conservative continuation) "
                   "matches; refusing to guess",
            table=table)

    row_i, value, row_label = row_hit
    quote = value
    if not normalised_contains(document_text, quote):
        return CellLocateResult(
            status=NONE,
            reason="the cell text is not a verbatim substring of the document; "
                   "refusing to invent a quote",
            table=table)
    header = paths[col]
    caption = (table.caption or table.table_id or "").strip()
    return CellLocateResult(
        status=UNIQUE,
        reason=f"unique cell {table.table_id!r} {row_label!r} / {header!r}",
        table=table, row_index=row_i, column_index=col,
        value=value, quote=quote, row_label=row_label,
        column_header=header, table_caption=caption,
    )


_N_EQ = re.compile(r"n\s*=\s*(\d+)", re.IGNORECASE)

_ROW_MAP_NONE = frozenset({"none", "n/a", "na", "unknown", "null"})
_ROW_MAP_NUMERIC = re.compile(r"^[\d.,\s%()+\-]+$")

SOURCE_ROW_MAP_VERSION = "source-row-map-v1"


def render_source_row_map_prompt(*, outcome: str, row_labels: list[str]) -> str:
    """Ask which row label refers to the outcome. Never ask for a value."""
    numbered = "\n".join(
        f"{i}. {label}" for i, label in enumerate(row_labels, start=1)
        if (label or "").strip())
    return (
        "Which row label in this table refers to the outcome below?\n"
        "Copy one row label exactly, or reply NONE.\n"
        "Do not report any number, count, percentage, or cell value.\n"
        "\n"
        f"OUTCOME: {outcome.strip()}\n"
        "\n"
        "ROW LABELS:\n"
        f"{numbered}\n"
    )


def interpret_row_map_response(raw: str, labels: list[str]) -> str | None:
    """A verbatim row label, or None (NONE / unknown / a number / not in the table)."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        for banned in ("value", "n", "count", "events", "total"):
            if data.get(banned) not in (None, ""):
                return None
        text = str(
            data.get("row_label") or data.get("label") or data.get("row") or ""
        ).strip()
        if not text:
            return None
    text = text.strip().strip('"').strip("'")
    if not text:
        return None
    first = text.splitlines()[0].strip().strip('"').strip("'")
    if first.lower() in _ROW_MAP_NONE:
        return None
    if _ROW_MAP_NUMERIC.fullmatch(first):
        return None
    exact = {label: label for label in labels if label}
    if first in exact:
        return exact[first]
    return None


def _unique_n_in_path(path: str) -> str | None:
    matches = list(_N_EQ.finditer(path or ""))
    if not matches:
        return None
    numbers = {m.group(1) for m in matches}
    if len(numbers) != 1:
        return None
    return next(iter(numbers))


def _is_after_psm(path: str) -> bool:
    tokens = set(normalise(path).split())
    return "after" in tokens and "psm" in tokens


def _header_n_quote(path: str, n: str, document_text: str) -> str:
    """The header fragment that is actually in the document, containing n and N."""
    leaf = path.rsplit(" / ", 1)[-1]
    for candidate in (path, leaf):
        if not candidate or n not in candidate:
            continue
        if _N_EQ.search(candidate) is None:
            continue
        if normalised_contains(document_text, candidate):
            return candidate
    return ""


def _subgroup_n_from_headers(
    tables: list[CapturedTable] | None,
    *,
    group: str,
    cohort_display: str,
    cohorts: dict[str, list[str]] | None,
    document_text: str,
) -> CellLocateResult | None:
    """Unique arm-column ``n = N``, or None to keep the existing cell path.

    Every scoring column across every table is in the pool. If any After-PSM
    column exists, the pool shrinks to those — that is the matched sample the
    review reports, not a guess between Before and After. Any other split
    (two MI columns, unmatched vs matched without ``after``+``psm`` in the
    path) that still has more than one N is a refuse.
    """
    pooled: list[tuple[CapturedTable, int, str, str, str]] = []
    for table in list(tables or []):
        paths = table.column_paths()
        if not paths:
            continue
        for col in _scoring_columns(
                paths, group=group, cohort_display=cohort_display,
                cohorts=cohorts):
            path = paths[col]
            n = _unique_n_in_path(path)
            if n is None:
                continue
            quote = _header_n_quote(path, n, document_text)
            if not quote:
                continue
            pooled.append((table, col, path, n, quote))
    if not pooled:
        return None
    after = [item for item in pooled if _is_after_psm(item[2])]
    chosen = after or pooled
    numbers = {item[3] for item in chosen}
    if len(numbers) != 1:
        return None
    table, col, path, n, quote = chosen[0]
    caption = (table.caption or table.table_id or "").strip()
    return CellLocateResult(
        status=UNIQUE,
        reason=f"unique column-header n= of {table.table_id!r} {path!r}",
        table=table, row_index=None, column_index=col,
        value=n, quote=quote, row_label="",
        column_header=path, table_caption=caption,
    )


def _scoring_columns(
    paths: list[str], *, group: str,
    cohort_display: str, cohorts: dict[str, list[str]] | None,
) -> list[int]:
    """Every value column with positive cohort affinity, not only the unique best."""
    needles = _needles(
        cohort_display, group,
        *(((cohorts or {}).get(group, []) or [])),
        *(((cohorts or {}).get(slug(cohort_display), []) or [])
          if cohort_display else []),
    )
    hits: list[int] = []
    for j, path in enumerate(paths):
        if j == 0:
            continue
        score = max((label_affinity(needle, path) for needle in needles),
                    default=0.0)
        if score > 0:
            hits.append(j)
    return hits


def _unique_column(
    paths: list[str], *, combined: bool, group: str,
    cohort_display: str, cohorts: dict[str, list[str]] | None,
    after_psm_tiebreak: bool = False,
) -> int | None:
    """One value column, or None. Column 0 is the row-label axis, never a value."""
    winners = _scored_column_winners(
        paths, combined=combined, group=group,
        cohort_display=cohort_display, cohorts=cohorts)
    if len(winners) == 1:
        return winners[0]
    if after_psm_tiebreak and len(winners) > 1:
        after = [j for j in winners if _is_after_psm(paths[j])]
        if len(after) == 1:
            return after[0]
    return None


def _scored_column_winners(
    paths: list[str], *, combined: bool, group: str,
    cohort_display: str, cohorts: dict[str, list[str]] | None,
) -> list[int]:
    if combined:
        return [j for j, path in enumerate(paths)
                if j > 0 and _is_total_column(path)]
    needles = _needles(
        cohort_display, group,
        *(((cohorts or {}).get(group, []) or [])),
        *(((cohorts or {}).get(slug(cohort_display), []) or [])
          if cohort_display else []),
    )
    scored: list[tuple[float, int]] = []
    for j, path in enumerate(paths):
        if j == 0:
            continue
        score = max((label_affinity(needle, path) for needle in needles),
                    default=0.0)
        if score > 0:
            scored.append((score, j))
    if not scored:
        return []
    scored.sort(reverse=True)
    best = scored[0][0]
    return [j for score, j in scored if score == best]


def _is_total_column(path: str) -> bool:
    tokens = set(normalise(path).split())
    return bool(tokens & _TOTAL_WORDS)


def _is_continuation_label(label: str) -> bool:
    """True only for statistic descriptors such as ``median (range)``."""
    words = set(normalise(label).split())
    return bool(words) and words <= _STATISTIC_WORDS


def _cell(row: list[str], col: int) -> str:
    if col < 0 or col >= len(row):
        return ""
    return (row[col] or "").strip()


def _unique_row(
    rows: list[list[str]], col: int, field_needles: list[str],
) -> tuple[int, str, str] | None:
    """(row_index, value, row_label) or None. Continuation is all-or-nothing."""
    if not field_needles:
        return None
    field_rows = [
        i for i, row in enumerate(rows)
        if row and any(_field_needle_in(needle, row[0]) for needle in field_needles)
        and not _is_continuation_label(row[0])
    ]
    if len(field_rows) != 1:
        return None
    i = field_rows[0]
    field_label = (rows[i][0] or "").strip()
    direct = _cell(rows[i], col)
    if direct:
        return i, direct, field_label
    valued: list[tuple[int, str, str]] = []
    for j in range(i + 1, len(rows)):
        label = (rows[j][0] or "").strip()
        if not _is_continuation_label(label):
            break
        value = _cell(rows[j], col)
        if value:
            valued.append((j, value, f"{field_label} / {label}"))
    if len(valued) != 1:
        return None
    return valued[0]


def _row_from_mapped_label(
    rows: list[list[str]], col: int, label: str,
) -> tuple[int, str, str] | None:
    """Take the unique row whose label equals ``label`` exactly."""
    hits = [
        i for i, row in enumerate(rows)
        if row and (row[0] or "").strip() == label
    ]
    if len(hits) != 1:
        return None
    i = hits[0]
    value = _cell(rows[i], col)
    if not value:
        return None
    return i, value, label
