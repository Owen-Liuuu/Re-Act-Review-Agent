"""Rendering a captured table for a human to check against the PDF.

Pure string functions — no printing, no I/O — so the output is testable and the
checkpoint layer stays responsible for how it reaches a terminal or a file.

The job is comparison, not beauty: a reviewer should be able to put this next to
the paper and see whether the extraction is faithful. So cells keep their
verbatim text (elided only when very long), and the shape problems the capture
found are shown alongside rather than silently smoothed over.
"""
from __future__ import annotations

import csv
import io
import re

from react_review.schemas.table import CapturedTable

_MAX_CELL = 22


def _clip(text: str, limit: int = _MAX_CELL) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_captured_table(
    table: CapturedTable, *, max_rows: int = 12, max_cell: int = _MAX_CELL,
) -> str:
    """A monospace preview of one captured table."""
    paths = [_clip(p, max_cell) for p in table.column_paths()]
    body = [[_clip(c, max_cell) for c in row] + [""] * (table.width - len(row))
            for row in table.rows[:max_rows]]

    widths = [len(p) for p in paths]
    for row in body:
        for j, cell in enumerate(row):
            if j < len(widths):
                widths[j] = max(widths[j], len(cell))

    def line(cells: list[str]) -> str:
        return "  " + " | ".join(c.ljust(widths[j]) for j, c in enumerate(cells[:len(widths)]))

    head = f"  [{table.table_id}] {table.caption or '(no caption)'}"
    meta = f"  {len(table.rows)} row(s) x {table.width} column(s)"
    if table.role:
        meta += f" · role: {table.role}"
    out = [head, meta, "", line(paths), "  " + "-+-".join("-" * w for w in widths)]
    out += [line(row) for row in body]
    if len(table.rows) > max_rows:
        out.append(f"  … {len(table.rows) - max_rows} more row(s)")
    if table.footnotes:
        out.append("  footnotes: " + " | ".join(_clip(f, 60) for f in table.footnotes))
    return "\n".join(out)


def render_table_set(tables: list[CapturedTable], **kw) -> str:
    """Every captured table, numbered so a human can drop one by number."""
    if not tables:
        return "  (no tables captured)"
    blocks = []
    for i, t in enumerate(tables, start=1):
        body = render_captured_table(t, **kw)
        first, sep, rest = body.partition("\n")
        numbered = f"  ({i}) {first.strip()}"
        blocks.append(numbered if not sep else numbered + "\n" + rest)
    return "\n\n".join(blocks)


def render_shape_report(table: CapturedTable) -> list[str]:
    """Shape complaints plus whatever the model said it could not read."""
    out = [f"{table.table_id}: {p}" for p in table.validate_shape()]
    out += [f"{table.table_id}: model could not read — {d}" for d in table.difficulties]
    if table.extraction_confidence and table.extraction_confidence < 0.6:
        out.append(f"{table.table_id}: low extraction confidence "
                   f"({table.extraction_confidence:.2f})")
    return out


def to_csv(table: CapturedTable) -> str:
    """The captured table as CSV, so it can be opened next to the PDF."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(table.column_paths())
    for row in table.rows:
        writer.writerow(list(row) + [""] * (table.width - len(row)))
    return buf.getvalue()


_FOREST_CAPTION = re.compile(
    r"^(?:figure\s+[\d.]+\s+)?forest plot of\s+",
    re.IGNORECASE,
)


def display_caption(caption: str) -> str:
    """Human-facing caption: strip the boilerplate 'Forest plot of' prefix."""
    text = " ".join((caption or "").split())
    text = _FOREST_CAPTION.sub("", text)
    return text.rstrip(".")


def forest_summary_line(table: CapturedTable) -> str:
    """One-line forest status: study-row count and checksum."""
    kinds = table.row_kinds or []
    n_study = sum(1 for kind in kinds if kind == "study") or len(table.rows)
    if table.checksum_failures:
        return f"{n_study} study row(s) · checksum failed"
    return f"{n_study} study row(s) · checksum ok"


def render_display_summary(
    tables: list[CapturedTable], figures: list[CapturedTable],
) -> str:
    """At-a-glance tables/figures block for the combined Displays captured pause."""
    lines = ["  tables:"]
    if tables:
        for table in tables:
            cap = table.caption or "(no caption)"
            lines.append(f"    [{table.table_id}] {cap}")
            lines.append(
                f"              {len(table.rows)} row(s) x {table.width} column(s)")
    else:
        lines.append("    (none)")
    lines.append("  figures:")
    if figures:
        for fig in figures:
            lines.append(
                f"    [{fig.table_id}] {display_caption(fig.caption or fig.outcome)}")
            lines.append(f"               {forest_summary_line(fig)}")
    else:
        lines.append("    (none)")
    return "\n".join(lines)
