"""A table captured from a review VERBATIM, in whatever shape it actually has.

This is deliberately schema-free. The review's data-extraction table is the entry
gate of the whole audit — "only if this table is correctly extracted does
anything downstream have meaning" — and imposing our own column names on it at
capture time is exactly how a reader stops being able to check the extraction
against the paper. So: cells are strings, headers keep their levels, placeholder
text like ``NR`` or ``—`` survives, and columns nobody understands are still kept.

Interpretation happens later, against a table a human has already approved.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


def _fill_spans(header_rows: list[list[str]], width: int) -> list[list[str]]:
    """Forward-fill spanned header cells, but only for multi-level headers.

    A header like ``["Study", "", "EAT (mm)", ""]`` over a second row
    ``["", "", "T1DM", "Control"]`` means "EAT (mm)" spans two columns. With a
    single header row there is nothing to span over, so blanks are left alone
    rather than inventing a name for an unnamed column.
    """
    if len(header_rows) < 2:
        return [list(r) + [""] * (width - len(r)) for r in header_rows]
    filled = []
    for row in header_rows:
        out, last = [], ""
        for j in range(width):
            cell = row[j].strip() if j < len(row) else ""
            if cell:
                last = cell
            out.append(cell or last)
        filled.append(out)
    return filled


class CapturedTable(BaseModel):
    """One table, exactly as it appears in the review."""

    table_id: str
    caption: str = ""
    page_hint: str = ""
    # What the model thinks this table is for — advisory only, never a filter.
    role: str = ""                      # characteristics | outcomes | quality | other
    header_rows: list[list[str]] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    row_axis_columns: list[str] = Field(default_factory=list)
    shape_notes: str = ""
    # The cohort names this table actually uses — the review's own words.
    cohort_labels_seen: list[str] = Field(default_factory=list)
    extraction_confidence: float = 0.0
    # What the model could not read. Asking for this, and showing it, is how a
    # failure becomes visible instead of becoming a confidently wrong cell.
    difficulties: list[str] = Field(default_factory=list)

    @property
    def width(self) -> int:
        return max([len(r) for r in self.header_rows] + [len(r) for r in self.rows] + [0])

    def column_paths(self) -> list[str]:
        """One label per column, joining header levels: ``EAT (mm) / T1DM``."""
        filled = _fill_spans(self.header_rows, self.width)
        paths = []
        for j in range(self.width):
            parts: list[str] = []
            for row in filled:
                cell = row[j].strip() if j < len(row) else ""
                if cell and (not parts or parts[-1] != cell):
                    parts.append(cell)
            paths.append(" / ".join(parts))
        return paths

    def row_labels(self) -> list[str]:
        """The study each data row belongs to, one per row.

        Review tables routinely merge the study cell across a study's cohort
        rows, leaving the continuation rows blank. A blank identifier therefore
        means "same study as above", not "unknown study", so it is filled down.
        """
        paths = self.column_paths()
        axis = [j for j, p in enumerate(paths)
                if p in self.row_axis_columns or p.split(" / ")[0] in self.row_axis_columns]
        if not axis:
            axis = [0]
        labels, last = [], ""
        for row in self.rows:
            parts = [row[j].strip() for j in axis if j < len(row) and row[j].strip()]
            label = " ".join(parts)
            if label:
                last = label
            labels.append(label or last)
        return labels

    def validate_shape(self) -> list[str]:
        """Deterministic complaints about the captured shape (never fatal)."""
        problems: list[str] = []
        if not self.rows:
            problems.append("the table has no data rows")
        if not self.header_rows:
            problems.append("the table has no header row")
        width = self.width
        for i, row in enumerate(self.rows):
            if len(row) != width:
                problems.append(f"row {i + 1} has {len(row)} cells, expected {width}")
        paths = self.column_paths()
        seen: set[str] = set()
        for p in paths:
            if p and p in seen:
                problems.append(f"duplicate column label {p!r}")
            seen.add(p)
        for j, p in enumerate(paths):
            if not p:
                problems.append(f"column {j + 1} has no header")
        return problems


class CapturedTableSet(BaseModel):
    """Every table captured from one review, plus what was removed and why."""

    tables: list[CapturedTable] = Field(default_factory=list)
    source_pdf: str = ""
    # Human intervention is itself an auditable step: a table missing from the
    # report must be distinguishable from a table the model never found.
    dropped: list[str] = Field(default_factory=list)
    dropped_reason: str = ""

    def by_id(self, table_id: str) -> CapturedTable | None:
        return next((t for t in self.tables if t.table_id == table_id), None)

    def keep_only(self, table_ids: set[str], *, reason: str) -> "CapturedTableSet":
        """Return a copy holding only ``table_ids``, recording what was removed."""
        removed = [t.table_id for t in self.tables if t.table_id not in table_ids]
        return CapturedTableSet(
            tables=[t for t in self.tables if t.table_id in table_ids],
            source_pdf=self.source_pdf,
            dropped=[*self.dropped, *removed],
            dropped_reason=reason if removed else self.dropped_reason,
        )
