"""Structured tables from an uploaded source PDF. No model. No review text.

A1 kept the file on disk. This reopens it and reads the text-layer grid the way
PMC XML already produces ``CapturedTable`` objects. Failures return an empty
list: the caller keeps the flattened ``full_text`` path rather than guessing.
"""
from __future__ import annotations

from pathlib import Path

import structlog

from react_review.schemas.table import CapturedTable

logger = structlog.get_logger(__name__)


def captured_table_from_grid(
    grid: list[list], *, table_id: str, page_hint: str = "",
) -> CapturedTable | None:
    """First non-empty row is the header; the rest are data. Skip empty grids."""
    rows = [
        ["" if cell is None else str(cell).replace("\n", " ").strip() for cell in row]
        for row in (grid or [])
    ]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return None
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    return CapturedTable(
        table_id=table_id,
        header_rows=[padded[0]],
        rows=padded[1:],
        page_hint=page_hint,
        capture_method="pdf_tables",
        display_kind="pdf_table",
        capture_path="text",
    )


def tables_from_pdf(path: Path | str) -> list[CapturedTable]:
    """PyMuPDF ``find_tables`` over every page. Never raises into retrieve."""
    try:
        import fitz
    except ImportError:
        return []
    target = Path(path)
    if not target.is_file():
        return []
    try:
        document = fitz.open(str(target))
    except Exception as exc:  # noqa: BLE001
        logger.debug("pdf_tables_open_failed", path=str(target), error=str(exc)[:120])
        return []
    try:
        found: list[CapturedTable] = []
        serial = 0
        for page_index, page in enumerate(document, start=1):
            try:
                finder = page.find_tables()
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "pdf_tables_page_failed", page=page_index, error=str(exc)[:120],
                )
                continue
            for table in getattr(finder, "tables", None) or []:
                serial += 1
                try:
                    grid = table.extract()
                except Exception:  # noqa: BLE001
                    continue
                captured = captured_table_from_grid(
                    grid,
                    table_id=f"pdf_p{page_index}_t{serial}",
                    page_hint=str(page_index),
                )
                if captured is not None:
                    found.append(captured)
        return found
    finally:
        document.close()
