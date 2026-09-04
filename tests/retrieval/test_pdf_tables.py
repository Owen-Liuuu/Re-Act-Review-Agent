"""A2: uploaded PDFs become CapturedTable grids. Empty grid → text path."""
from __future__ import annotations

import pytest

from react_review.retrieval.local_pdf import LocalPdfRetriever, local_pdf_path
from react_review.retrieval.pdf_tables import captured_table_from_grid, tables_from_pdf
from react_review.steps.paper_verification.schemas import ReferenceEntry

_DOI = "10.1000/a2-tables"


def test_first_row_is_the_header_and_the_rest_are_data():
    table = captured_table_from_grid(
        [["Age", "Total", "MIE"], ["years", "73", "70"]],
        table_id="t1", page_hint="1",
    )
    assert table is not None
    assert table.header_rows == [["Age", "Total", "MIE"]]
    assert table.rows == [["years", "73", "70"]]
    assert table.capture_method == "pdf_tables"
    assert table.column_paths()[1] == "Total"


def test_empty_or_single_row_grids_are_refused():
    assert captured_table_from_grid([], table_id="t") is None
    assert captured_table_from_grid([["Age", "Total"]], table_id="t") is None


def test_none_cells_become_empty_strings():
    table = captured_table_from_grid(
        [["H1", None], [None, "1"]], table_id="t",
    )
    assert table is not None
    assert table.header_rows == [["H1", ""]]
    assert table.rows == [["", "1"]]


@pytest.mark.asyncio
async def test_retrieve_attaches_parsed_tables_without_changing_text(
    tmp_path, monkeypatch,
):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fixture")
    monkeypatch.setattr(
        "react_review.retrieval.local_pdf._pdf_text", lambda _path: "Age Total 73",
    )
    from react_review.retrieval.pdf_tables import captured_table_from_grid
    grid = captured_table_from_grid(
        [["Age", "Total"], ["years", "73"]], table_id="pdf_p1_t1", page_hint="1",
    )
    monkeypatch.setattr(
        "react_review.retrieval.local_pdf.tables_from_pdf", lambda _path: [grid],
    )

    document = await LocalPdfRetriever({_DOI: pdf}).retrieve(
        ReferenceEntry(title="t", doi=_DOI),
    )

    assert document is not None
    assert document.full_text == "Age Total 73"
    assert local_pdf_path(document) == str(pdf)
    assert len(document.tables) == 1
    assert document.tables[0].rows[0][1] == "73"
    dumped = document.model_dump()
    assert "source_pdf_path" not in dumped
    assert dumped["tables"]


@pytest.mark.asyncio
async def test_retrieve_survives_an_unreadable_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a pdf")
    monkeypatch.setattr(
        "react_review.retrieval.local_pdf._pdf_text", lambda _path: "text",
    )
    document = await LocalPdfRetriever({_DOI: pdf}).retrieve(
        ReferenceEntry(title="t", doi=_DOI),
    )
    assert document is not None
    assert document.tables == []
    assert tables_from_pdf(pdf) == []


def test_a_lined_pdf_may_yield_a_grid(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "grid.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=200)
    x0, y0, cw, rh = 50, 50, 90, 28
    labels = [["Age", "Total"], ["years", "73"]]
    for row_i, row in enumerate(labels):
        for col_i, label in enumerate(row):
            rect = fitz.Rect(x0 + col_i * cw, y0 + row_i * rh,
                             x0 + (col_i + 1) * cw, y0 + (row_i + 1) * rh)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 8, rect.y0 + 18), label, fontsize=11)
    document.save(str(pdf))
    document.close()
    tables = tables_from_pdf(pdf)
    if not tables:
        pytest.skip("this PyMuPDF build did not detect the synthetic grid")
    blob = " ".join(
        " ".join(cell for cell in row)
        for table in tables
        for row in (table.header_rows + table.rows)
    )
    assert "73" in blob or "Age" in blob
