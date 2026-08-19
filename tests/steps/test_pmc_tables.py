"""PMC XML tables become CapturedTable without changing the TSV text path."""
from __future__ import annotations

from pathlib import Path

from react_review.schemas.table import CapturedTable
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.fulltext_retriever import (
    FullTextRetriever,
    _table_to_text,
    pmc_xml_to_tables,
)
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract import FetchFullTextTool

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pmc_tables"

SYNTHETIC_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <article-title>Fixture paper</article-title>
  <abstract><p>An abstract sentence.</p></abstract>
  <body>
    <sec>
      <title>Methods</title>
      <p>We enrolled participants.</p>
    </sec>
    <sec>
      <title>Results</title>
      <p>See Table 1.</p>
    </sec>
    <table-wrap id="Tab1">
      <label>Table 1</label>
      <caption><p>Patient characteristics</p></caption>
      <table>
        <thead>
          <tr><th></th><th>T1DM-patients (n=88)</th><th>Controls (n=60)</th></tr>
        </thead>
        <tbody>
          <tr><td>Age (years)</td><td>61.3 ± 7.1</td><td>62.3 ± 6.8</td></tr>
          <tr><td>Body mass index</td><td>25.8 ± 3.9</td><td>25.5 ± 4.2</td></tr>
        </tbody>
      </table>
    </table-wrap>
  </body>
</article>
"""

# Frozen bytes of `_pmc_xml_to_text` on SYNTHETIC_XML. A change here is a
# change to the existing text path, which S1 is forbidden to make.
FROZEN_SYNTHETIC_TEXT = (
    "TITLE: Fixture paper\n"
    "\n"
    "ABSTRACT:\n"
    "An abstract sentence.\n"
    "\n"
    "\n"
    "## Methods\n"
    "We enrolled participants.\n"
    "\n"
    "\n"
    "## Results\n"
    "See Table 1.\n"
    "\n"
    "\n"
    "TABLE: Patient characteristics\n"
    "\tT1DM-patients (n=88)\tControls (n=60)\n"
    "Age (years)\t61.3 ± 7.1\t62.3 ± 6.8\n"
    "Body mass index\t25.8 ± 3.9\t25.5 ± 4.2\n"
)

TABLE1_SHAPE = {
    "svanteson_2019": {"width": 4, "rows": 28},
    "de_gonzalo_calvo_2018": {"width": 2, "rows": 25},
    "colom_2018": {"width": 2, "rows": 21},
}


def _norm(text: str) -> str:
    return text.replace("\xa0", " ").replace("\u2009", " ")


def _table_1(tables: list[CapturedTable]) -> CapturedTable:
    for table in tables:
        blob = _norm(f"{table.table_id} {table.caption}").lower()
        if "table 1" in blob or table.table_id.lower() in {"tab1", "table_1", "t1"}:
            return table
    raise AssertionError(
        "no Table 1 among "
        + ", ".join(f"{t.table_id!r}/{t.caption!r}" for t in tables)
    )


def test_paper_document_schema_does_not_grow_tables():
    assert "tables" not in PaperDocument.model_fields


def test_synthetic_xml_text_path_is_byte_identical():
    assert FullTextRetriever._pmc_xml_to_text(SYNTHETIC_XML) == FROZEN_SYNTHETIC_TEXT


def test_synthetic_xml_emits_table_1_grid():
    tables = pmc_xml_to_tables(SYNTHETIC_XML)
    assert FullTextRetriever._pmc_xml_to_text(SYNTHETIC_XML) == FROZEN_SYNTHETIC_TEXT
    table = _table_1(tables)
    assert table.width == 3
    assert len(table.rows) == 2
    assert table.rows[0][0] == "Age (years)"
    assert table.rows[0][1] == "61.3 ± 7.1"


def test_srma_pmc_papers_emit_table_1_with_pdf_shape():
    for study_id, shape in TABLE1_SHAPE.items():
        xml_text = (FIXTURES / f"{study_id}.xml").read_text(encoding="utf-8")
        text_before = FullTextRetriever._pmc_xml_to_text(xml_text)
        tables = pmc_xml_to_tables(xml_text)
        assert FullTextRetriever._pmc_xml_to_text(xml_text) == text_before
        assert tables, f"{study_id} produced no tables"
        table = _table_1(tables)
        assert table.width == shape["width"], study_id
        assert len(table.rows) == shape["rows"], study_id


def test_captured_grid_cells_match_the_existing_tsv_walk():
    import xml.etree.ElementTree as ET

    xml_text = (FIXTURES / "svanteson_2019.xml").read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)
    tables = {t.table_id: t for t in pmc_xml_to_tables(xml_text)}
    for wrap in root.iter("table-wrap"):
        table_el = wrap.find(".//table")
        if table_el is None:
            continue
        captured = tables[wrap.get("id")]
        from_grid = "\n".join(
            "\t".join(row) for row in captured.header_rows + captured.rows
        )
        assert from_grid == _table_to_text(table_el)


async def test_fetch_result_carries_tables_outside_paper_document():
    document = PaperDocument(
        paper_id="pmc:1",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text="TABLE: Patient characteristics\nAge\t1\t2\n",
        document_scope=DocumentScope.FULL_TEXT,
    )
    captured = CapturedTable(
        table_id="Tab1", caption="Table 1 Patient characteristics",
        header_rows=[["", "A", "B"]], rows=[["Age", "1", "2"]],
    )

    class _Retriever(PaperRetriever):
        def __init__(self) -> None:
            self.captured_tables = [captured]

        async def retrieve(self, reference: ReferenceEntry) -> PaperDocument:
            return document

    result = await FetchFullTextTool(_Retriever()).run(document.reference)
    assert result.tables == [captured]
    assert "tables" not in PaperDocument.model_fields
    assert "tables" not in document.model_dump()
