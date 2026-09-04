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
    "capovilla_2023": {"width": 4, "rows": 37},
    "li_2025": {"width": 10, "rows": 48},
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


def test_paper_document_carries_tables_alongside_full_text():
    assert "tables" in PaperDocument.model_fields
    document = PaperDocument(
        paper_id="pmc:1",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text="TABLE: Patient characteristics\nAge\t1\t2\n",
        document_scope=DocumentScope.FULL_TEXT,
    )
    dumped = document.model_dump()
    assert dumped["full_text"]
    assert "tables" not in dumped
    assert "source_pdf_path" not in dumped


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


def _naive_tsv(table_el) -> str:
    """Document-order cells, no span expansion — the ``full_text`` walk."""
    rows: list[str] = []
    for tr in table_el.iter("tr"):
        cells = [
            "".join(cell.itertext()).strip()
            for cell in tr if cell.tag in ("th", "td")
        ]
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows)


def test_table_to_text_still_walks_cells_in_document_order():
    """B0 must not change ``_table_to_text``: it feeds ``full_text``."""
    import xml.etree.ElementTree as ET

    for xml_path in sorted(FIXTURES.glob("*.xml")):
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        for wrap in root.iter("table-wrap"):
            table_el = wrap.find(".//table")
            if table_el is None:
                continue
            assert _table_to_text(table_el) == _naive_tsv(table_el), xml_path.name


def test_expanded_grid_every_row_matches_column_path_width():
    for xml_path in sorted(FIXTURES.glob("*.xml")):
        for table in pmc_xml_to_tables(xml_path.read_text(encoding="utf-8")):
            n = len(table.column_paths())
            assert n == table.width, (xml_path.name, table.table_id)
            for i, row in enumerate(table.header_rows + table.rows):
                assert len(row) == n, (
                    xml_path.name, table.table_id, i, len(row), n)


def _path_key(text: str) -> str:
    return (
        text.replace("\u2009", "")
        .replace("\xa0", " ")
        .replace("\u2013", "-")
        .replace("\u2212", "-")
    )


def _li_2025_tables() -> dict[str, CapturedTable]:
    xml_text = (FIXTURES / "li_2025.xml").read_text(encoding="utf-8")
    return {t.table_id: t for t in pmc_xml_to_tables(xml_text)}


def test_li_2025_table_1_column_paths_after_span_expand():
    import xml.etree.ElementTree as ET

    table = _li_2025_tables()["Tab1"]
    assert [_path_key(p) for p in table.column_paths()] == [
        "Characteristic",
        "Total(n=469)",
        "Before PSM / MIE (n=358)",
        "Before PSM / OE(n=111)",
        "P value",
        "Smd",
        "After PSM / MIE (n=92)",
        "After PSM / OE(n=55)",
        "P value",
        "Smd",
    ]
    root = ET.fromstring((FIXTURES / "li_2025.xml").read_text(encoding="utf-8"))
    wrap = next(w for w in root.iter("table-wrap") if w.get("id") == "Tab1")
    tsv_header0 = _table_to_text(wrap.find(".//table")).splitlines()[0].split("\t")
    assert len(tsv_header0) == 8
    assert len(table.column_paths()) == 10


def test_li_2025_median_row_sits_under_those_column_paths():
    table = _li_2025_tables()["Tab1"]
    paths = [_path_key(p) for p in table.column_paths()]
    row = next(
        r for r in table.rows
        if _path_key(r[0]).lower().startswith("median")
    )
    by_col = dict(zip(paths, (_path_key(c) for c in row)))
    assert by_col["Total(n=469)"] == "73(70-88)"
    assert by_col["After PSM / MIE (n=92)"] == "73(70-83)"
    assert by_col["Before PSM / MIE (n=358)"] == "73(70-85)"
    assert by_col["After PSM / OE(n=55)"] == "72(70-88)"


def test_li_2025_table_2_and_3_column_paths_after_span_expand():
    tables = _li_2025_tables()
    assert [_path_key(p) for p in tables["Tab2"].column_paths()] == [
        "Characteristic",
        "Total(n=469)",
        "Before PSM / MIE (n=358)",
        "Before PSM / OE(n=111)",
        "P value",
        "After PSM / MIE (n=92)",
        "After PSM / OE(n=55)",
        "P value",
    ]
    assert [_path_key(p) for p in tables["Tab3"].column_paths()] == [
        "Adverse events",
        "Before PSM / MIE(n=358)",
        "Before PSM / OE(n=111)",
        "Before PSM / P value",
        "After PSM / MIE(n=92)",
        "After PSM / OE(n=55)",
        "After PSM / P value",
    ]


SPAN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <article-title>Span fixture</article-title>
  <table-wrap id="Tab1">
    <label>Table 1</label>
    <caption><p>Groups</p></caption>
    <table>
      <thead>
        <tr>
          <th rowspan="2">Characteristic</th>
          <th></th>
          <th colspan="2">Before PSM</th>
          <th rowspan="2">P value</th>
        </tr>
        <tr>
          <th>Total</th>
          <th>MIE</th>
          <th>OE</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Age</td><td>1</td><td>2</td><td>3</td><td>0.1</td>
        </tr>
      </tbody>
    </table>
  </table-wrap>
</article>
"""


def test_colspan_rowspan_expand_to_a_rectangle_without_changing_tsv():
    import xml.etree.ElementTree as ET

    tables = pmc_xml_to_tables(SPAN_XML)
    table = tables[0]
    assert table.column_paths() == [
        "Characteristic", "Total",
        "Before PSM / MIE", "Before PSM / OE", "P value",
    ]
    assert table.rows[0] == ["Age", "1", "2", "3", "0.1"]
    root = ET.fromstring(SPAN_XML)
    table_el = root.find(".//table")
    tsv = _table_to_text(table_el)
    assert tsv == _naive_tsv(table_el)
    # Document-order TSV still has the unexpanded header cell counts.
    header0 = tsv.splitlines()[0].split("\t")
    assert len(header0) == 4
    assert len(table.column_paths()) == 5



async def test_fetch_result_copies_tables_onto_the_document():
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
    assert result.document is not None
    assert result.document.tables == [captured]
    assert result.document.full_text == document.full_text
