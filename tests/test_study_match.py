"""Tests for study-id resolution and the local PDF retriever."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.csv_io import load_included_studies
from react_review.retrieval.local_pdf import LocalPdfRetriever
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.study_match import (
    build_reference_resolver,
    resolve_studies,
    resolve_study,
)

BENCH = Path(__file__).resolve().parents[1] / "eval" / "benchmark"
STUDIES = load_included_studies(BENCH / "included_studies.csv")


@pytest.mark.parametrize(
    "parser_sid, canonical",
    [
        ("ahmad_2022", "ahmad_2022"),
        ("yaz_2011", "yazici_2011"),                 # non-ASCII slug prefix
        ("de_2018", "de_gonzalo_calvo_2018"),        # particle name, year disambiguates
        ("colom_2018", "colom_2018"),                # same year as de_… but surname differs
        ("iacobellis_2014", "iacobellis_2014"),
    ],
)
def test_resolve_study(parser_sid, canonical):
    s = resolve_study(parser_sid, STUDIES)
    assert s is not None and s.study_id == canonical


def test_resolve_study_unknown_returns_none():
    assert resolve_study("wongabongo_1999", STUDIES) is None


def test_resolve_studies_relabels_and_maps():
    items = [
        ReviewDataItem(study_id="yaz_2011", group="t1dm", field_type="eat_thickness",
                       value="3.30", unit="mm"),
        ReviewDataItem(study_id="unknownstudy_2000", group="t1dm",
                       field_type="bmi", value="24"),
    ]
    relabelled, sid_map = resolve_studies(items, STUDIES)
    assert relabelled[0].study_id == "yazici_2011"      # relabelled
    assert relabelled[1].study_id == "unknownstudy_2000"  # unresolved kept as-is
    assert "yazici_2011" in sid_map

    ref = build_reference_resolver(sid_map)("yazici_2011")
    assert isinstance(ref, ReferenceEntry)
    assert ref.doi == "10.1007/s12020-011-9478-x"


# --- LocalPdfRetriever ---

@pytest.mark.asyncio
async def test_local_retriever_no_mapping_returns_none():
    r = LocalPdfRetriever({}, base_dir=BENCH)
    assert await r.retrieve(ReferenceEntry(title="x", doi="10.1/x")) is None


@pytest.mark.asyncio
async def test_local_retriever_missing_file_returns_none():
    r = LocalPdfRetriever({"10.1/x": "pdf/does_not_exist.pdf"}, base_dir=BENCH)
    assert await r.retrieve(ReferenceEntry(title="x", doi="10.1/x")) is None


@pytest.mark.asyncio
async def test_local_retriever_reads_real_pdf_if_present():
    ahmad = next(s for s in STUDIES if s.study_id == "ahmad_2022")
    pdf_path = BENCH / ahmad.source_pdf
    if not pdf_path.is_file():
        pytest.skip("benchmark source PDFs are git-ignored / not present locally")
    r = LocalPdfRetriever({ahmad.doi: ahmad.source_pdf}, base_dir=BENCH)
    doc = await r.retrieve(ReferenceEntry(title="Ahmad", doi=ahmad.doi))
    assert doc is not None
    assert len(doc.full_text) > 1000
    assert doc.metadata["source"] == "local_pdf"
