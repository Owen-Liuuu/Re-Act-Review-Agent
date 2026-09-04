"""Local-then-online retrieval: unique match or refuse, never guess."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.retrieval.composite import CompositeRetriever
from react_review.retrieval.labels import origin_label
from react_review.retrieval.local_pdf import LocalPdfRecord, LocalPdfRetriever, local_pdf_path
from react_review.schemas.evidence import IncludedStudy
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry

LI_TITLE = (
    "Li K, Lu S, Li C, et al. Long-term outcomes of minimally "
    "invasive esophagectomy vs. open esophagectomy. "
    "Langenbecks Arch Surg. 2025;410(1):311."
)
CAPOVILLA_TITLE = (
    "Capovilla G, Uzun E, Scarton A, et al. Minimally invasive "
    "Ivor Lewis esophagectomy in the elderly patient. "
    "Front Oncol. 2023;13:1104109."
)
LI_DOI = "10.1007/s00423-025-03877-4"


def _doc(paper_id: str, source: str) -> PaperDocument:
    return PaperDocument(
        paper_id=paper_id,
        reference=ReferenceEntry(title=paper_id),
        full_text="full text",
        document_scope=DocumentScope.FULL_TEXT,
        metadata={"source": source},
    )


class _StubRetriever(PaperRetriever):
    def __init__(self, document=None, *, tables=None) -> None:
        self.document = document
        self.captured_tables = list(tables or [])
        self.calls = 0

    async def retrieve(self, reference: ReferenceEntry):
        self.calls += 1
        return self.document


@pytest.fixture
def pdf_text(monkeypatch, tmp_path):
    path = tmp_path / "li_2025.pdf"
    path.write_bytes(b"%PDF-1.4 fixture")
    monkeypatch.setattr(
        "react_review.retrieval.local_pdf._pdf_text",
        lambda _path: "Li 2025 local full text",
    )
    return path


@pytest.mark.asyncio
async def test_doi_hit_reads_the_upload(pdf_text, tmp_path):
    retriever = LocalPdfRetriever({LI_DOI: pdf_text.name}, base_dir=tmp_path)
    doc = await retriever.retrieve(ReferenceEntry(title="other", doi=LI_DOI))
    assert doc is not None
    assert doc.full_text == "Li 2025 local full text"
    assert doc.metadata["source"] == "local_pdf"
    assert local_pdf_path(doc) == str(pdf_text)
    assert doc.metadata["path"] == str(pdf_text)
    assert "tables" not in doc.model_dump() or not doc.tables
    assert "source_pdf_path" not in doc.model_dump()


@pytest.mark.asyncio
async def test_pmid_hit_when_doi_is_absent(pdf_text, tmp_path):
    retriever = LocalPdfRetriever(records=[LocalPdfRecord(
        path=Path(pdf_text.name), pmid="36726501", title="")], base_dir=tmp_path)
    doc = await retriever.retrieve(ReferenceEntry(title="x", pmid="36726501"))
    assert doc is not None
    assert doc.metadata["source"] == "local_pdf_pmid"


@pytest.mark.asyncio
async def test_strict_title_hit(pdf_text, tmp_path):
    retriever = LocalPdfRetriever(records=[LocalPdfRecord(
        path=Path(pdf_text.name), title=LI_TITLE)], base_dir=tmp_path)
    doc = await retriever.retrieve(ReferenceEntry(title=LI_TITLE))
    assert doc is not None
    assert doc.metadata["source"] == "local_pdf_title"


@pytest.mark.asyncio
async def test_capovilla_does_not_match_a_li_upload(pdf_text, tmp_path):
    """The silent-wrong-paper shape: similar MIE-in-elderly titles, different works."""
    retriever = LocalPdfRetriever.from_included(
        [IncludedStudy(
            study_id="li_2025", review_citation=LI_TITLE,
            doi=LI_DOI, source_pdf=pdf_text.name)],
        base_dir=tmp_path,
    )
    capovilla = ReferenceEntry(title=CAPOVILLA_TITLE, pmid="36726501")
    assert await retriever.retrieve(capovilla) is None


@pytest.mark.asyncio
async def test_ambiguous_doi_refuses_rather_than_picking(pdf_text, tmp_path):
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF")
    retriever = LocalPdfRetriever(records=[
        LocalPdfRecord(path=Path(pdf_text.name), doi=LI_DOI),
        LocalPdfRecord(path=Path(other.name), doi=LI_DOI),
    ], base_dir=tmp_path)
    assert await retriever.retrieve(ReferenceEntry(title="x", doi=LI_DOI)) is None


@pytest.mark.asyncio
async def test_two_statistic_like_titles_that_both_match_refuse(pdf_text, tmp_path):
    other = tmp_path / "copy.pdf"
    other.write_bytes(b"%PDF")
    retriever = LocalPdfRetriever(records=[
        LocalPdfRecord(path=Path(pdf_text.name), title=LI_TITLE),
        LocalPdfRecord(path=Path(other.name), title=LI_TITLE),
    ], base_dir=tmp_path)
    assert await retriever.retrieve(ReferenceEntry(title=LI_TITLE)) is None


@pytest.mark.asyncio
async def test_slug_matches_named_upload_without_doi(pdf_text, tmp_path):
    retriever = LocalPdfRetriever.from_included(
        [IncludedStudy(study_id="li_2025", source_pdf=pdf_text.name)],
        base_dir=tmp_path,
    )
    doc = await retriever.retrieve(ReferenceEntry(title="li_2025"))
    assert doc is not None
    assert doc.metadata["source"] == "local_pdf_title"


@pytest.mark.asyncio
async def test_full_citation_is_not_a_slug_guess_at_the_filename(pdf_text, tmp_path):
    retriever = LocalPdfRetriever.from_included(
        [IncludedStudy(study_id="li_2025", source_pdf=pdf_text.name)],
        base_dir=tmp_path,
    )
    assert await retriever.retrieve(ReferenceEntry(title=CAPOVILLA_TITLE)) is None


@pytest.mark.asyncio
async def test_composite_uses_the_first_document_and_skips_the_rest():
    local = _StubRetriever(_doc("local", "local_pdf"))
    online = _StubRetriever(_doc("online", "pmc"), tables=["table"])
    composite = CompositeRetriever([local, online])
    doc = await composite.retrieve(ReferenceEntry(title="x", doi=LI_DOI))
    assert doc.paper_id == "local"
    assert local.calls == 1 and online.calls == 0
    assert composite.captured_tables == []


@pytest.mark.asyncio
async def test_composite_falls_through_on_local_miss():
    local = _StubRetriever(None)
    online = _StubRetriever(_doc("online", "pmc"), tables=["grid"])
    composite = CompositeRetriever([local, online])
    doc = await composite.retrieve(ReferenceEntry(title=CAPOVILLA_TITLE))
    assert doc.paper_id == "online"
    assert local.calls == 1 and online.calls == 1
    assert composite.captured_tables == ["grid"]


@pytest.mark.asyncio
async def test_composite_all_miss_returns_none():
    composite = CompositeRetriever([_StubRetriever(None), _StubRetriever(None)])
    assert await composite.retrieve(ReferenceEntry(title="x")) is None


def test_origin_labels_are_the_contract_phrases():
    assert origin_label("local_pdf") == "uploaded (matched by DOI)"
    assert origin_label("local_pdf_pmid") == "uploaded (matched by PMID)"
    assert origin_label("local_pdf_title") == "uploaded (matched by title)"
    assert origin_label("pmc") == "PMC (online)"
    assert origin_label("") == "not retrieved"
    assert origin_label("fallback-metadata-only") == "not retrieved"
