"""A1: keep the uploaded PDF for A2. Text only. No new PaperDocument field."""
from __future__ import annotations

import inspect

import pytest

from react_review.retrieval.local_pdf import (
    LocalPdfRetriever,
    _pdf_text,
    local_pdf_path,
)
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.fulltext_retriever import FullTextRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry

_DOI = "10.1000/a1-retain"


def test_paper_document_has_no_source_pdf_path_field():
    assert "source_pdf_path" not in PaperDocument.model_fields


def test_pdf_text_extracts_plain_text_and_does_not_parse_tables():
    source = inspect.getsource(_pdf_text)
    assert "get_text" in source
    assert "find_tables" not in source
    assert "get_tables" not in source
    assert "CapturedTable" not in source


@pytest.mark.asyncio
async def test_retrieve_keeps_the_original_file_and_exposes_its_path(
    tmp_path, monkeypatch,
):
    pdf = tmp_path / "paper.pdf"
    payload = b"%PDF-1.4 A1 retain fixture"
    pdf.write_bytes(payload)
    monkeypatch.setattr(
        "react_review.retrieval.local_pdf._pdf_text",
        lambda _path: "extracted text",
    )

    document = await LocalPdfRetriever({_DOI: pdf}).retrieve(
        ReferenceEntry(title="t", doi=_DOI),
    )

    assert document is not None
    assert local_pdf_path(document) == str(pdf)
    assert document.metadata["path"] == str(pdf)
    assert pdf.is_file()
    assert pdf.read_bytes() == payload
    dumped = document.model_dump()
    assert "source_pdf_path" not in dumped
    assert "tables" not in dumped


def test_fallback_document_has_no_upload_path():
    document = FullTextRetriever._fallback_document(ReferenceEntry(title="t"))
    assert local_pdf_path(document) == ""
    assert "path" not in document.metadata
    assert "source_pdf_path" not in document.model_dump()
