"""The retrieval chain must say how much of a paper it actually obtained."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from react_review.agents.collector import _provenance
from react_review.core.config import AppConfig, PubMedSettings
from react_review.pipeline.factory import create_pipeline
from react_review.orchestrator.audit_pipeline import AuditPipeline
from react_review.retrieval.local_pdf import LocalPdfRetriever
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.package import EvidencePackage
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.fulltext_retriever import FullTextRetriever
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.mock_impl import MockPaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.store import EvidencePackageStore
from react_review.tools import build_catalogue
from react_review.tools.extract import FetchFullTextTool


_REF = ReferenceEntry(
    title="Scope test paper", authors=["A. Author"], year=2020,
    journal="Journal", doi="10.1000/scope",
)


class _StaticRetriever(PaperRetriever):
    def __init__(self, document: PaperDocument | None) -> None:
        self.document = document

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        return self.document


def _retriever(*, unpaywall_enabled: bool = True) -> FullTextRetriever:
    return FullTextRetriever(
        PubMedSettings(email="pubmed@example.org"),
        unpaywall_email="unpaywall@example.org",
        unpaywall_enabled=unpaywall_enabled,
    )


def test_legacy_document_without_scope_is_unknown_not_full_text():
    document = PaperDocument.model_validate({
        "paper_id": "legacy", "reference": _REF.model_dump(),
        "full_text": "A historical artifact did not declare its scope.",
    })

    assert document.document_scope is DocumentScope.UNKNOWN
    assert document.document_scope is not DocumentScope.FULL_TEXT
    assert "document_scope" not in document.model_dump(mode="json")


@pytest.mark.asyncio
async def test_metadata_fallback_exists_but_is_not_retrieved():
    document = FullTextRetriever._fallback_document(_REF)
    result = await FetchFullTextTool(_StaticRetriever(document)).run(_REF)

    assert document.document_scope is DocumentScope.METADATA_ONLY
    assert result.document is document
    assert result.retrieved is False


@pytest.mark.asyncio
async def test_abstract_is_retrieved_without_pretending_to_be_full_text(monkeypatch):
    retriever = _retriever()
    monkeypatch.setattr(retriever, "_find_pubmed_pmid", AsyncMock(return_value="123"))
    monkeypatch.setattr(
        retriever, "_fetch_abstract",
        AsyncMock(return_value="This abstract states the study outcome."),
    )

    document = await retriever._try_pubmed_abstract(_REF)
    result = await FetchFullTextTool(_StaticRetriever(document)).run(_REF)

    assert document is not None
    assert document.document_scope is DocumentScope.ABSTRACT_ONLY
    assert result.retrieved is True


@pytest.mark.asyncio
async def test_pmc_document_is_explicit_full_text(monkeypatch):
    retriever = _retriever()
    monkeypatch.setattr(retriever, "_find_pmc_id", AsyncMock(return_value="PMC123"))
    monkeypatch.setattr(
        retriever, "_fetch_pmc_fulltext",
        AsyncMock(return_value="## Methods\n" + "method data " * 200),
    )

    document = await retriever._try_pmc(_REF)

    assert document is not None
    assert document.document_scope is DocumentScope.FULL_TEXT


@pytest.mark.asyncio
async def test_unpaywall_html_document_is_explicit_full_text(monkeypatch):
    retriever = _retriever()
    monkeypatch.setattr(
        retriever, "_unpaywall_find_oa",
        AsyncMock(return_value=("https://example.org/paper", "html")),
    )
    monkeypatch.setattr(
        retriever, "_download_html_text",
        AsyncMock(return_value="## Results\n" + "result data " * 200),
    )

    document = await retriever._try_unpaywall(_REF)

    assert document is not None
    assert document.document_scope is DocumentScope.FULL_TEXT


@pytest.mark.asyncio
async def test_openalex_abstract_is_explicit_abstract_only(monkeypatch):
    retriever = _retriever()
    words = [f"word{i}" for i in range(30)]
    work = {
        "id": "https://openalex.org/W1",
        "title": _REF.title,
        "abstract_inverted_index": {word: [i] for i, word in enumerate(words)},
    }
    monkeypatch.setattr(retriever, "_openalex_fetch_work", AsyncMock(return_value=work))
    monkeypatch.setattr(retriever, "_openalex_best_pdf_url", lambda _work: "")

    document = await retriever._try_openalex(_REF)

    assert document is not None
    assert document.document_scope is DocumentScope.ABSTRACT_ONLY


@pytest.mark.asyncio
async def test_openalex_pdf_document_is_explicit_full_text(monkeypatch):
    retriever = _retriever()
    work = {"id": "https://openalex.org/W1", "title": _REF.title}
    monkeypatch.setattr(retriever, "_openalex_fetch_work", AsyncMock(return_value=work))
    monkeypatch.setattr(
        retriever, "_openalex_best_pdf_url",
        lambda _work: "https://example.org/paper.pdf",
    )
    monkeypatch.setattr(
        retriever, "_download_and_parse_pdf",
        AsyncMock(return_value="## Results\n" + "result data " * 200),
    )

    document = await retriever._try_openalex(_REF)

    assert document is not None
    assert document.document_scope is DocumentScope.FULL_TEXT


@pytest.mark.asyncio
async def test_unpaywall_disabled_makes_no_unpaywall_request(monkeypatch):
    retriever = _retriever(unpaywall_enabled=False)
    request = AsyncMock(return_value=("https://example.org/paper.pdf", "pdf"))
    monkeypatch.setattr(retriever, "_unpaywall_find_oa", request)

    assert await retriever._try_unpaywall(_REF) is None
    request.assert_not_awaited()


@pytest.mark.asyncio
async def test_mock_retriever_declares_full_text():
    document = await MockPaperRetriever().retrieve(_REF)

    assert document is not None
    assert document.document_scope is DocumentScope.FULL_TEXT


@pytest.mark.asyncio
async def test_local_pdf_retriever_declares_full_text(tmp_path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"fixture")
    monkeypatch.setattr(
        "react_review.retrieval.local_pdf._pdf_text",
        lambda _path: "Locally extracted source text.",
    )
    retriever = LocalPdfRetriever({_REF.doi: pdf})

    document = await retriever.retrieve(_REF)

    assert document is not None
    assert document.document_scope is DocumentScope.FULL_TEXT


def test_scope_is_persisted_in_evidence_package_processing_record(tmp_path):
    document = PaperDocument(
        paper_id="p", reference=_REF, full_text="full paper",
        document_scope=DocumentScope.FULL_TEXT,
        metadata={"source": "local_pdf", "path": "paper.pdf"},
    )
    package = EvidencePackage(
        run_id="scope-round-trip",
        processing_records=[AgentRun(
            agent="collector", status="finished",
            steps=[StepRecord(
                index=0, tool="fetch_fulltext", observation=_provenance(document),
            )],
        )],
    )

    store = EvidencePackageStore(tmp_path)
    store.save(package)
    loaded = store.load(package.run_id)

    assert loaded.processing_records[0].steps[0].observation["document_scope"] == "full_text"


def test_real_catalogue_wires_unpaywall_enabled_flag():
    config = AppConfig(
        mock_mode=False,
        unpaywall={"enabled": False, "email": "configured@example.org"},
    )

    fetch = build_catalogue(config).get("fetch_fulltext")

    assert fetch._retriever._unpaywall_enabled is False


def test_real_pipeline_wires_unpaywall_enabled_flag():
    config = AppConfig(
        mock_mode=False,
        unpaywall={"enabled": False, "email": "configured@example.org"},
    )

    pipeline = create_pipeline(config)

    assert pipeline._paper_retriever._unpaywall_enabled is False


def test_collection_summary_reports_each_document_scope_separately():
    papers = [
        {"study_id": "a", "subject": "a.pdf", "n_claims": 1, "n_found": 1,
         "warnings": [], "document_scope": "full_text"},
        {"study_id": "b", "subject": "doi:b", "n_claims": 1, "n_found": 1,
         "warnings": [], "document_scope": "abstract_only"},
        {"study_id": "c", "subject": "doi:c", "n_claims": 1, "n_found": 0,
         "warnings": ["missing"], "document_scope": "metadata_only"},
    ]

    rendered = AuditPipeline._render_collection(papers, 3)

    assert "full_text       1" in rendered
    assert "abstract_only   1" in rendered
    assert "metadata_only   1" in rendered
