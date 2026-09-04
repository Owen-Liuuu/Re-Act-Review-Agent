"""Extract stage: full-text retrieval."""
from __future__ import annotations

from pathlib import Path

from react_review.retrieval.local_pdf import local_pdf_path
from react_review.steps.data_extraction.schemas import DocumentScope
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.base import Tool, ToolStage
from react_review.tools.models import FetchResult


class FetchFullTextTool(Tool):
    """Retrieve full text via the 4-tier chain (PMC → Unpaywall → OpenAlex → abstract)."""

    name = "fetch_fulltext"
    stage = ToolStage.EXTRACT
    input_model = ReferenceEntry
    output_model = FetchResult

    def __init__(self, retriever: PaperRetriever, *, source_tables=None) -> None:
        self._retriever = retriever
        self._source_tables = source_tables

    async def run(self, payload: ReferenceEntry) -> FetchResult:
        doc = await self._retriever.retrieve(payload)
        retrieved = (
            doc is not None
            and doc.document_scope is not DocumentScope.METADATA_ONLY
        )
        tables = list(getattr(self._retriever, "captured_tables", []) or [])
        if doc is not None:
            existing = list(getattr(doc, "tables", None) or [])
            if tables and not existing:
                doc = doc.model_copy(update={"tables": tables})
            elif existing and not tables:
                tables = existing
        if (
            self._source_tables is not None
            and doc is not None
            and not tables
            and str((doc.metadata or {}).get("source") or "").startswith("local_pdf")
            and Path(local_pdf_path(doc)).is_file()
        ):
            captured = await self._source_tables.capture(doc.full_text)
            if captured:
                tables = list(captured)
                doc = doc.model_copy(update={"tables": tables})
        return FetchResult(
            reference=payload, retrieved=retrieved, document=doc, tables=tables,
        )
