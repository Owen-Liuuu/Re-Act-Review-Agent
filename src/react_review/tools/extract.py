"""Extract stage: full-text retrieval."""
from __future__ import annotations

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

    def __init__(self, retriever: PaperRetriever) -> None:
        self._retriever = retriever

    async def run(self, payload: ReferenceEntry) -> FetchResult:
        doc = await self._retriever.retrieve(payload)
        retrieved = (
            doc is not None
            and doc.document_scope is not DocumentScope.METADATA_ONLY
        )
        return FetchResult(reference=payload, retrieved=retrieved, document=doc)
