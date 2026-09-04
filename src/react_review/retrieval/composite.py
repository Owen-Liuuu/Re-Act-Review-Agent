"""Try local files first, then the online chain. First non-None document wins.

A miss on the local side is not a guess: the next retriever is asked. Matching
the uploaded file for paper A to citation B is worse than fetching B online.
"""
from __future__ import annotations

from collections.abc import Sequence

from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry


class CompositeRetriever(PaperRetriever):
    """Walk ``retrievers`` in order; return the first document that is not None.

    ``captured_tables`` mirrors whichever retriever produced the document, so
    ``FetchFullTextTool`` keeps seeing PMC grids after an online hit and the
    uploaded PDF's ``find_tables`` grids after a local hit.
    """

    def __init__(self, retrievers: Sequence[PaperRetriever]) -> None:
        self._retrievers = list(retrievers)
        self.captured_tables: list = []

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        self.captured_tables = []
        for retriever in self._retrievers:
            document = await retriever.retrieve(reference)
            if document is not None:
                self.captured_tables = list(
                    getattr(retriever, "captured_tables", []) or [])
                return document
        return None
