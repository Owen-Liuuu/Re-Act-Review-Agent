"""A PaperRetriever that reads LOCAL source PDFs (no network).

For running the audit on a benchmark whose source papers are copyrighted and
kept locally (mapped by DOI via included_studies ``source_pdf``), instead of the
network fetch chain. Same ``PaperRetriever`` interface, so the Collector's
``fetch_fulltext`` tool wraps it transparently.
"""
from __future__ import annotations

from pathlib import Path

import structlog

from react_review.normalize.doi import normalize_doi
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)


def _pdf_text(path: Path) -> str:
    import fitz  # PyMuPDF

    from react_review.normalize.text import clean_pdf_text

    doc = fitz.open(str(path))
    try:
        return clean_pdf_text("\n\n".join(doc[i].get_text() for i in range(len(doc))))
    finally:
        doc.close()


class LocalPdfRetriever(PaperRetriever):
    """Resolve a reference to a local PDF by normalised DOI and read its text."""

    def __init__(self, doi_to_path: dict[str, str | Path], base_dir: Path | str | None = None) -> None:
        self._base = Path(base_dir) if base_dir else None
        self._map: dict[str, Path] = {}
        for doi, p in doi_to_path.items():
            key = normalize_doi(doi)
            if key:
                self._map[key] = Path(p)

    def _resolve_path(self, rel: Path) -> Path:
        return rel if rel.is_absolute() or self._base is None else self._base / rel

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        key = normalize_doi(reference.doi or "")
        rel = self._map.get(key)
        if rel is None:
            logger.debug("local_pdf_no_mapping", doi=reference.doi)
            return None
        path = self._resolve_path(rel)
        if not path.is_file():
            logger.warning("local_pdf_missing", path=str(path))
            return None
        text = _pdf_text(path)
        return PaperDocument(
            paper_id=reference.doi or str(path),
            reference=reference,
            full_text=text,
            metadata={"source": "local_pdf", "path": str(path)},
        )
