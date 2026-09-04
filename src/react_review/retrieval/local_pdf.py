"""A PaperRetriever that reads LOCAL source PDFs (no network).

For running the audit on a benchmark whose source papers are copyrighted and
kept locally (mapped by DOI via included_studies ``source_pdf``), instead of the
network fetch chain. Same ``PaperRetriever`` interface, so the Collector's
``fetch_fulltext`` tool wraps it transparently.

Matching is exact and unique, in this order: DOI, PMID, strict title. Zero or
two hits is a refusal — the caller then tries the next retriever. Guessing
which upload belongs to which citation is the Capovilla mismatch: a silent
wrong paper, worse than fetching the right one online.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import structlog

from react_review.normalize.doi import normalize_doi, normalize_pmid, printed_pmid
from react_review.retrieval.pdf_tables import tables_from_pdf
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)

# Below this, a "title" is a slug or a fragment, not a citation we can trust.
_TITLE_MIN_CHARS = 24
# High enough that two MIE-in-the-elderly papers do not collide. Exact
# normalised equality also counts, so a full-citation reprint still hits.
_TITLE_RATIO = 0.95
_SLUG = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d{4}$")
_KIND = {"doi": "local_pdf", "pmid": "local_pdf_pmid", "title": "local_pdf_title"}


def _pdf_text(path: Path) -> str:
    import fitz  # PyMuPDF

    from react_review.normalize.text import clean_pdf_text

    doc = fitz.open(str(path))
    try:
        return clean_pdf_text("\n\n".join(doc[i].get_text() for i in range(len(doc))))
    finally:
        doc.close()


def local_pdf_path(document) -> str:
    """The uploaded file this document was read from, or empty.

    ``PaperDocument`` is inside the evidence-adequacy hash, so the path cannot
    be a new field. It rides in ``metadata["path"]``, which A2 reopens. Text
    extraction does not delete or rewrite the file.
    """
    if document is None:
        return ""
    return str((getattr(document, "metadata", None) or {}).get("path") or "")


def _norm_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


@dataclass(frozen=True)
class LocalPdfRecord:
    """One uploaded file and the keys that may uniquely identify it."""

    path: Path
    doi: str = ""
    pmid: str = ""
    title: str = ""
    names: tuple[str, ...] = field(default_factory=tuple)


class LocalPdfRetriever(PaperRetriever):
    """Resolve a reference to a local PDF by DOI, then PMID, then strict title."""

    def __init__(
        self,
        doi_to_path: dict[str, str | Path] | None = None,
        base_dir: Path | str | None = None,
        records: list[LocalPdfRecord] | None = None,
    ) -> None:
        self._base = Path(base_dir) if base_dir else None
        self._records: list[LocalPdfRecord] = list(records or [])
        self.captured_tables: list = []
        for doi, path in (doi_to_path or {}).items():
            key = normalize_doi(doi)
            if key:
                self._records.append(LocalPdfRecord(path=Path(path), doi=key))

    @classmethod
    def from_included(cls, studies, base_dir: Path | str | None = None) -> "LocalPdfRetriever":
        """Build the catalog from ``included_studies`` rows that name a PDF."""
        records: list[LocalPdfRecord] = []
        for study in studies:
            rel = getattr(study, "source_pdf", "") or ""
            if not rel:
                continue
            path = Path(rel)
            citation = getattr(study, "review_citation", "") or ""
            study_id = getattr(study, "study_id", "") or ""
            names = tuple(n for n in (_slug(study_id), _slug(path.stem)) if n)
            records.append(LocalPdfRecord(
                path=path,
                doi=normalize_doi(getattr(study, "doi", "") or ""),
                pmid=printed_pmid(citation),
                title=citation,
                names=names,
            ))
        return cls(records=records, base_dir=base_dir)

    def _resolve_path(self, rel: Path) -> Path:
        return rel if rel.is_absolute() or self._base is None else self._base / rel

    def _match(self, reference: ReferenceEntry) -> tuple[LocalPdfRecord, str] | None:
        doi = normalize_doi(reference.doi or "")
        if doi:
            hits = [r for r in self._records if r.doi == doi]
            if len(hits) > 1:
                logger.debug("local_pdf_ambiguous_doi", doi=doi, n=len(hits))
                return None
            if len(hits) == 1:
                return hits[0], "doi"

        pmid = normalize_pmid(reference.pmid or "") or printed_pmid(reference.title or "")
        if pmid:
            hits = [r for r in self._records if r.pmid == pmid]
            if len(hits) > 1:
                logger.debug("local_pdf_ambiguous_pmid", pmid=pmid, n=len(hits))
                return None
            if len(hits) == 1:
                return hits[0], "pmid"

        title_hits = self._title_hits(reference.title or "")
        if len(title_hits) > 1:
            logger.debug("local_pdf_ambiguous_title", n=len(title_hits))
            return None
        if len(title_hits) == 1:
            return title_hits[0], "title"

        slug_hits = self._slug_hits(reference.title or "")
        if len(slug_hits) > 1:
            logger.debug("local_pdf_ambiguous_slug", n=len(slug_hits))
            return None
        if len(slug_hits) == 1:
            return slug_hits[0], "title"
        return None

    def _title_hits(self, title: str) -> list[LocalPdfRecord]:
        query = _norm_title(title)
        if len(query) < _TITLE_MIN_CHARS:
            return []
        hits: list[LocalPdfRecord] = []
        seen: set[int] = set()
        for record in self._records:
            candidate = _norm_title(record.title)
            if len(candidate) < _TITLE_MIN_CHARS:
                continue
            if candidate == query or SequenceMatcher(None, candidate, query).ratio() >= _TITLE_RATIO:
                key = id(record)
                if key not in seen:
                    seen.add(key)
                    hits.append(record)
        return hits

    def _slug_hits(self, title: str) -> list[LocalPdfRecord]:
        """Exact study-id / filename-stem identity, never a fuzzy title guess.

        Used when the CSV has a file but no DOI: the user named the upload.
        A full citation is not a slug, so Capovilla's title cannot hit Li's file.
        """
        query = _slug(title)
        if not _SLUG.match(query):
            return []
        hits: list[LocalPdfRecord] = []
        seen: set[int] = set()
        for record in self._records:
            if query in record.names:
                key = id(record)
                if key not in seen:
                    seen.add(key)
                    hits.append(record)
        return hits

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        self.captured_tables = []
        matched = self._match(reference)
        if matched is None:
            logger.debug("local_pdf_no_mapping", doi=reference.doi, pmid=reference.pmid)
            return None
        record, matched_by = matched
        path = self._resolve_path(record.path)
        if not path.is_file():
            logger.warning("local_pdf_missing", path=str(path))
            return None
        text = _pdf_text(path)
        tables = tables_from_pdf(path)
        self.captured_tables = list(tables)
        opened = str(path)
        return PaperDocument(
            paper_id=reference.doi or opened,
            reference=reference,
            full_text=text,
            document_scope=DocumentScope.FULL_TEXT,
            tables=tables,
            metadata={"source": _KIND[matched_by], "path": opened},
        )
