"""How a retrieved paper is named to a reader — upload vs online vs missing.

Kept out of the retrievers themselves so the report, the checkpoint title and
the web log all print the same words, and so a change of phrasing cannot drift
between those three surfaces.
"""
from __future__ import annotations

# Retriever metadata ``source`` values that mean "we opened a local file".
_UPLOADED = {
    "local_pdf": "DOI",
    "local_pdf_pmid": "PMID",
    "local_pdf_title": "title",
}


def origin_label(retriever_kind: str) -> str:
    """One short phrase: where this paper's text came from.

    ``uploaded (matched by DOI)`` / ``PMC (online)`` / ``not retrieved`` —
    the three shapes a mixed upload+online run has to make visible.
    """
    kind = (retriever_kind or "").strip()
    if not kind or kind == "fallback-metadata-only":
        return "not retrieved"
    if kind in _UPLOADED:
        return f"uploaded (matched by {_UPLOADED[kind]})"
    if kind == "pmc":
        return "PMC (online)"
    if kind == "unpaywall":
        return "Unpaywall (online)"
    if kind.startswith("openalex"):
        return "OpenAlex (online)"
    if kind == "pubmed_abstract":
        return "PubMed abstract (online)"
    return kind
