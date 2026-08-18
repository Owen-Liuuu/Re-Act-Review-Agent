"""Deterministic DOI normalization for matching."""
from __future__ import annotations

import re

_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:")
_DOI_IN_TEXT = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)
_PMID_IN_TEXT = re.compile(r"\bPMID[:\s]*([0-9]{5,8})\b", re.I)


def normalize_doi(doi: str | None) -> str:
    """Lower-case and strip the URL / ``doi:`` prefix. '' for a missing DOI."""
    d = (doi or "").strip().lower()
    for p in _PREFIXES:
        if d.startswith(p):
            d = d[len(p):]
            break
    return d.strip().rstrip("/")


def _doi_tail(raw: str) -> str:
    return normalize_doi(raw.rstrip(").,;]}"))


def dois_printed_in(text: str) -> list[str]:
    """Every DOI that actually appears in ``text``, in order."""
    out: list[str] = []
    for match in _DOI_IN_TEXT.finditer(text or ""):
        doi = _doi_tail(match.group(0))
        if doi:
            out.append(doi)
    return out


def printed_doi(candidate: str | None, citation: str, refs_text: str = "") -> str:
    """Keep a DOI only when the review printed it.

    The model may return a well-formed DOI that the reference never showed
    (a Frontiers article number rewritten as ``10.3389/…``). Those are dropped.
    A DOI the model omitted is copied from the citation line when that line
    prints exactly one.
    """
    cand = normalize_doi(candidate)
    blob = f"{citation or ''}\n{refs_text or ''}".lower().replace(" ", "")
    if cand and cand.replace(" ", "") in blob:
        return cand
    found = dois_printed_in(citation or "")
    return found[0] if len(found) == 1 else ""


def printed_pmid(citation: str) -> str:
    """The PubMed ID printed on this citation line, or empty."""
    match = _PMID_IN_TEXT.search(citation or "")
    return match.group(1) if match else ""
