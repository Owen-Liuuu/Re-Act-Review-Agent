"""Deterministic DOI normalization for matching."""
from __future__ import annotations

_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:")


def normalize_doi(doi: str | None) -> str:
    """Lower-case and strip the URL / ``doi:`` prefix. '' for a missing DOI."""
    d = (doi or "").strip().lower()
    for p in _PREFIXES:
        if d.startswith(p):
            d = d[len(p):]
            break
    return d.strip().rstrip("/")
