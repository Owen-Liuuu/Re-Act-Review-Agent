"""Typed models for reference reconciliation (citation → a gated DOI/work)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ReferenceQuery(BaseModel):
    """A citation to resolve to a real work (from the review's reference list)."""

    citation: str = ""                  # verbatim reference text (free-text fallback)
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str = ""


class CandidateWork(BaseModel):
    """One candidate match returned by a citation resolver / online service."""

    doi: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str = ""
    pmcid: str = ""
    source: str = ""                    # crossref | openalex | europepmc | ...


class ResolvedReference(BaseModel):
    """The reconciler's gated decision for one citation."""

    status: str = "unresolved_source"   # resolved | unresolved_source
    doi: str = ""
    pmcid: str = ""
    source: str = ""                    # which service the accepted match came from
    confidence: float = 0.0
    matched_title: str = ""
    agreed_sources: list[str] = Field(default_factory=list)   # services agreeing on the DOI

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


class ResolveReferenceInput(BaseModel):
    """Input to the ``resolve_reference`` tool."""

    citation: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    journal: str = ""
    doi: str = ""                       # already known → passthrough (no lookup)


class ResolveReferenceResult(ResolvedReference):
    """Tool output — the resolved reference, plus whether a lookup was skipped."""

    passthrough: bool = False           # DOI already present; no online lookup done
