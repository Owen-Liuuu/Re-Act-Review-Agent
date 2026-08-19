"""Reference reconciliation: citation → a gated DOI via one or more services.

Queries every configured resolver, scores each candidate against the citation
(the deterministic gate), and accepts the best match only when it clears the
threshold AND carries a DOI. Agreement ACROSS services on the same DOI raises
confidence. A non-match is ``unresolved_source`` — never a wrong-paper guess.

Identifier lookups (DOI or PMID) skip the title-search candidate gate: the
record is the named work. The gate applies only to title-search fallback.
"""
from __future__ import annotations

from collections import defaultdict

import structlog

from react_review.normalize.doi import normalize_doi
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.search.clients import CitationResolver
from react_review.tools.search.gate import DEFAULT_THRESHOLD, candidate_fits_citation, score_match
from react_review.tools.search.models import CandidateWork, ReferenceQuery, ResolvedReference

logger = structlog.get_logger(__name__)

# Last unresolved note per citation key, so the HITL title can say
# "N candidates, none matched" without writing into the collector.
_UNRESOLVED_NOTES: dict[str, str] = {}


def mismatch_note(n_seen: int) -> str:
    """Title-search reject: candidates existed, none matched the citation."""
    return f"retrieved {n_seen} candidates, none matched the citation"


def unresolved_note_for(reference: ReferenceEntry | None) -> str:
    """Human-readable reject reason for a citation, if the reconciler recorded one."""
    if reference is None:
        return ""
    query = ReferenceQuery(
        citation=reference.title or "",
        title=reference.title or "",
        authors=list(reference.authors or []),
        year=reference.year,
        journal=reference.journal or "",
        doi=getattr(reference, "doi", "") or "",
        pmid=str(getattr(reference, "pmid", "") or ""),
    )
    return _UNRESOLVED_NOTES.get(_query_key(query), "")


def _query_key(q: ReferenceQuery) -> str:
    """Normalized identity of a citation — collapses repeat lookups of the same study."""
    base = (q.title or q.citation or "").strip().lower()
    authors = "|".join(sorted(a.strip().lower() for a in q.authors))
    doi = normalize_doi(q.doi)
    pmid = (q.pmid or "").strip()
    return (f"{base}||{q.year or ''}||{authors}||{q.journal.strip().lower()}"
            f"||{doi}||{pmid}")


def _has_identifier(query: ReferenceQuery) -> bool:
    return bool(normalize_doi(query.doi) or (query.pmid or "").strip())


def _identifier_hit(query: ReferenceQuery, cand: CandidateWork) -> bool:
    doi = normalize_doi(query.doi)
    pmid = (query.pmid or "").strip()
    if doi and normalize_doi(cand.doi) == doi:
        return True
    if pmid and (cand.pmid or "").strip() == pmid:
        return True
    return False


class ReferenceReconciler:
    """Resolve a citation to a trusted DOI across citation resolvers."""

    def __init__(
        self,
        resolvers: list[CitationResolver],
        *,
        threshold: float = DEFAULT_THRESHOLD,
        agreement_bonus: float = 0.1,
    ) -> None:
        self._resolvers = resolvers
        self._threshold = threshold
        self._agreement_bonus = agreement_bonus
        self._cache: dict[str, ResolvedReference] = {}     # per-run: one lookup per study

    async def resolve(self, query: ReferenceQuery) -> ResolvedReference:
        key = _query_key(query)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = await self._resolve_uncached(query)
        self._cache[key] = result
        if result.status != "resolved" and result.note:
            _UNRESOLVED_NOTES[key] = result.note
        return result

    async def _resolve_uncached(self, query: ReferenceQuery) -> ResolvedReference:
        if _has_identifier(query):
            identified = await self._resolve_by_identifier(query)
            if identified.status == "resolved":
                return identified
        return await self._resolve_by_title(query)

    async def _gather(self, query: ReferenceQuery, *, identifier: bool
                      ) -> list[tuple[float, str, CandidateWork]]:
        scored: list[tuple[float, str, CandidateWork]] = []
        for r in self._resolvers:
            try:
                lookup = getattr(r, "resolve_identifier", None) if identifier else None
                if identifier and callable(lookup):
                    cands = await lookup(query)
                    matched = list(cands)
                else:
                    cands = await r.resolve(query)
                    matched = (
                        [c for c in cands if _identifier_hit(query, c)]
                        if identifier else list(cands))
            except Exception as exc:                       # noqa: BLE001
                logger.warning("resolver_failed", resolver=getattr(r, "name", "?"),
                               error=str(exc)[:120])
                continue
            for c in matched:
                score = score_match(query, c) if (query.title or query.citation) and c.title else 1.0
                scored.append((score, r.name, c))
        return scored

    async def _resolve_by_identifier(self, query: ReferenceQuery) -> ResolvedReference:
        scored = await self._gather(query, identifier=True)
        if not scored:
            return ResolvedReference(status="unresolved_source", candidates_seen=0)
        with_doi = [(s, name, c) for s, name, c in scored if c.doi]
        if not with_doi:
            return ResolvedReference(status="unresolved_source",
                                     candidates_seen=len(scored))
        return self._accept(with_doi, n_seen=len(scored), gated=False)

    async def _resolve_by_title(self, query: ReferenceQuery) -> ResolvedReference:
        scored = await self._gather(query, identifier=False)
        n_seen = len(scored)
        if not scored:
            return ResolvedReference(status="unresolved_source", candidates_seen=0)

        eligible = [(s, name, c) for s, name, c in scored
                    if candidate_fits_citation(query, c)]
        if not eligible:
            note = mismatch_note(n_seen)
            logger.info("reference_rejected_all_candidates", n=n_seen,
                        title=(query.title or query.citation or "")[:60])
            return ResolvedReference(
                status="unresolved_source", candidates_seen=n_seen, note=note)
        return self._accept(eligible, n_seen=n_seen, gated=True)

    def _accept(
        self,
        scored: list[tuple[float, str, CandidateWork]],
        *,
        n_seen: int,
        gated: bool,
    ) -> ResolvedReference:
        by_doi: dict[str, set[str]] = defaultdict(set)
        for _s, name, c in scored:
            if c.doi:
                by_doi[c.doi.lower()].add(name)

        best_score, best_name, best = max(scored, key=lambda t: t[0])
        agreed = sorted(by_doi.get(best.doi.lower(), {best_name})) if best.doi else [best_name]
        confidence = best_score + (self._agreement_bonus if len(agreed) >= 2 else 0.0)
        confidence = min(1.0, confidence)
        if gated and (confidence < self._threshold or not best.doi):
            logger.info("reference_unresolved", title=(best.title or "")[:60],
                        confidence=round(confidence, 3), has_doi=bool(best.doi))
            return ResolvedReference(status="unresolved_source", confidence=confidence,
                                     matched_title=best.title, candidates_seen=n_seen)
        if not best.doi:
            return ResolvedReference(status="unresolved_source", confidence=confidence,
                                     matched_title=best.title, candidates_seen=n_seen)
        return ResolvedReference(
            status="resolved", doi=best.doi.lower(), pmcid=best.pmcid, source=best_name,
            confidence=confidence, matched_title=best.title, agreed_sources=agreed,
            candidates_seen=n_seen,
        )
