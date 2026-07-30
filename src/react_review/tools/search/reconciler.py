"""Reference reconciliation: citation → a gated DOI via one or more services.

Queries every configured resolver, scores each candidate against the citation
(the deterministic gate), and accepts the best match only when it clears the
threshold AND carries a DOI. Agreement ACROSS services on the same DOI raises
confidence. A non-match is ``unresolved_source`` — never a wrong-paper guess.
"""
from __future__ import annotations

from collections import defaultdict

import structlog

from react_review.tools.search.clients import CitationResolver
from react_review.tools.search.gate import DEFAULT_THRESHOLD, score_match
from react_review.tools.search.models import CandidateWork, ReferenceQuery, ResolvedReference

logger = structlog.get_logger(__name__)


def _query_key(q: ReferenceQuery) -> str:
    """Normalized identity of a citation — collapses repeat lookups of the same study."""
    base = (q.title or q.citation or "").strip().lower()
    authors = "|".join(sorted(a.strip().lower() for a in q.authors))
    return f"{base}||{q.year or ''}||{authors}||{q.journal.strip().lower()}"


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
        return result

    async def _resolve_uncached(self, query: ReferenceQuery) -> ResolvedReference:
        # 1. gather scored candidates from every service (one failing service
        #    must not sink the rest).
        scored: list[tuple[float, str, CandidateWork]] = []
        for r in self._resolvers:
            try:
                cands = await r.resolve(query)
            except Exception as exc:                       # noqa: BLE001
                logger.warning("resolver_failed", resolver=getattr(r, "name", "?"),
                               error=str(exc)[:120])
                continue
            scored.extend((score_match(query, c), r.name, c) for c in cands)
        if not scored:
            return ResolvedReference(status="unresolved_source")

        # 2. cross-source agreement on a normalized DOI raises trust.
        by_doi: dict[str, set[str]] = defaultdict(set)
        for _s, name, c in scored:
            if c.doi:
                by_doi[c.doi.lower()].add(name)

        # 3. best-scoring candidate, plus an agreement bonus when ≥2 services
        #    independently point at its DOI.
        best_score, best_name, best = max(scored, key=lambda t: t[0])
        agreed = sorted(by_doi.get(best.doi.lower(), {best_name})) if best.doi else [best_name]
        confidence = best_score + (self._agreement_bonus if len(agreed) >= 2 else 0.0)
        confidence = min(1.0, confidence)

        if confidence < self._threshold or not best.doi:
            logger.info("reference_unresolved", title=(best.title or "")[:60],
                        confidence=round(confidence, 3), has_doi=bool(best.doi))
            return ResolvedReference(status="unresolved_source", confidence=confidence,
                                     matched_title=best.title)
        return ResolvedReference(
            status="resolved", doi=best.doi.lower(), pmcid=best.pmcid, source=best_name,
            confidence=confidence, matched_title=best.title, agreed_sources=agreed,
        )
