"""DKB retrieval — pick the KB entries most relevant to a raw field (DKB-2).

Kept behind a ``Retriever`` interface so the implementation can grow with the KB:
while the KB is small, a keyword-overlap retriever (or "return all") is plenty;
when it reaches hundreds of concepts, swap in an embedding/vector retriever
(DKB-2b) without touching the agent that consumes it.
"""
from __future__ import annotations

import re
from typing import Protocol

from react_review.dkb.base import KnowledgeBase
from react_review.dkb.schema import KnowledgeEntry


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


class Retriever(Protocol):
    """Return the KB entries most relevant to a query, best-first.

    Async because an embedding retriever calls out to a model (DKB-2b); the
    keyword retriever is sync work wrapped in an ``async def``.
    """

    async def retrieve(self, query: str, k: int = 8) -> list[KnowledgeEntry]: ...


class KeywordRetriever:
    """Rank entries by token overlap of the query with their names/synonyms.

    Deterministic, no embeddings. For a small KB every entry is a candidate; the
    ranking just puts the likely ones first so the agent's prompt stays focused.
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    async def retrieve(self, query: str, k: int = 8) -> list[KnowledgeEntry]:
        q = _tokens(query)
        scored: list[tuple[int, str, KnowledgeEntry]] = []
        for ft, e in self._kb.entries.items():
            overlap = len(q & _tokens(" ".join(e.all_names())))
            scored.append((overlap, ft, e))
        scored.sort(key=lambda t: (-t[0], t[1]))
        hits = [e for score, _, e in scored if score > 0][:k]
        # nothing overlapped → still give the agent the whole (small) KB to pick from
        return hits or [e for _, _, e in scored][:k]
