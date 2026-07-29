"""DKB vector retrieval — embedding-based candidate ranking (DKB-2b).

An ``EmbeddingRetriever`` implements the same ``Retriever`` interface as the
keyword one, so the agent doesn't change: it embeds each KB concept once (lazy,
cached) and ranks by cosine similarity to the query embedding. This is what pays
off as the KB grows past keyword matching. Kept pure-Python (no numpy dep); the
concept vectors are tiny and computed once.

Reproducibility note: embedding drift only reorders CANDIDATES — the final
field_type mapping is still cached deterministically in NormalizeFieldTool, so an
API embedder is acceptable here.
"""
from __future__ import annotations

import math
from typing import Protocol

from react_review.dkb.base import KnowledgeBase
from react_review.dkb.schema import KnowledgeEntry


class Embedder(Protocol):
    """Turn texts into vectors (e.g. GLM ``embedding-3``)."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class BackendEmbedder:
    """Adapt an OpenAI-compatible backend's ``embed`` to the Embedder interface."""

    def __init__(self, backend, model: str = "embedding-3") -> None:
        self._backend = backend
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._backend.embed(texts, model=self._model)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class EmbeddingRetriever:
    """Rank KB entries by embedding cosine similarity to the query."""

    def __init__(self, kb: KnowledgeBase, embedder: Embedder) -> None:
        self._kb = kb
        self._embedder = embedder
        self._vectors: dict[str, list[float]] | None = None

    @staticmethod
    def _entry_text(e: KnowledgeEntry) -> str:
        return f"{e.field_type}: {e.concept}. synonyms: " + ", ".join(e.synonyms)

    async def _ensure_vectors(self) -> None:
        if self._vectors is None:
            fts = list(self._kb.entries)
            texts = [self._entry_text(self._kb.entries[ft]) for ft in fts]
            self._vectors = dict(zip(fts, await self._embedder.embed(texts)))

    async def retrieve(self, query: str, k: int = 8) -> list[KnowledgeEntry]:
        await self._ensure_vectors()
        qv = (await self._embedder.embed([query]))[0]
        ranked = sorted(
            self._kb.entries.values(),
            key=lambda e: _cosine(qv, self._vectors.get(e.field_type, [])),
            reverse=True,
        )
        return ranked[:k]
