"""Tests for the DKB vector retriever (DKB-2b) with a fake, offline embedder."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.dkb import EmbeddingRetriever, KnowledgeBase
from react_review.dkb.embedding import _cosine

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"


class _FakeEmbedder:
    """Deterministic bag-of-chars embedding — no network, but real cosine ranking."""

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vecs = []
        for t in texts:
            v = [0.0] * 26
            for ch in t.lower():
                if "a" <= ch <= "z":
                    v[ord(ch) - 97] += 1.0
            vecs.append(v)
        return vecs


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine([], [1, 2]) == 0.0


@pytest.mark.asyncio
async def test_embedding_retriever_ranks_by_similarity():
    kb = KnowledgeBase.from_json(SEED)
    retr = EmbeddingRetriever(kb, _FakeEmbedder())
    # a query sharing the "body mass index" characters ranks bmi highly
    hits = await retr.retrieve("body mass index", k=3)
    assert "bmi" in {e.field_type for e in hits}


@pytest.mark.asyncio
async def test_embedding_retriever_caches_entry_vectors():
    kb = KnowledgeBase.from_json(SEED)
    emb = _FakeEmbedder()
    retr = EmbeddingRetriever(kb, emb)
    await retr.retrieve("age")
    await retr.retrieve("bmi")
    # entry vectors embedded once (call 1), then one query embed per retrieve
    assert emb.calls == 3
