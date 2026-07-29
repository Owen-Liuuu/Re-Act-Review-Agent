"""Dynamic Knowledge Base (DKB) — the evolution of the Tier-2 semantic layer.

DKB-1 (here): schema + deterministic knowledge base (scope, disambiguation, units)
migrated from the static vocabulary. Later: RAG retrieval + agent judgement +
provisional write-back (DKB-2), ontology import + promotion (DKB-3), corpus
learning (DKB-4). See docs/known-limitations.md and the normalization design memo.
"""
from react_review.dkb.base import KnowledgeBase
from react_review.dkb.schema import KnowledgeEntry, Provenance

__all__ = ["KnowledgeBase", "KnowledgeEntry", "Provenance"]
