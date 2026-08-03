"""Dynamic Knowledge Base (DKB) — the evolution of the Tier-2 semantic layer.

DKB-1 (here): schema + deterministic knowledge base (scope, disambiguation, units)
migrated from the static vocabulary. Later: RAG retrieval + agent judgement +
provisional write-back (DKB-2), ontology import + promotion (DKB-3), corpus
learning (DKB-4). See docs/known-limitations.md and the normalization design memo.
"""
from react_review.dkb.agent import AgentClassification, KnowledgeAgent
from react_review.dkb.base import KnowledgeBase
from react_review.dkb.embedding import BackendEmbedder, Embedder, EmbeddingRetriever
from react_review.dkb.learn import LearningSession, load_proposals, save_proposals
from react_review.dkb.ontology import import_ontology, load_runtime_knowledge
from react_review.dkb.promotion import PromotionTracker
from react_review.dkb.resolver import FieldResolver, ResolvedField, resolution_key
from react_review.dkb.retrieval import KeywordRetriever, Retriever
from react_review.dkb.schema import KnowledgeEntry, Provenance
from react_review.dkb.verify import (
    CandidateVerdict,
    evidence_contradicts,
    verify_candidate,
)

__all__ = [
    "KnowledgeBase", "KnowledgeEntry", "Provenance",
    "Retriever", "KeywordRetriever", "KnowledgeAgent", "AgentClassification",
    "Embedder", "BackendEmbedder", "EmbeddingRetriever",
    "PromotionTracker", "import_ontology", "load_runtime_knowledge",
    "FieldResolver", "ResolvedField", "resolution_key",
    "CandidateVerdict", "verify_candidate", "evidence_contradicts",
    "LearningSession", "load_proposals", "save_proposals",
]
