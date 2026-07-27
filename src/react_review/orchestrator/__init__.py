"""Deterministic orchestration: match review↔source, compare, aggregate.

The orchestrator sequences the stages and owns control flow; the LLM-backed work
lives in the tools it calls. In P1 the review and source evidence tables are fed
in directly (from the benchmark CSVs standing in for the parser/collector); P2's
agents produce them live.
"""
from react_review.orchestrator.matcher import build_pairs, match_key
from react_review.orchestrator.pipeline import AuditOrchestrator
from react_review.orchestrator.reflection import (
    ReflectionDecider,
    ReflectionOutcome,
    ReflectionSignals,
)
from react_review.orchestrator.judge import Judge
from react_review.orchestrator.audit_pipeline import AuditPipeline

__all__ = [
    "AuditOrchestrator",
    "build_pairs",
    "match_key",
    "ReflectionDecider",
    "ReflectionOutcome",
    "ReflectionSignals",
    "Judge",
    "AuditPipeline",
]
