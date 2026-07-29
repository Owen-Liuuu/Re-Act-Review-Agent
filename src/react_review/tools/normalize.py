"""Extract stage: normalize_field — Tier-2 semantic mapping (the 9th tool).

Maps a raw review column name to a canonical ``field_type`` via a three-step
cascade so most calls are deterministic and cheap:

    1. cache      : (raw_name + research_context) seen before  -> deterministic
    2. knowledge  : known synonym (+ unit / modality disambiguation) -> deterministic
    3. agent      : unknown -> DKB agent classifies against the RETRIEVED candidates,
                    records what it was grounded on, and writes any new concept back
                    as a PROVISIONAL entry (flagged for human review, promoted later)

The cache turns each unique name into a one-time cost; the knowledge base (DKB)
is the living dictionary that grows. ``provisional`` on the result marks answers
that came from an LLM / not-yet-authoritative entry, so the pipeline can route
them to human review.
"""
from __future__ import annotations

import structlog

from react_review.dkb import KeywordRetriever, KnowledgeAgent, KnowledgeBase, Retriever
from react_review.llm.base import LLMBackend
from react_review.tools.base import Tool, ToolStage
from react_review.tools.models import NormalizeInput, NormalizeResult

logger = structlog.get_logger(__name__)


def _cache_key(raw_field_name: str, unit: str, research_context: str) -> str:
    # The unit is part of the key: an ambiguous name like "EFT/ EAT" resolves to
    # eat_thickness (mm) or eat_volume (cm3) depending on it, so caching without
    # the unit would let the first-seen unit poison later lookups.
    return (
        f"{raw_field_name.strip().lower()}||{unit.strip().lower()}"
        f"||{research_context.strip().lower()}"
    )


class NormalizeFieldTool(Tool):
    """Resolve a raw column name to a canonical field_type (cache→knowledge→agent)."""

    name = "normalize_field"
    stage = ToolStage.EXTRACT
    input_model = NormalizeInput
    output_model = NormalizeResult

    def __init__(
        self,
        knowledge: KnowledgeBase,
        backend: LLMBackend | None = None,
        cache: dict[str, str] | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self._kb = knowledge
        # default retriever is keyword (offline, deterministic); pass an
        # EmbeddingRetriever for vector search (DKB-2b) at scale.
        self._agent = (
            KnowledgeAgent(backend, retriever or KeywordRetriever(knowledge))
            if backend else None
        )
        self._cache: dict[str, str] = cache if cache is not None else {}

    @property
    def kb(self) -> KnowledgeBase:
        """The knowledge base (callers use it for e.g. ``scope_of``)."""
        return self._kb

    def _is_provisional(self, field_type: str) -> bool:
        e = self._kb.entries.get(field_type)
        return bool(e and e.status == "provisional")

    async def run(self, payload: NormalizeInput) -> NormalizeResult:
        key = _cache_key(payload.raw_field_name, payload.unit, payload.research_context)

        # 1. cache
        if key in self._cache:
            ft = self._cache[key]
            return NormalizeResult(field_type=ft, source="cache",
                                   provisional=self._is_provisional(ft))

        # 2. knowledge base (synonym + unit/modality disambiguation)
        ft = self._kb.resolve(payload.raw_field_name, payload.unit, payload.modality)
        if ft:
            self._cache[key] = ft
            return NormalizeResult(field_type=ft, source="vocabulary",
                                   provisional=self._is_provisional(ft))

        # 3. DKB agent (grounded classification + provisional write-back)
        if self._agent is None:
            raise ValueError(
                f"cannot resolve field name {payload.raw_field_name!r}: not in the "
                "knowledge base and no LLM backend configured"
            )
        result = await self._agent.classify(
            payload.raw_field_name, payload.unit,
            payload.research_context, payload.modality,
        )
        if result.is_new and result.entry is not None:
            self._kb.add(result.entry)          # provisional; promoted later (DKB-3)
        self._cache[key] = result.field_type
        return NormalizeResult(field_type=result.field_type, source="llm",
                               is_new=result.is_new, provisional=True)
