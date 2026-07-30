"""DKB Resolver — the single owner of field normalization.

The Parser does RAW extraction only and knows no domain knowledge; it hands a raw
field name here. The Resolver runs the cascade and returns a ``ResolvedField`` with
an explicit STATUS:

    cache / DKB-1 deterministic match  → authoritative   (a firm, known concept)
    DKB-2 retrieval + LLM fallback     → candidate        (provisional — NOT firm)
    nothing                            → unresolved

A miss never becomes a confirmed field_type: it is a CANDIDATE the run uses
tentatively, surfaced for human review and collected as a proposal. In audit mode
(``write_back=False``) candidates are NOT merged into the KB — that is the separate
developer "learn" step (promotion + curation). The per-run cache still dedupes so
the same raw name isn't sent to the LLM twice within one run.
"""
from __future__ import annotations

import re

import structlog
from pydantic import BaseModel, Field

from react_review.dkb.agent import KnowledgeAgent
from react_review.dkb.base import KnowledgeBase
from react_review.dkb.retrieval import KeywordRetriever
from react_review.dkb.schema import KnowledgeEntry
from react_review.dkb.verify import verify_candidate

logger = structlog.get_logger(__name__)


class ResolvedField(BaseModel):
    """The outcome of resolving one raw field name.

    The Resolver PROVIDES knowledge (concept + scope); the Parser APPLIES scope.
    """

    raw_field_name: str
    field_type: str | None = None                # None ⇒ unresolved
    status: str = "unresolved"                    # authoritative | candidate | unresolved
    scope: str = "cohort"                         # study | cohort (knowledge, for the parser)
    source: str = "none"                          # cache | deterministic | retrieval_llm
    grounded_on: list[str] = Field(default_factory=list)
    confidence: float = 0.0

    @property
    def resolved(self) -> bool:
        return self.field_type is not None

    @property
    def provisional(self) -> bool:
        return self.status == "candidate"


def _cache_key(raw_name: str, unit: str, context: str) -> str:
    return f"{raw_name.strip().lower()}||{unit.strip().lower()}||{context.strip().lower()}"


class FieldResolver:
    """Resolve raw field names to canonical concepts. Owns all domain knowledge."""

    def __init__(
        self,
        kb: KnowledgeBase,
        agent: KnowledgeAgent | None = None,
        *,
        backend=None,
        cache: dict[str, str] | None = None,
        write_back: bool = False,
    ) -> None:
        self._kb = kb
        if agent is None and backend is not None:      # convenience: build a default agent
            agent = KnowledgeAgent(backend, KeywordRetriever(kb))
        self._agent = agent
        self._cache: dict[str, str] = cache if cache is not None else {}
        self._write_back = write_back
        # Candidate concepts proposed this session (fuel for the developer learn step).
        self.proposals: list[KnowledgeEntry] = []

    @property
    def kb(self) -> KnowledgeBase:
        return self._kb

    def _status_of(self, field_type: str) -> str:
        e = self._kb.entries.get(field_type)
        return "candidate" if (e and e.status == "provisional") else "authoritative"

    def _scope_of(self, field_type: str) -> str:
        e = self._kb.entries.get(field_type)
        return e.scope if e else "cohort"

    async def resolve(
        self, raw_field_name: str, unit: str = "",
        modality: str = "", research_context: str = "", value: object = None,
    ) -> ResolvedField:
        key = _cache_key(raw_field_name, unit, research_context)

        # 1. cache (per-run; dedupes repeat names without re-hitting the LLM)
        if key in self._cache:
            ft = self._cache[key]
            if not ft:                              # cached miss / rejected candidate
                return ResolvedField(raw_field_name=raw_field_name)
            return ResolvedField(raw_field_name=raw_field_name, field_type=ft,
                                 status=self._status_of(ft), scope=self._scope_of(ft),
                                 source="cache")

        # 2. DKB-1 deterministic match (synonym + unit/modality disambiguation)
        ft = self._kb.resolve(raw_field_name, unit, modality)
        if ft:
            self._cache[key] = ft
            return ResolvedField(raw_field_name=raw_field_name, field_type=ft,
                                 status=self._status_of(ft), scope=self._scope_of(ft),
                                 source="deterministic")

        # 3. DKB-2 retrieval + LLM → a CANDIDATE (never a firm field_type on a miss)
        if self._agent is None:
            return ResolvedField(raw_field_name=raw_field_name)     # unresolved (kept upstream)
        c = await self._agent.classify(raw_field_name, unit, research_context, modality)

        # The LLM mapping is a hypothesis — accept it as a candidate only if it
        # survives deterministic verification (grounding + unit-kind + range);
        # otherwise keep the field UNRESOLVED so it goes to human review.
        verdict = verify_candidate(
            c.field_type, kb=self._kb, unit=unit, value=value,
            is_new=(c.entry is not None), confidence=c.confidence,
            grounded_on=c.grounded_on,
        )
        if not verdict.ok:
            logger.info("dkb_candidate_rejected", raw=raw_field_name,
                        field_type=c.field_type, reason=verdict.reason)
            if verdict.checks.get("range", True):   # unit/grounding fail is stable → cache;
                self._cache[key] = ""               # a range fail is value-dependent → don't
            return ResolvedField(raw_field_name=raw_field_name)

        if c.entry is not None:
            self.proposals.append(c.entry)          # collect the proposal for the learn step
            if self._write_back:                    # only the developer/learn path mutates the KB
                self._kb.add(c.entry)
        self._cache[key] = c.field_type
        scope = c.entry.scope if c.entry is not None else self._scope_of(c.field_type)
        return ResolvedField(
            raw_field_name=raw_field_name, field_type=c.field_type, status="candidate",
            scope=scope, source="retrieval_llm",
            grounded_on=c.grounded_on, confidence=c.confidence,
        )
