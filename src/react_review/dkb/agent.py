"""DKB agent — grounded field_type classification on a KB miss (DKB-2).

When deterministic resolution fails, the agent classifies the raw field using
only the RETRIEVED candidates (not the whole KB), records which entries it was
grounded on (auditability), and returns any new concept as a ``provisional``
entry with LLM provenance — never silently authoritative. Promotion to
authoritative (human confirm / repeated agreement) is DKB-3.
"""
from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from react_review.dkb.retrieval import Retriever
from react_review.dkb.schema import KnowledgeEntry, Provenance
from react_review.llm.base import LLMBackend, parse_llm_response

logger = structlog.get_logger(__name__)

_PROMPT = """You map a systematic-review table column name to a canonical field_type.

Research context: {context}
Column name: "{raw_field_name}"
Reported unit: "{unit}"

Candidate field_types (choose the best fit; otherwise propose a NEW snake_case field_type):
{candidates}

Return a single JSON object, no commentary:
{{"field_type": "snake_case_name", "concept": "short description",
  "value_type": "numeric|text|categorical", "default_unit": "unit or empty",
  "is_new": true or false,
  "grounded_on": ["field_types from the candidates you used to decide"],
  "confidence": 0.0}}
"""


class AgentClassification(BaseModel):
    """The agent's grounded decision for one raw field."""

    field_type: str
    is_new: bool = False
    confidence: float = 0.0
    grounded_on: list[str] = Field(default_factory=list)
    entry: KnowledgeEntry | None = None      # the provisional entry to add, if new


class KnowledgeAgent:
    """Classify a raw field into a field_type, grounded in retrieved KB entries."""

    def __init__(self, backend: LLMBackend, retriever: Retriever) -> None:
        self._backend = backend
        self._retriever = retriever

    async def classify(
        self, raw_name: str, unit: str = "",
        research_context: str = "", modality: str = "",
    ) -> AgentClassification:
        cands = await self._retriever.retrieve(f"{raw_name} {unit} {research_context}".strip())
        cand_list = "\n".join(
            f"- {e.field_type}: {e.concept}"
            + (f" (unit {e.default_unit})" if e.default_unit else "")
            for e in cands
        ) or "- (none yet)"
        prompt = _PROMPT.format(
            context=research_context or "a systematic review",
            raw_field_name=raw_name, unit=unit, candidates=cand_list,
        )
        data = parse_llm_response(await self._backend.complete(prompt), self._backend.model_id)

        ft = (data.get("field_type") or "").strip().lower().replace(" ", "_")
        if not ft:
            raise ValueError(f"agent returned no field_type for {raw_name!r}")
        known = {e.field_type for e in cands}
        is_new = bool(data.get("is_new", ft not in known)) and ft not in known
        grounded = [str(c) for c in (data.get("grounded_on") or []) if isinstance(c, str)]
        confidence = float(data.get("confidence") or 0.0)

        entry = None
        if is_new:
            entry = KnowledgeEntry(
                field_type=ft,
                concept=(data.get("concept") or "").strip(),
                value_type=(data.get("value_type") or "numeric").strip().lower(),
                default_unit=(data.get("default_unit") or "").strip(),
                synonyms=[raw_name],
                provenance=Provenance(source="llm", confidence=confidence,
                                      citation=", ".join(grounded)),
                status="provisional",       # LLM-proposed is never authoritative
            )
        logger.info("dkb_agent_classify", raw=raw_name, field_type=ft,
                    is_new=is_new, grounded_on=grounded)
        return AgentClassification(field_type=ft, is_new=is_new, confidence=confidence,
                                   grounded_on=grounded, entry=entry)
