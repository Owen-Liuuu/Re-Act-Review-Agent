"""Extract stage: normalize_field — Tier-2 semantic mapping (the 9th tool).

Maps a raw review column name to a canonical ``field_type`` via a three-step
cascade so most calls are deterministic and cheap:

    1. cache      : (raw_name + research_context) seen before  -> deterministic
    2. vocabulary : known synonym (+ unit disambiguation)      -> deterministic
    3. LLM        : unknown -> map into an existing field_type or add a new one,
                    then write it back to the vocabulary and cache

The LLM step is what lets this generalise across medical domains where a static
dictionary cannot; the cache turns each unique name into a one-time cost and the
vocabulary is the living dictionary that grows.
"""
from __future__ import annotations

import structlog

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.normalize.vocabulary import FieldTypeEntry, Vocabulary
from react_review.tools.base import Tool, ToolStage
from react_review.tools.models import NormalizeInput, NormalizeResult

logger = structlog.get_logger(__name__)


_PROMPT = """You map a systematic-review table column name to a canonical field_type.

Research context: {context}
Column name: "{raw_field_name}"
Reported unit: "{unit}"

Known field_types (choose one if it fits; otherwise propose a NEW snake_case field_type):
{vocab_list}

Return a single JSON object, no commentary:
{{"field_type": "snake_case_name", "concept": "short description",
  "value_type": "numeric|text|categorical", "default_unit": "unit or empty",
  "is_new": true or false}}
"""


def _cache_key(raw_field_name: str, unit: str, research_context: str) -> str:
    # The unit is part of the key: an ambiguous name like "EFT/ EAT" resolves to
    # eat_thickness (mm) or eat_volume (cm3) depending on it, so caching without
    # the unit would let the first-seen unit poison later lookups.
    return (
        f"{raw_field_name.strip().lower()}||{unit.strip().lower()}"
        f"||{research_context.strip().lower()}"
    )


class NormalizeFieldTool(Tool):
    """Resolve a raw column name to a canonical field_type (cache→vocab→LLM)."""

    name = "normalize_field"
    stage = ToolStage.EXTRACT
    input_model = NormalizeInput
    output_model = NormalizeResult

    def __init__(
        self,
        vocabulary: Vocabulary,
        backend: LLMBackend | None = None,
        cache: dict[str, str] | None = None,
    ) -> None:
        self._vocab = vocabulary
        self._backend = backend
        self._cache: dict[str, str] = cache if cache is not None else {}

    async def run(self, payload: NormalizeInput) -> NormalizeResult:
        key = _cache_key(payload.raw_field_name, payload.unit, payload.research_context)

        # 1. cache
        if key in self._cache:
            return NormalizeResult(field_type=self._cache[key], source="cache")

        # 2. vocabulary (synonym + unit disambiguation)
        ft = self._vocab.resolve(payload.raw_field_name, payload.unit)
        if ft:
            self._cache[key] = ft
            return NormalizeResult(field_type=ft, source="vocabulary")

        # 3. LLM fallback
        if self._backend is None:
            raise ValueError(
                f"cannot resolve field name {payload.raw_field_name!r}: not in the "
                "vocabulary and no LLM backend configured"
            )
        ft, entry, is_new = await self._llm_resolve(payload)
        if is_new:
            self._vocab.add(entry)
        self._cache[key] = ft
        return NormalizeResult(field_type=ft, source="llm", is_new=is_new)

    async def _llm_resolve(
        self, payload: NormalizeInput
    ) -> tuple[str, FieldTypeEntry, bool]:
        vocab_list = "\n".join(
            f"- {e.field_type}: {e.concept}"
            + (f" (unit {e.default_unit})" if e.default_unit else "")
            for e in self._vocab.entries.values()
        ) or "- (none yet)"
        prompt = _PROMPT.format(
            context=payload.research_context or "a systematic review",
            raw_field_name=payload.raw_field_name,
            unit=payload.unit,
            vocab_list=vocab_list,
        )
        raw = await self._backend.complete(prompt)
        data = parse_llm_response(raw, self._backend.model_id)

        ft = (data.get("field_type") or "").strip().lower().replace(" ", "_")
        if not ft:
            raise ValueError(
                f"LLM returned no field_type for {payload.raw_field_name!r}"
            )
        is_new = ft not in self._vocab.entries
        entry = FieldTypeEntry(
            field_type=ft,
            concept=(data.get("concept") or "").strip(),
            value_type=(data.get("value_type") or "numeric").strip().lower(),
            default_unit=(data.get("default_unit") or "").strip(),
            synonyms=[payload.raw_field_name] if is_new else [],
        )
        logger.info(
            "normalize_field_llm",
            raw=payload.raw_field_name,
            field_type=ft,
            is_new=is_new,
        )
        return ft, entry, is_new
