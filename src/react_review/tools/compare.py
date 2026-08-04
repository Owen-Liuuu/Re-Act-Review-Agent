"""Compare stage: the compare_values tool (wraps audit.compare_values).

Deterministic comparison decides first and always. Only when it reports that it
could not read the values as numbers — and both sides are real text — is the
question handed to a model, whose answer is then put through
:mod:`react_review.audit.semantic_control` before it becomes a verdict.

Semantics are OFF by default here: the library, the tests and the eval runners
must stay deterministic unless a caller explicitly asks for more.
"""
from __future__ import annotations

import structlog

from react_review.audit import ToleranceTable, compare_values
from react_review.audit.compare import is_missing_value
from react_review.audit.semantic_cache import SemanticCache, SemanticCacheMiss
from react_review.audit.semantic_control import (
    DEFAULT_MIN_CONFIDENCE,
    apply_semantic_control,
)
from react_review.core.enums import AuditLabel
from react_review.schemas.audit import MatchResult
from react_review.tools.base import Tool, ToolStage
from react_review.tools.models import CompareInput
from react_review.tools.semantic_compare import (
    DEFAULT_SEMANTIC_PROFILE,
    cache_key,
    semantic_prompt_version,
)

logger = structlog.get_logger(__name__)

class CompareValuesTool(Tool):
    """Audit one review↔source value pair using the dual-band tolerance."""

    name = "compare_values"
    stage = ToolStage.COMPARE
    input_model = CompareInput
    output_model = MatchResult

    def __init__(
        self,
        tolerance: ToleranceTable,
        *,
        semantic=None,                       # SemanticCompareTool | None
        semantic_mode: str = "off",          # off | cache-only | on
        semantic_cache: SemanticCache | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        semantic_profile: str = DEFAULT_SEMANTIC_PROFILE,
    ) -> None:
        self._tol = tolerance
        self._semantic = semantic
        self._mode = semantic_mode
        self._cache = semantic_cache
        self._min_confidence = min_confidence
        # The prompt contract is part of the question, so it is part of the key:
        # a judgement recorded under one contract must not be served to a run
        # asking under another.
        self._semantic_version = semantic_prompt_version(semantic_profile)

    async def run(self, payload: CompareInput) -> MatchResult:
        ft = payload.field_type
        result = compare_values(
            field_type=ft,
            review_value=payload.review_value,
            source_value=payload.source_value,
            review_unit=payload.review_unit,
            source_unit=payload.source_unit,
            rel_tolerance=self._tol.rel_tolerance(ft),
            sd_rel_tolerance=self._tol.sd_rel_tolerance(ft),
            p_value_abs_tolerance=self._tol.p_value_abs_tolerance(ft),
            null_value=self._tol.null_value(ft),
            source_components=payload.source_components,
        )
        if not self._should_escalate(payload, result):
            return result
        return await self._semantic_verdict(payload, result)

    def _should_escalate(self, payload: CompareInput, result: MatchResult) -> bool:
        """Only an unreadable-as-numbers pair of real text values qualifies."""
        if self._mode == "off" or result.label is not AuditLabel.NOT_COMPARABLE:
            return False
        if "could not be parsed" not in result.reason:   # e.g. a bound, already decided
            return False
        if is_missing_value(payload.review_value) or is_missing_value(payload.source_value):
            return False
        return bool(self._semantic or self._cache)

    async def _semantic_verdict(
        self, payload: CompareInput, deterministic: MatchResult,
    ) -> MatchResult:
        review_value = str(payload.review_value)
        source_value = str(payload.source_value)
        # With no model configured, the recording says which model made it.
        model_id = (getattr(self._semantic, "model_id", "")
                    or getattr(self._cache, "model_id", "") or "cache")
        key = cache_key({
            "model_id": model_id, "prompt_version": self._semantic_version,
            "field_type": payload.field_type, "column_header": payload.column_header,
            "research_context": payload.research_context,
            "review_value": review_value, "review_unit": payload.review_unit,
            "source_value": source_value, "source_unit": payload.source_unit,
            "source_quote": payload.source_quote, "seed": 42,
        })

        # SemanticCache implements ``__len__``; an empty cache is therefore
        # falsey.  Test identity instead so the first lookup is recorded as a
        # miss and run telemetry agrees with the number of stored judgements.
        verdict = self._cache.get(key) if self._cache is not None else None
        if verdict is None:
            if self._mode == "cache-only":
                # Fail loudly: a "reproduce the recording" run that quietly calls
                # the model is no longer reproducing anything.
                raise SemanticCacheMiss(
                    f"no recorded judgement for {payload.field_type}: "
                    f"{review_value!r} vs {source_value!r}")
            if self._semantic is None:
                return deterministic
            verdict = await self._semantic.judge(
                field_type=payload.field_type, column_header=payload.column_header,
                research_context=payload.research_context,
                review_value=review_value, source_value=source_value,
                source_quote=payload.source_quote)
            if self._cache is not None:
                self._cache.put(key, verdict)

        outcome = apply_semantic_control(
            verdict, review_value=review_value, source_value=source_value,
            source_quote=payload.source_quote,
            rel_tolerance=self._tol.rel_tolerance(payload.field_type),
            min_confidence=self._min_confidence)
        logger.info("semantic_compare", field_type=payload.field_type,
                    relation=verdict.relation, label=outcome.label.value,
                    failed_control=outcome.failed_control)
        return deterministic.model_copy(update={
            "label": outcome.label, "reason": outcome.reason,
            "match_mode": "semantic", "review_required": outcome.review_required,
            "semantic": verdict, "semantic_relation": verdict.relation,
            "semantic_controls": outcome.checks,
        })
