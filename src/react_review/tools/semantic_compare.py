"""Ask a model whether two text values denote the same thing.

The deterministic comparator can only decide numbers, so every categorical cell
became ``not_comparable`` — "ICU" against "intensive care unit", a hazard ratio
spelled differently, "Good" against "Good Quality". Enumerating those pairs as
rules does not scale and, worse, a model will obey a bad rule silently rather
than report that the rule is wrong.

So the model is given the job, not the rules: it says what it thinks and why,
and :mod:`react_review.audit.semantic_control` decides whether to believe it.
This module never makes the verdict — it only produces a claim to be checked.
"""
from __future__ import annotations

import hashlib
import json

import structlog
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.schemas.semantic import SemanticVerdict

logger = structlog.get_logger(__name__)

PROMPT_VERSION = "semantic-v1"

_PROMPT = """You are auditing a systematic review against its source papers.

A review reports a value for one field; the source paper states another. Decide
whether the two refer to the SAME thing, or whether the review has changed the
claim.

## FIELD
{field}
Research context: {context}

## THE TWO VALUES
Review reports : {review_value}
Source states  : {source_value}
Quoted from the source paper: "{quote}"

## WHAT TO DECIDE
- same            — the same thing, worded differently ("ICU" / "intensive care unit")
- review_broader  — the review states something LESS specific than the source
                    ("France" where the source says "France, surgical ICU")
- source_broader  — the source is less specific than the review
- different       — not the same thing
- unknown         — you cannot tell from what you were given

Judge the MEANING in this field's context. Do not decide by string similarity,
and do not treat two different numbers as the same thing.

Return one JSON object, nothing else:
{{"relation": "same|review_broader|source_broader|different|unknown",
  "equivalent": true or false,
  "confidence": 0.0,
  "rationale": "one sentence, why",
  "review_normalized": "the review value in plain words",
  "source_normalized": "the source value in plain words",
  "evidence_span": "the EXACT substring of the quote above that you relied on — copy it verbatim, or leave empty if the quote does not support it"}}
"""


def cache_key(payload: dict) -> str:
    """Identity of a semantic question.

    Every input that can change the answer belongs here — the field and the
    research context steer the judgement as much as the two values do, so a key
    without them would serve one pair's answer to a different question.
    """
    material = json.dumps({k: payload.get(k, "") for k in (
        "model_id", "prompt_version", "field_type", "column_header",
        "research_context", "review_value", "review_unit",
        "source_value", "source_unit", "source_quote", "seed",
    )}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class SemanticCompareTool:
    """Produce a :class:`SemanticVerdict` for one text pair."""

    def __init__(self, backend: LLMBackend, *, seed: int = 42) -> None:
        self._backend = backend
        self._seed = seed

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    async def judge(
        self, *, field_type: str, column_header: str, research_context: str,
        review_value: str, source_value: str, source_quote: str = "",
    ) -> SemanticVerdict:
        field = column_header or field_type or "(unnamed field)"
        prompt = _PROMPT.format(
            field=field, context=research_context or "a systematic review",
            review_value=review_value, source_value=source_value,
            quote=source_quote or "(no quote was captured)")
        provenance = {
            "model_id": self._backend.model_id, "prompt_version": PROMPT_VERSION,
            "seed": self._seed,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        }
        try:
            raw = await self._backend.complete(prompt, seed=self._seed)
            data = parse_llm_response(raw, self._backend.model_id)
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("semantic_compare_failed", error=str(exc)[:160])
            return SemanticVerdict(relation="unknown",
                                   rationale=f"the judgement call failed: {exc}"[:200],
                                   provenance={**provenance, "error": str(exc)[:200]})
        return SemanticVerdict(
            relation=str(data.get("relation") or "unknown").strip().lower(),
            equivalent=bool(data.get("equivalent")),
            confidence=float(data.get("confidence") or 0.0),
            rationale=str(data.get("rationale") or "").strip(),
            review_normalized=str(data.get("review_normalized") or "").strip(),
            source_normalized=str(data.get("source_normalized") or "").strip(),
            evidence_span=str(data.get("evidence_span") or "").strip(),
            provenance=provenance,
        )
