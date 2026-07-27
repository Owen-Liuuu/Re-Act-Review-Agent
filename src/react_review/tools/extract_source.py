"""Extract stage: extract_source_value — directed, group-aware source lookup.

Stage 2 of the normalization pipeline: given a paper document and ONE target
concept (field_type + group), find that specific value with a verbatim quote —
NOT a blind whole-table extraction. The canonical concept came from the review
side; here the LLM maps it back to whatever the source paper calls it.
"""
from __future__ import annotations

import structlog
from pydantic import BaseModel

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.tools.base import Tool, ToolStage

logger = structlog.get_logger(__name__)

_MAX_TEXT = 20000

_PROMPT = """You are extracting ONE specific value from a source paper for an audit.

## RESEARCH CONTEXT
{context}

## TARGET
Find the value of **{concept}** (field_type: {field_type}) for the **{group_desc}**.
Expected unit (hint, may differ in the paper): "{unit_hint}"

## RULES
- Return the value EXACTLY as printed (keep "mean ± SD" / "median (IQR)" verbatim).
- Return the unit EXACTLY as the paper prints it (this may reveal a unit error).
- Give a verbatim quote (the sentence or table cell the value comes from).
- If the paper does not report this value for this group, set found=false.
- Do NOT infer or compute; only report what is written.

## PAPER TEXT
{paper_text}

## OUTPUT — one JSON object, nothing else:
{{"found": true or false, "value": "verbatim value or null", "unit": "verbatim unit or empty",
  "quote": "verbatim supporting sentence/cell", "source_field_name": "the paper's own label for this field",
  "location": "where (e.g. Table 2, Results)"}}
"""


def _group_desc(group: str) -> str:
    g = (group or "").strip().lower()
    if g == "t1dm":
        return "type 1 diabetes (T1DM / diabetic) group"
    if g == "control":
        return "healthy control group"
    if g in ("all", "-", ""):
        return "whole study cohort (no diabetes/control split)"
    return f"{group} group"


class ExtractSourceValueInput(BaseModel):
    document: PaperDocument
    field_type: str
    group: str = "-"
    concept: str = ""
    unit_hint: str = ""
    research_context: str = ""


class SourceValueResult(BaseModel):
    found: bool = False
    value: str | None = None
    unit: str = ""
    quote: str = ""
    source_field_name: str = ""
    location: str = ""


class ExtractSourceValueTool(Tool):
    """Find one target value (field_type + group) in a source paper, with a quote."""

    name = "extract_source_value"
    stage = ToolStage.EXTRACT
    input_model = ExtractSourceValueInput
    output_model = SourceValueResult

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def run(self, payload: ExtractSourceValueInput) -> SourceValueResult:
        prompt = _PROMPT.format(
            context=payload.research_context or "a systematic review",
            concept=payload.concept or payload.field_type,
            field_type=payload.field_type,
            group_desc=_group_desc(payload.group),
            unit_hint=payload.unit_hint,
            paper_text=(payload.document.full_text or "")[:_MAX_TEXT],
        )
        try:
            raw = await self._backend.complete(prompt)
            data = parse_llm_response(raw, self._backend.model_id)
        except Exception as exc:
            logger.warning("extract_source_value_failed", error=str(exc)[:160])
            return SourceValueResult(found=False)

        value = data.get("value")
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a"):
            value = None
        found = bool(data.get("found")) and value is not None
        return SourceValueResult(
            found=found,
            value=value if found else None,
            unit=(data.get("unit") or "").strip(),
            quote=(data.get("quote") or "").strip(),
            source_field_name=(data.get("source_field_name") or "").strip(),
            location=(data.get("location") or "").strip(),
        )
