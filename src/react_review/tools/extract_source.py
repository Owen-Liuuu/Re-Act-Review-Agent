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
- PREFER the data table (e.g. Table 1) over prose. A narrative sentence like
  "Regarding diabetic children, the mean age was X" reports only ONE cohort —
  NEVER reuse that number for another cohort, even if the paper says the groups
  "did not differ significantly".
- Tables list cohorts as COLUMNS in a fixed order — a row reads e.g.
  "Age (years) | <diabetic value> | <control value> | <P value>". Identify which
  column is the {group_desc} and read THAT column's cell in the target row.
- First list every cohort/column the paper reports for this field in ``cohorts_seen``.
- Set ``group_label_in_paper`` to the paper's own name for the cohort you took the
  value from; it MUST be the {group_desc}. Your quote MUST be the {group_desc}'s
  OWN cell or sentence — never another cohort's.
- If the paper reports no value specifically for the {group_desc}, set found=false —
  do NOT substitute or infer another cohort's value.
- Return the value and unit EXACTLY as printed (keep "mean ± SD" / "median (IQR)").
- Do NOT infer or compute; only report what is written.

## PAPER TEXT
{paper_text}

## OUTPUT — one JSON object, nothing else:
{{"cohorts_seen": ["each cohort/column label the paper reports for this field"],
  "group_label_in_paper": "the paper's name for the cohort this value is taken from",
  "found": true or false, "value": "verbatim value or null", "unit": "verbatim unit or empty",
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


# Keyword signals for a deterministic guard: reject an extracted value whose
# reported cohort clearly contradicts the requested group (the model read the
# wrong column). Kept conservative — only reject on an unambiguous contradiction.
_T1DM_KW = ("diabet", "t1dm", "t1d", "dm group")
_CONTROL_KW = ("control", "healthy", "non-diab", "nondiab")


def _group_mismatch(target: str, label_in_paper: str) -> bool:
    """True when the paper-reported cohort contradicts the target group."""
    t = (target or "").strip().lower()
    label = (label_in_paper or "").strip().lower()
    if not label or t in ("all", "-", ""):
        return False
    has_t1dm = any(k in label for k in _T1DM_KW)
    has_control = any(k in label for k in _CONTROL_KW)
    if t == "control":
        return has_t1dm and not has_control
    if t == "t1dm":
        return has_control and not has_t1dm
    return False


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
    group_label_in_paper: str = ""
    wrong_group_rejected: bool = False


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
        label_in_paper = (data.get("group_label_in_paper") or "").strip()
        quote = (data.get("quote") or "").strip()

        # Guard: reject a value whose supporting evidence contradicts the target
        # group — either the claimed cohort label OR the quote itself names the
        # WRONG cohort only (e.g. a "Regarding diabetic children …" sentence used
        # for a Control ask). Better a flagged not-found than a false mismatch.
        if found and (_group_mismatch(payload.group, label_in_paper)
                      or _group_mismatch(payload.group, quote)):
            logger.info("extract_source_group_mismatch", target=payload.group,
                        got=label_in_paper, quote=quote[:80])
            return SourceValueResult(found=False, group_label_in_paper=label_in_paper,
                                     wrong_group_rejected=True)

        return SourceValueResult(
            found=found,
            value=value if found else None,
            unit=(data.get("unit") or "").strip(),
            quote=(data.get("quote") or "").strip(),
            source_field_name=(data.get("source_field_name") or "").strip(),
            location=(data.get("location") or "").strip(),
            group_label_in_paper=label_in_paper,
        )
