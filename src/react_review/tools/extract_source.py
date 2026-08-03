"""Extract stage: extract_source_value — directed, group-aware source lookup.

Stage 2 of the normalization pipeline: given a paper document and ONE target
concept (field_type + group), find that specific value with a verbatim quote —
NOT a blind whole-table extraction. The canonical concept came from the review
side; here the LLM maps it back to whatever the source paper calls it.
"""
from __future__ import annotations

import re

import structlog
from pydantic import BaseModel, Field

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.tools.base import Tool, ToolStage

logger = structlog.get_logger(__name__)

_MAX_TEXT = 20000

_PROMPT = """You are extracting ONE specific value from a source paper for an audit.

## RESEARCH CONTEXT
{context}

## TARGET
Find the value of **{concept}** for the **{group_desc}**.
(The review's column was labelled "{raw_label}"; internal field_type: {field_type}.)
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
- When found=false you MUST fill ``not_found_reason``: say what you looked at and
  why the value is not there (absent from the paper / reported for other cohorts
  only / the text is truncated / the table did not come through, …). "Not found"
  without a reason cannot be acted on.
- Return the value and unit EXACTLY as printed (keep "mean ± SD" / "median (IQR)").
- Do NOT infer or compute; only report what is written.

## PAPER TEXT
{paper_text}

## OUTPUT — one JSON object, nothing else:
{{"cohorts_seen": ["each cohort/column label the paper reports for this field"],
  "group_label_in_paper": "the paper's name for the cohort this value is taken from",
  "found": true or false, "value": "verbatim value or null", "unit": "verbatim unit or empty",
  "quote": "verbatim supporting sentence/cell", "source_field_name": "the paper's own label for this field",
  "location": "where (e.g. Table 2, Results)",
  "not_found_reason": "when found=false, why — otherwise empty"}}
"""


def cohort_description(group: str, *, display: str = "",
                       variants: list[str] | None = None) -> str:
    """How to describe the wanted cohort to the extractor — in the REVIEW's words.

    No disease vocabulary lives in this code any more: whatever the review calls
    its arm is what the model is asked for. An unidentified cohort is stated as
    such rather than described as the whole study, which would have the model
    fetch a pooled number for a claim that is about one arm.
    """
    g = (group or "").strip().lower()
    if not g:
        return ("cohort the review did not identify — report every cohort the "
                "paper distinguishes and say which one you took the value from")
    if g in ("all", "-"):
        return "whole study cohort (the review reports no separate arms here)"
    name = display or group
    also = [v for v in (variants or []) if v and v != name]
    suffix = f" (the paper may call it: {', '.join(also)})" if also else ""
    return f'"{name}" cohort{suffix}'


def cohort_conflicts(
    target: str, label_in_paper: str, *,
    cohorts: dict[str, list[str]] | None = None,
) -> tuple[str, str]:
    """Does the paper's own cohort label contradict the one we asked for?

    Returns ``(verdict, reason)`` where verdict is ``ok`` | ``wrong_cohort`` |
    ``ambiguous``. **Ambiguous is not ok**: a tie, or a label matching no known
    cohort, means the guard could not confirm the value belongs to the requested
    arm, and letting that through as clean is exactly the silent pass this guard
    exists to prevent. It is kept as evidence but forced to human review.
    """
    label = (label_in_paper or "").strip().lower()
    t = (target or "").strip().lower()
    if not label or t in ("all", "-", ""):
        return "ok", ""
    if not cohorts:
        # No cohort information was supplied at all (e.g. auditing two CSVs whose
        # groups are already canonical). The guard is not configured here, which
        # is different from being unable to confirm a cohort it does know about.
        return "ok", ""
    if t not in cohorts:
        return "ambiguous", (f"cohort {target!r} is not one this review was found "
                             "to report, so the paper's label could not be checked "
                             "against it")

    scores = {key: _overlap(label, variants) for key, variants in cohorts.items()}
    best_key, best = max(scores.items(), key=lambda kv: kv[1])
    mine = scores.get(t, 0.0)
    if best < 0.5 or best_key == t:
        # Nothing matched well, or the best match IS the target.
        if best < 0.5:
            return "ambiguous", (f"the paper calls this cohort {label_in_paper!r}, "
                                 "which matches no cohort this review reports")
        return "ok", ""
    if best - mine >= 0.25:
        return "wrong_cohort", (f"the paper attributes this value to "
                                f"{label_in_paper!r}, which matches cohort "
                                f"{best_key!r}, not the requested {target!r}")
    return "ambiguous", (f"the paper's label {label_in_paper!r} fits {best_key!r} "
                         f"and {target!r} about equally well")


def _overlap(label: str, variants: list[str]) -> float:
    """Best word-overlap of any variant against the paper's label (0..1)."""
    label_words = set(re.findall(r"[a-z0-9]+", label))
    best = 0.0
    for variant in variants:
        words = set(re.findall(r"[a-z0-9]+", (variant or "").lower()))
        if words:
            best = max(best, len(words & label_words) / len(words))
    return best


class ExtractSourceValueInput(BaseModel):
    document: PaperDocument
    field_type: str
    group: str = "-"
    concept: str = ""
    raw_field_name: str = ""      # the review's own column label — the extraction target
    unit_hint: str = ""
    research_context: str = ""
    # The review's own name for the wanted cohort, and every cohort it reports —
    # so the guard can check the paper's label without any disease vocabulary.
    cohort_display: str = ""
    cohorts: dict[str, list[str]] = {}


class SourceValueResult(BaseModel):
    found: bool = False
    value: str | None = None
    unit: str = ""
    quote: str = ""
    source_field_name: str = ""
    location: str = ""
    group_label_in_paper: str = ""
    wrong_group_rejected: bool = False
    # ok | wrong_cohort | ambiguous — "ambiguous" keeps the value but must reach
    # a human, because the guard could not confirm which arm it belongs to.
    cohort_check: str = "ok"
    cohort_reason: str = ""
    # Why nothing was found, in the model's own words, and what it DID see.
    # Both were previously discarded, leaving "found=false" with no explanation.
    not_found_reason: str = ""
    cohorts_seen: list[str] = Field(default_factory=list)
    error: str = ""


class ExtractSourceValueTool(Tool):
    """Find one target value (field_type + group) in a source paper, with a quote."""

    name = "extract_source_value"
    stage = ToolStage.EXTRACT
    input_model = ExtractSourceValueInput
    output_model = SourceValueResult

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def run(self, payload: ExtractSourceValueInput) -> SourceValueResult:
        # The target description prefers the canonical concept, but falls back to
        # the review's RAW column label — so an UNRESOLVED field (no field_type)
        # is still extractable: the raw name itself says what to look for.
        target = payload.concept or payload.raw_field_name or payload.field_type
        prompt = _PROMPT.format(
            context=payload.research_context or "a systematic review",
            concept=target,
            raw_label=payload.raw_field_name or target,
            field_type=payload.field_type or "(unresolved)",
            group_desc=cohort_description(
                payload.group, display=payload.cohort_display,
                variants=payload.cohorts.get(payload.group, [])),
            unit_hint=payload.unit_hint,
            paper_text=(payload.document.full_text or "")[:_MAX_TEXT],
        )
        try:
            raw = await self._backend.complete(prompt)
            data = parse_llm_response(raw, self._backend.model_id)
        except Exception as exc:
            # The text is CARRIED, not just logged: without it the Collector can
            # only record "not found", and a transport error becomes
            # indistinguishable from a paper that genuinely omits the value.
            logger.warning("extract_source_value_failed", error=str(exc)[:160])
            return SourceValueResult(
                found=False, error=str(exc)[:300],
                not_found_reason=f"the extraction call failed: {type(exc).__name__}")

        value = data.get("value")
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a"):
            value = None
        found = bool(data.get("found")) and value is not None
        label_in_paper = (data.get("group_label_in_paper") or "").strip()
        quote = (data.get("quote") or "").strip()

        # Guard: check the paper's OWN cohort label — and the quote, which may
        # name a different arm than the label claims — against the cohorts this
        # review reports. Three outcomes, and "not confirmed" is not "fine".
        verdict, reason = "ok", ""
        if found:
            verdict, reason = cohort_conflicts(payload.group, label_in_paper,
                                               cohorts=payload.cohorts)
            if verdict != "wrong_cohort":
                # The quote is prose, not a label: it can only ever PROVE the
                # value came from another arm ("Regarding diabetic children …"
                # quoted for a control ask). Not resembling a short cohort name
                # is normal for a sentence and says nothing either way.
                quote_verdict, quote_reason = cohort_conflicts(
                    payload.group, quote, cohorts=payload.cohorts)
                if quote_verdict == "wrong_cohort":
                    verdict, reason = quote_verdict, quote_reason

        if verdict == "wrong_cohort":
            logger.info("extract_source_wrong_cohort", target=payload.group,
                        got=label_in_paper, quote=quote[:80])
            return SourceValueResult(
                found=False, group_label_in_paper=label_in_paper,
                wrong_group_rejected=True, cohort_check=verdict, cohort_reason=reason,
                not_found_reason=reason)

        return SourceValueResult(
            found=found,
            value=value if found else None,
            unit=(data.get("unit") or "").strip(),
            quote=quote,
            source_field_name=(data.get("source_field_name") or "").strip(),
            location=(data.get("location") or "").strip(),
            group_label_in_paper=label_in_paper,
            cohort_check=verdict, cohort_reason=reason,
            # The model was asked WHY when it found nothing; keep its answer,
            # and keep the cohorts it saw — both were being thrown away.
            not_found_reason=("" if found else
                              (data.get("not_found_reason") or "").strip()),
            cohorts_seen=[str(c) for c in (data.get("cohorts_seen") or [])
                          if isinstance(c, (str, int, float))],
        )
