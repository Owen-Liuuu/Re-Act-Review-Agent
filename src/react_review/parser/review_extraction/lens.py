"""Step 0 — compress front matter into a ReviewLens. Later steps never see the abstract."""
from __future__ import annotations

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.parser.review_extraction.prompts import render_extraction_prompt
from react_review.parser.review_extraction.schemas import ReviewLens
from react_review.parser.review_extraction.windows import clip_words, front_matter

_LIMITS = {
    "lens_one_line": 40,
    "domain": 12,
    "population": 20,
    "comparison": 20,
}


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _clip_lens(raw: dict) -> ReviewLens:
    outcomes = _as_str_list(raw.get("outcomes"))[:8]
    return ReviewLens(
        lens_one_line=clip_words(str(raw.get("lens_one_line") or ""), _LIMITS["lens_one_line"]),
        domain=clip_words(str(raw.get("domain") or ""), _LIMITS["domain"]),
        population=clip_words(str(raw.get("population") or ""), _LIMITS["population"]),
        comparison=clip_words(str(raw.get("comparison") or ""), _LIMITS["comparison"]),
        outcomes=outcomes,
        not_audit_focus=_as_str_list(raw.get("not_audit_focus")),
        difficulties=_as_str_list(raw.get("difficulties")),
    )


async def read_lens(backend: LLMBackend, text: str, *, seed: int = 42) -> ReviewLens:
    """LLM-compress FRONT MATTER. The raw abstract is not returned."""
    window = front_matter(text)
    if not window.strip():
        return ReviewLens(difficulties=["no front matter was available"])
    prompt = render_extraction_prompt("review_lens_v1", front_matter=window)
    try:
        raw = parse_llm_response(
            await backend.complete(prompt, seed=seed), backend.model_id)
    except Exception:  # noqa: BLE001
        return ReviewLens(difficulties=["lens model call failed"])
    if not isinstance(raw, dict):
        return ReviewLens(difficulties=["lens response was not an object"])
    lens = _clip_lens(raw)
    if not lens.lens_one_line and not lens.difficulties:
        lens.difficulties.append("lens_one_line was empty after compression")
    return lens
