"""Step 1 — which tables/figures are on the evidence chain. Lens + results window only."""
from __future__ import annotations

import structlog

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.parser.review_extraction.prompts import render_extraction_prompt
from react_review.parser.review_extraction.schemas import DisplayHit, ReviewLens
from react_review.parser.review_extraction.windows import results_window

logger = structlog.get_logger(__name__)

_KINDS = {"pdf_table", "forest_plot", "other"}


def _hit(raw: object, index: int) -> DisplayHit | None:
    if not isinstance(raw, dict):
        return None
    display_id = str(raw.get("display_id") or "").strip() or f"display_{index}"
    kind = str(raw.get("kind") or "other").strip().lower()
    if kind not in _KINDS:
        kind = "other"
    evidence = raw.get("evidence_chain")
    if isinstance(evidence, str):
        evidence = evidence.strip().lower() in {"true", "yes", "1"}
    page_hint = str(raw.get("page_hint") or "").strip()
    if page_hint and not page_hint.isdigit():
        logger.warning("page_hint_not_numeric", got=page_hint[:40], display_id=display_id)
        page_hint = ""
    return DisplayHit(
        display_id=display_id,
        kind=kind,  # type: ignore[arg-type]
        caption=str(raw.get("caption") or "").strip(),
        page_hint=page_hint,
        evidence_chain=bool(evidence),
        reason=str(raw.get("reason") or "").strip(),
    )


async def localize(
    backend: LLMBackend, lens: ReviewLens, text: str,
) -> list[DisplayHit]:
    """Return candidate displays. Product rules live in the prompt, not in regex."""
    window = results_window(text)
    prompt = render_extraction_prompt(
        "evidence_localize_v2",
        lens=lens.as_ruler() or "(empty lens)",
        results_window=window or "(no results window)",
    )
    try:
        raw = parse_llm_response(await backend.complete(prompt), backend.model_id)
    except Exception:  # noqa: BLE001
        return []
    displays = raw.get("displays") if isinstance(raw, dict) else None
    if not isinstance(displays, list):
        return []
    hits: list[DisplayHit] = []
    for i, body in enumerate(displays, start=1):
        hit = _hit(body, i)
        if hit is not None:
            hits.append(hit)
    return hits


def selected(hits: list[DisplayHit], *, kind: str) -> list[DisplayHit]:
    return [h for h in hits if h.evidence_chain and h.kind == kind]
