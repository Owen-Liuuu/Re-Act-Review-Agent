"""Step 1 — which tables/figures are on the evidence chain. Lens + results window only."""
from __future__ import annotations

import re

import structlog

from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.observe import trace
from react_review.parser.review_extraction.prompts import render_extraction_prompt
from react_review.parser.review_extraction.schemas import DisplayHit, ReviewLens
from react_review.parser.review_extraction.windows import results_window

logger = structlog.get_logger(__name__)

_KINDS = {"pdf_table", "forest_plot", "other"}
#: Function words only. Outcome identity comes from the review's own lens
#: labels at runtime — this set must not grow disease vocabulary.
_STOP = frozenset({
    "a", "an", "the", "of", "and", "or", "for", "with", "vs", "versus",
    "in", "on", "to", "by", "from", "after", "per", "study", "plot",
    "forest", "figure", "table", "ruler", "outcome", "named", "included",
})


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
        outcome=str(raw.get("outcome") or "").strip(),
    )


def _content_tokens(text: str) -> set[str]:
    return {
        tok for tok in re.findall(r"[0-9a-z]+(?:-[0-9a-z]+)*", text.casefold())
        if tok not in _STOP and (len(tok) >= 3 or any(ch.isdigit() for ch in tok))
    }


def match_lens_outcome(text: str, outcomes: list[str]) -> str:
    """Return this review's own outcome label that ``text`` is about, or empty.

    Candidates are the lens strings of the review being parsed. No disease
    vocabulary lives here: a cardiology review yields cardiology names because
    that is what the lens copied, not because this function knows them.
    Empty when nothing uniquely fits, so a figure number cannot become the
    claim identity.
    """
    blob = " ".join(str(text or "").split())
    labels = [str(item or "").strip() for item in outcomes if str(item or "").strip()]
    if not blob or not labels:
        return ""
    folded = blob.casefold()
    exact = [name for name in labels if name.casefold() in folded]
    if exact:
        return max(exact, key=lambda name: (len(name), name))
    text_toks = _content_tokens(blob)
    if not text_toks:
        return ""
    scored: list[tuple[int, str]] = []
    for name in labels:
        overlap = len(_content_tokens(name) & text_toks)
        if overlap:
            scored.append((overlap, name))
    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    best_n, best = scored[0]
    if any(n == best_n for n, label in scored[1:]):
        return ""
    return best


def stamp_display_outcomes(hits: list[DisplayHit], outcomes: list[str]) -> None:
    """Put a lens outcome on each hit. Localize did not name one → leave empty."""
    for hit in hits:
        blob = " ".join(part for part in (hit.caption, hit.reason, hit.outcome) if part)
        hit.outcome = match_lens_outcome(blob, outcomes)


async def localize(
    backend: LLMBackend, lens: ReviewLens, text: str,
    *, notes: list[str] | None = None,
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
    except Exception as exc:  # noqa: BLE001
        trace(
            notes, "evidence_localize_call_failed",
            error_type=type(exc).__name__,
            message=f"evidence localize call failed: {exc}"[:200],
            error=str(exc)[:160],
        )
        return []
    displays = raw.get("displays") if isinstance(raw, dict) else None
    if not isinstance(displays, list):
        got = type(raw).__name__
        trace(
            notes, "evidence_localize_unparseable_response",
            error_type=got,
            message=f"evidence localize unparseable response: got {got}",
            got=got,
        )
        return []
    hits: list[DisplayHit] = []
    for i, body in enumerate(displays, start=1):
        hit = _hit(body, i)
        if hit is not None:
            hits.append(hit)
    return hits


def selected(hits: list[DisplayHit], *, kind: str) -> list[DisplayHit]:
    return [h for h in hits if h.evidence_chain and h.kind == kind]
