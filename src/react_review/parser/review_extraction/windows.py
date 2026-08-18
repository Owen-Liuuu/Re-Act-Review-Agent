"""Deterministic text windows. No LLM, and never the whole PDF."""
from __future__ import annotations

import re

_INTRO = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*\s+)?introduction\b", re.I)
_METHODS = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*\s+)?methods?\b", re.I)
_RESULTS = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*\s+)?results?\b", re.I)
_AFTER_RESULTS = re.compile(
    r"(?m)^\s*(?:\d+(?:\.\d+)*\s+)?(?:discussion|references|bibliography)\b",
    re.I)
_REFS = re.compile(r"(?im)^\s*(references|bibliography|reference list)\b")


def clip_words(text: str, n: int) -> str:
    """Keep at most ``n`` whitespace-separated words."""
    words = str(text or "").split()
    return " ".join(words[: max(0, n)])


def front_matter(text: str, *, limit: int = 4000) -> str:
    """Title + abstract-ish region, stopping at Introduction when present."""
    body = text or ""
    intro = _INTRO.search(body)
    head = body[: intro.start()] if intro else body[:8000]
    return head[:limit]


def _strip_refs(text: str) -> str:
    last = -1
    for match in _REFS.finditer(text):
        last = match.start()
    if last > 200:
        return text[:last]
    return text


def results_window(text: str, *, limit: int = 15000) -> str:
    """Results (or equivalent) through the start of Discussion / References.

    Localize sees this window plus the lens — not the abstract, not the
    reference list, not a 50k-character dump of the PDF.
    """
    body = text or ""
    start_match = _RESULTS.search(body)
    if start_match is None:
        return _strip_refs(body)[:limit]
    rest = body[start_match.start():]
    end = len(rest)
    for match in _AFTER_RESULTS.finditer(rest):
        if match.start() > 40:
            end = match.start()
            break
    return rest[:end][:limit]


def capture_window(text: str, *, limit: int = 20000) -> str:
    """Methods/Results through Discussion — enough to transcribe selected tables.

    The abstract stays in the lens. The reference list is not a table source.
    """
    body = text or ""
    start_match = _METHODS.search(body) or _RESULTS.search(body)
    if start_match is None:
        return _strip_refs(body)[:limit]
    rest = body[start_match.start():]
    end = len(rest)
    for match in _AFTER_RESULTS.finditer(rest):
        if match.start() > 40:
            end = match.start()
            break
    return rest[:end][:limit]


def missed_forest_hint(window: str, hits: list) -> str:
    """A warning when the results text talks about a forest plot we did not list."""
    text = (window or "").lower()
    if "forest" not in text:
        return ""
    if any(getattr(h, "kind", "") == "forest_plot" for h in hits):
        return ""
    return ("results text mentions a forest plot, but none was listed — "
            "the figure may be image-only")
