"""Terminal rendering for checkpoints — pure string functions plus safe output.

Windows consoles are frequently GBK, where box-drawing glyphs and ``±`` raise
UnicodeEncodeError. Everything here degrades to ASCII rather than crashing a run
at the moment it is trying to show its work.
"""
from __future__ import annotations

import sys

from react_review.hitl.events import StepEvent

_BOX = {"h": "─", "v": "│", "tl": "┌", "tr": "┐", "bl": "└", "br": "┘"}
_ASCII = {"h": "-", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+"}


def supports_unicode(stream=None) -> bool:
    """True when the output encoding can carry box-drawing characters."""
    enc = getattr(stream or sys.stdout, "encoding", None) or ""
    try:
        "─±".encode(enc)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def box_chars(stream=None) -> dict[str, str]:
    return _BOX if supports_unicode(stream) else _ASCII


def safe_print(text: str) -> None:
    """Print, replacing characters the console cannot encode (Windows GBK)."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def rule(title: str = "", width: int = 78, stream=None) -> str:
    ch = box_chars(stream)["h"]
    if not title:
        return ch * width
    head = f"{ch}{ch} {title} "
    return head + ch * max(0, width - len(head))


def render_event(event: StepEvent, *, width: int = 78, stream=None) -> str:
    """The full checkpoint block: what step, which file, what it produced."""
    return render_screen([event], width=width, stream=stream)


def render_screen(events: list[StepEvent], *, width: int = 78, stream=None,
                  stable: bool = False) -> str:
    """One visible screen. Several journaled steps may share it.

    ``stable=True`` drops the three lines that change between identical runs
    (elapsed time, absolute path, backend/token row) so a checkpoint log can
    be diffed. It does not use a second renderer.
    """
    if not events:
        return ""
    if _is_review_parsed(events):
        return _render_review_parsed(
            events, width=width, stream=stream, stable=stable)
    if len(events) == 1:
        return _render_one(events[0], width=width, stream=stream, stable=stable)
    heading_event = events[0].model_copy(update={
        "title": events[-1].title or events[0].title,
        "render_blocks": [block for event in events for block in event.render_blocks],
        "warnings": [w for event in events for w in event.warnings],
    })
    return _render_one(heading_event, width=width, stream=stream, stable=stable)


def _render_one(event: StepEvent, *, width: int = 78, stream=None,
                stable: bool = False) -> str:
    number = event.screen or event.index
    heading = f"[{number}] {event.title or event.stage.value}"
    if event.elapsed_ms and not stable:
        heading += f"  ({_elapsed_label(event.elapsed_ms)})"
    lines: list[str] = ["", rule(heading, width, stream)]
    if stable:
        lines.append(
            f"    interaction: {event.interaction or '-'}      "
            f"decision: {event.decision or '-'}")
    if event.subject and not stable:
        lines.append("")
        lines.append(f"  file: {event.subject}")
    backend = _backend_line(event)
    if backend and not stable:
        lines.append(backend)
    for block in event.render_blocks:
        lines.append("")
        lines.append(block)
    if event.warnings:
        lines.append("")
        lines.append("  warnings:")
        lines.extend(f"    ! {w}" for w in event.warnings)
    return "\n".join(lines)


_REVIEW_PARSED = {
    "cohort_registry", "field_resolution", "checklist_review",
}


def _is_review_parsed(events: list[StepEvent]) -> bool:
    stages = {event.stage.value for event in events}
    return len(events) > 1 and stages <= _REVIEW_PARSED


def _render_review_parsed(events: list[StepEvent], *, width: int = 78,
                          stream=None, stable: bool = False) -> str:
    head = events[0]
    number = head.screen or head.index
    heading = f"[{number}] Review parsed"
    lines: list[str] = ["", rule(heading, width, stream)]
    if stable:
        last = events[-1]
        lines.append(
            f"    interaction: {last.interaction or '-'}      "
            f"decision: {last.decision or '-'}")
    if head.subject and not stable:
        lines.append("")
        lines.append(f"  file: {head.subject}")
    by_stage = {event.stage.value: event for event in events}
    lines.append("")
    lines.append(_cohorts_line(by_stage.get("cohort_registry")))
    lines.append(_concepts_line(by_stage.get("field_resolution")))
    lines.append(_checklist_line(by_stage.get("checklist_review")))
    warnings = [w for event in events for w in event.warnings]
    if warnings:
        lines.append("")
        lines.append("  warnings:")
        lines.extend(f"    ! {w}" for w in warnings)
    return "\n".join(lines)


def _cohorts_line(event: StepEvent | None) -> str:
    if event is None:
        return "    cohorts:    (not reported)"
    cohorts = event.payload.get("cohorts") or []
    unknown = event.payload.get("unassigned") or []
    if not cohorts:
        return "    cohorts:    0 found"
    bits = [f'{row.get("key")} "{row.get("display")}"' for row in cohorts]
    tag = "unknown" if unknown else "discovered"
    return (f"    cohorts:    {len(cohorts)} found — "
            f"{' · '.join(bits)}   [{tag}]")


def _concepts_line(event: StepEvent | None) -> str:
    if event is None:
        return "    concepts:   (not reported)"
    kb = event.payload.get("knowledge_base") or {}
    count = kb.get("concept_count")
    fingerprint = str(kb.get("fingerprint") or "")
    short = fingerprint[:8] + ("…" if len(fingerprint) > 8 else "")
    n = count if count is not None else len(event.payload.get("resolutions") or [])
    return f"    concepts:   {n} in KB · fingerprint {short}"


def _checklist_line(event: StepEvent | None) -> str:
    if event is None:
        return "    checklist:  (not applied)"
    name = str(event.payload.get("name") or event.payload.get("checklist_id")
               or "checklist")
    version = event.payload.get("version")
    items = event.payload.get("assessments") or []
    gaps = event.payload.get("gaps") or []
    ver = f" v{version}" if version not in (None, "") else ""
    return (f"    checklist:  {name}{ver} · {len(items)} item · "
            f"{len(gaps)} required gap")


def _elapsed_label(elapsed_ms: int) -> str:
    seconds = max(0, int(round(elapsed_ms / 1000)))
    return f"{seconds}s"


def render_progress(
    label: str,
    index: int | None = None,
    total: int | None = None,
    *,
    caption: str = "",
    elapsed_s: float | None = None,
) -> str:
    """One discrete progress line. Never uses carriage-return overwrite."""
    parts = ["   ⋯", label]
    if index is not None and total is not None:
        parts.append(f"{index}/{total}")
    if caption:
        clipped = caption if len(caption) <= 42 else caption[:41] + "…"
        parts.append(f'"{clipped}"')
    if elapsed_s is not None:
        parts.append(f"{int(round(max(0, elapsed_s)))}s")
    return " ".join(parts)


def render_prompt(event: StepEvent, *, allow_skip: bool = False,
                  undo_available: bool = False) -> str:
    """The one-line question. Keep it cheap to answer: C continues."""
    opts = ["[C]Continue", "[S]Stop"]
    if event.selectable_items():
        opts.append("[N]On <n>")
        opts.append("[F]Off <n>")
    if undo_available:
        opts.append("[U]Undo")
    if "retry" in event.offers:
        opts.append("[R]Retry")
    if "retry_alt" in event.offers:
        opts.append("[M]Retry with Model 2")
    opts += ["[D]Detail", "[O]Open artifact"]
    if allow_skip:
        opts.append("[A]All (skip remaining checkpoints)")
    return "  " + "  ".join(opts) + " > "


def render_selectable(event: StepEvent, *, action: str = "set") -> str:
    """The numbered list shown when the human sets an item on or off."""
    lines = [f"  {action} which?"]
    for i, item in enumerate(event.selectable_items(), start=1):
        label = item.get("label") or item.get("id") or item.get("table_id") or f"item {i}"
        lines.append(f"    [{i}] {label}")
    return "\n".join(lines)


def _backend_line(event: StepEvent) -> str:
    if (not event.backend_profile and not event.backend_model_id
            and event.backend_reasoning_tokens is None):
        return ""
    parts = []
    if event.backend_profile:
        parts.append(f"profile: {event.backend_profile}")
    if event.backend_model_id:
        parts.append(f"model_id: {event.backend_model_id}")
    if event.backend_reasoning:
        parts.append(f"reasoning: {event.backend_reasoning}")
    if event.backend_reasoning_tokens is None:
        parts.append("reasoning_tokens: None")
    else:
        parts.append(f"reasoning_tokens: {event.backend_reasoning_tokens}")
    return "  " + " · ".join(parts)


def checkpoint_log_header(
    *,
    run_id: str,
    review: str = "",
    models: dict[str, str] | None = None,
    prompts: dict[str, str] | None = None,
    config_summary: str = "",
) -> str:
    """Once-per-file identity. Volatile per-step facts stay out of the body."""
    lines = [f"run_id: {run_id}"]
    if review:
        lines.append(f"review: {review}")
    for name, model in (models or {}).items():
        if model:
            lines.append(f"model.{name}: {model}")
    for name, profile in (prompts or {}).items():
        if profile:
            lines.append(f"prompt.{name}: {profile}")
    if config_summary:
        lines.append(f"config: {config_summary}")
    return "\n".join(lines) + "\n"
