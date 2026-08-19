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
    number = event.screen or event.index
    heading = f"[{number}] {event.title or event.stage.value}"
    if event.elapsed_ms:
        heading += f"  ({_elapsed_label(event.elapsed_ms)})"
    lines: list[str] = ["", rule(heading, width, stream)]
    if event.subject:
        lines.append("")
        lines.append(f"  file: {event.subject}")
    for block in event.render_blocks:
        lines.append("")
        lines.append(block)
    if event.warnings:
        lines.append("")
        lines.append("  warnings:")
        lines.extend(f"    ! {w}" for w in event.warnings)
    return "\n".join(lines)


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
