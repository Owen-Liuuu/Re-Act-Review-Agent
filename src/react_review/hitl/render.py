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
    lines: list[str] = ["", rule(f"[{event.index}] {event.title or event.stage.value}",
                                width, stream)]
    if event.subject:
        lines.append(f"  file: {event.subject}")
    for block in event.render_blocks:
        lines.append("")
        lines.append(block)
    if event.warnings:
        lines.append("")
        lines.append("  warnings:")
        lines.extend(f"    ! {w}" for w in event.warnings)
    return "\n".join(lines)


def render_prompt(event: StepEvent, *, allow_skip: bool = False) -> str:
    """The one-line question. Keep it cheap to answer: C continues."""
    opts = ["[C]ontinue", "[S]top"]
    if event.selectable_items():
        opts.append("[X] drop one")
    if "retry" in event.offers:
        opts.append("[R]etry")
    if "retry_alt" in event.offers:
        opts.append("retry with [M]odel2")
    opts += ["[D]etail", "[O]pen artifact"]
    if allow_skip:
        opts.append("[A]ll (skip remaining checkpoints)")
    return "  " + "  ".join(opts) + " > "


def render_selectable(event: StepEvent) -> str:
    """The numbered list shown when the human asks to drop something."""
    lines = ["  drop which?"]
    for i, item in enumerate(event.selectable_items(), start=1):
        label = item.get("label") or item.get("id") or item.get("table_id") or f"item {i}"
        lines.append(f"    [{i}] {label}")
    return "\n".join(lines)
