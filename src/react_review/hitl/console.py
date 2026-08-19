"""The interactive checkpoint — the ONLY place this project reads the keyboard.

Everything else in the pipeline talks to the abstract :class:`CheckpointGate`, so
tests, CI and the eval runners never touch a terminal. Only the CLI installs this.

Design constraints that shaped it:

* **Cheap to answer.** One key. ``C`` continues; nothing else is required of a
  reviewer who is satisfied with what they just read.
* **Never wedge an automated run.** If stdin is not a TTY the gate returns
  CONTINUE without reading anything, so a piped or scripted invocation cannot hang.
* **Never block the event loop.** The key read happens on a worker thread.
"""
from __future__ import annotations

import asyncio
import json
import sys

from react_review.hitl.events import StepEvent
from react_review.hitl.gate import Decision
from react_review.hitl.policy import CheckpointPolicy, Mode
from react_review.hitl.render import (
    render_event,
    render_progress,
    render_prompt,
    render_selectable,
    safe_print,
)

# Keys that mean "go on" — Enter and space included so the reflex works.
_CONTINUE_KEYS = {"c", "\r", "\n", " "}
_STOP_KEYS = {"s", "q"}


class ConsoleCheckpoint:
    """Show a step, then wait for a keypress deciding whether to continue."""

    def __init__(
        self,
        policy: CheckpointPolicy | None = None,
        *,
        journal=None,
        allow_skip: bool = False,
    ) -> None:
        self._policy = policy or CheckpointPolicy.key_stages()
        self._journal = journal
        self._allow_skip = allow_skip
        self._skip_all = False
        self._drop_undo: list[tuple[int, dict, list[str]]] = []

    def progress(
        self,
        label: str,
        index: int | None = None,
        total: int | None = None,
        *,
        caption: str = "",
        elapsed_s: float | None = None,
    ) -> None:
        safe_print(render_progress(
            label, index, total, caption=caption, elapsed_s=elapsed_s))

    async def check(self, event: StepEvent, *, force_gate: bool = False) -> Decision:
        mode = Mode.GATE if force_gate else self._policy.mode_for(event.stage)
        if mode is Mode.SILENT:
            event.decision = Decision.CONTINUE.value
            return Decision.CONTINUE

        safe_print(render_event(event))

        if mode is Mode.SHOW or self._skip_all:
            event.decision = Decision.CONTINUE.value
            return Decision.CONTINUE

        decision = await self._ask(event)
        event.decision = decision.value
        return decision

    async def _ask(self, event: StepEvent) -> Decision:
        """Prompt until a key maps to a decision (inspect actions re-prompt)."""
        self._drop_undo = []
        while True:
            safe_print(render_prompt(
                event, allow_skip=self._allow_skip,
                undo_available=bool(self._drop_undo)))
            key = await asyncio.to_thread(self._read_key)

            if key in _CONTINUE_KEYS:
                return Decision.CONTINUE
            if key in _STOP_KEYS:
                return Decision.STOP
            if key == "r" and "retry" in event.offers:
                return Decision.RETRY
            if key == "m" and "retry_alt" in event.offers:
                return Decision.RETRY_ALT
            if key == "a" and self._allow_skip:
                self._skip_all = True
                return Decision.SKIP_REST
            if key == "n" and event.selectable_items():
                await self._set_one(event, on=True)
                continue
            if key == "f" and event.selectable_items():
                await self._set_one(event, on=False)
                continue
            if key == "u" and self._drop_undo:
                self._undo_set(event)
                continue
            if key == "d":
                safe_print(json.dumps(event.payload, indent=2, ensure_ascii=False))
                continue
            if key == "o":
                safe_print(f"  artifact: {self._artifact_path(event)}")
                continue
            # Anything else (including a swallowed arrow key) just re-prompts.

    async def _set_one(self, event: StepEvent, *, on: bool) -> None:
        """Set one selectable on or off, then re-show what remains active."""
        safe_print(render_selectable(event, action="on" if on else "off"))
        key = await asyncio.to_thread(self._read_key)
        if not key.isdigit():
            return
        items = event.selectable_items()
        choice = int(key)
        if not 1 <= choice <= len(items):
            safe_print("  (no such item)")
            return
        snapshot = dict(items[choice - 1])
        dropped_before = list(event.dropped)
        rid = event.set_on(choice) if on else event.set_off(choice)
        if not rid:
            safe_print("  (no such item)")
            return
        self._drop_undo.append((choice - 1, snapshot, dropped_before))
        safe_print(f"  set {rid} {'on' if on else 'off'}")
        if len(event.dropped) == len(event.selectable_items()):
            safe_print("  ! nothing left to process; [S]Stop unless this is intended")

    def _undo_set(self, event: StepEvent) -> None:
        """Restore the last On/Off from this pause. Does not cross steps."""
        index, item, dropped = self._drop_undo.pop()
        items = event.selectable_items()
        if 0 <= index < len(items):
            items[index].clear()
            items[index].update(item)
        event.dropped[:] = dropped
        rid = str(item.get("id") or item.get("table_id") or index + 1)
        safe_print(f"  restored {rid}")

    def _artifact_path(self, event: StepEvent) -> str:
        run_dir = getattr(self._journal, "run_dir", None)
        return str(run_dir / "steps" / f"{event.slug}.json") if run_dir else "(not journaled)"

    @staticmethod
    def _read_key() -> str:
        """Read one keypress. Returns "" for keys that should just re-prompt."""
        if not sys.stdin.isatty():
            return "c"          # piped/scripted: never block, always continue
        try:
            import msvcrt
        except ImportError:
            return ConsoleCheckpoint._read_key_posix()

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()     # arrow/function key: swallow the second half
            return ""
        if ch == "\x03":        # getwch does NOT raise KeyboardInterrupt for us
            raise KeyboardInterrupt
        return ch.lower()

    @staticmethod
    def _read_key_posix() -> str:
        import termios
        import tty

        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()
