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

    async def check(self, event: StepEvent) -> Decision:
        mode = self._policy.mode_for(event.stage)
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
        while True:
            safe_print(render_prompt(event, allow_skip=self._allow_skip))
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
            if key == "x" and event.selectable_items():
                await self._drop_one(event)
                continue
            if key == "d":
                safe_print(json.dumps(event.payload, indent=2, ensure_ascii=False))
                continue
            if key == "o":
                safe_print(f"  artifact: {self._artifact_path(event)}")
                continue
            # Anything else (including a swallowed arrow key) just re-prompts.

    async def _drop_one(self, event: StepEvent) -> None:
        """Let the human remove one captured item, then re-show what remains."""
        safe_print(render_selectable(event))
        key = await asyncio.to_thread(self._read_key)
        if not key.isdigit():
            return
        removed = event.drop(int(key))
        if not removed:
            safe_print("  (no such item)")
            return
        remaining = event.selectable_items()
        safe_print(f"  dropped {removed} — {len(remaining)} left")
        if not remaining:
            safe_print("  ! nothing left to process; [S]top unless this is intended")

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
