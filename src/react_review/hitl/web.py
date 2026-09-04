"""Web checkpoint gate — the run pauses here until a browser posts a decision.

The reporter has already written the step artifact. This object only waits.
``interaction = "web"`` is what later proves a person confirmed the step.
"""
from __future__ import annotations

import asyncio
import queue
import threading

from react_review.hitl.events import StepEvent
from react_review.hitl.gate import Decision
from react_review.hitl.policy import CheckpointPolicy, Mode


class StaleDecision(ValueError):
    """Posted for a step that is not the one currently waiting."""


class WebCheckpoint:
    """A :class:`CheckpointGate` filled by ``POST /runs/{id}/decision``."""

    def __init__(self, policy: CheckpointPolicy | None = None) -> None:
        self._policy = policy or CheckpointPolicy.key_stages()
        self._inbox: queue.Queue[Decision] = queue.Queue()
        self._lock = threading.Lock()
        self._waiting: StepEvent | None = None

    def progress(
        self,
        label: str,
        index: int | None = None,
        total: int | None = None,
        *,
        caption: str = "",
        elapsed_s: float | None = None,
    ) -> None:
        """Print each in-flight stage to the serve terminal (testing trail)."""
        from react_review.hitl.render import render_progress, safe_print
        safe_print(render_progress(
            label, index, total, caption=caption, elapsed_s=elapsed_s))

    @property
    def waiting(self) -> StepEvent | None:
        with self._lock:
            return self._waiting

    async def check(self, event: StepEvent, *, force_gate: bool = False,
                    hold_display: bool = False) -> Decision:
        mode = Mode.GATE if force_gate else self._policy.mode_for(event.stage)
        if hold_display or mode is not Mode.GATE:
            event.interaction = "silent" if mode is Mode.SILENT else "show"
            event.decision = Decision.CONTINUE.value
            return Decision.CONTINUE

        while True:
            try:
                self._inbox.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._waiting = event
        try:
            decision = await asyncio.to_thread(self._inbox.get)
        finally:
            with self._lock:
                self._waiting = None
        event.interaction = "web"
        event.decision = decision.value
        return decision

    def submit(
        self,
        index: int,
        decision: Decision,
        *,
        dropped: list[str] | None = None,
    ) -> None:
        with self._lock:
            waiting = self._waiting
            if waiting is None or waiting.index != index:
                raise StaleDecision(f"no gate waiting at step {index}")
            if dropped is not None:
                _apply_dropped(waiting, dropped)
            self._inbox.put(decision)


def _apply_dropped(event: StepEvent, dropped_ids: list[str]) -> None:
    want = {str(x) for x in dropped_ids}
    for i, item in enumerate(event.selectable_items(), start=1):
        rid = str(item.get("id") or item.get("table_id")
                  or item.get("display_id") or i)
        if rid in want:
            event.set_off(i)
        else:
            event.set_on(i)
