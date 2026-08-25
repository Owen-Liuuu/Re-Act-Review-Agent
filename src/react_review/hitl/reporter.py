"""StepReporter — the one call a pipeline stage makes to report and ask.

Stages should not know about journals, gates, or rendering:

    decision = await reporter.step(StepStage.TABLE_CAPTURE, title=..., subject=pdf,
                                   payload=..., render_blocks=[...], offers=["retry"])
    if decision is Decision.STOP:
        raise RunStopped(...)

Ordering matters and is fixed here: the artifact is written FIRST, then the gate
is consulted, then the decision is folded back into the artifact. A Ctrl-C at the
prompt therefore still leaves the step's full content on disk.
"""
from __future__ import annotations

import time

from pathlib import Path

from react_review.core.config import PathSettings
from react_review.core.exceptions import RunStopped
from react_review.hitl.events import StepEvent, StepStage, SubjectKind
from react_review.hitl.gate import AutoContinue, CheckpointGate, Decision
from react_review.hitl.journal import NullJournal, RunJournal
from react_review.hitl.policy import Mode
from react_review.hitl.render import checkpoint_log_header, render_screen
from react_review.llm.reasoning import take_backend_trace


class StepReporter:
    """Emit step artifacts and consult the checkpoint gate."""

    def __init__(
        self,
        run_id: str = "",
        *,
        gate: CheckpointGate | None = None,
        journal: RunJournal | NullJournal | None = None,
    ) -> None:
        self.run_id = run_id
        self.gate: CheckpointGate = gate or AutoContinue()
        self.journal = journal or NullJournal()
        # The event most recently gated. A checkpoint may EDIT it (dropping a
        # captured table, say), so stages that offer edits read the result back
        # from here rather than from the payload they passed in.
        self.last_event: StepEvent | None = None
        self._screen = 0
        self._held_screen = 0
        self.checkpoint_header: str = ""
        self._checkpoint_header_written = False

    async def step(
        self,
        stage: StepStage,
        *,
        title: str = "",
        subject: str = "",
        subject_kind: SubjectKind = SubjectKind.NONE,
        payload: dict | None = None,
        render_blocks: list[str] | None = None,
        warnings: list[str] | None = None,
        offers: list[str] | None = None,
        selectable: str = "",
        sidecars: dict[str, str] | None = None,
        started: float | None = None,
        force_gate: bool = False,
        hold_display: bool = False,
    ) -> Decision:
        """Report one step and return the human's decision."""
        event = StepEvent(
            run_id=self.run_id, index=self.journal.next_index(),
            screen=self._next_screen(
                stage, force_gate=force_gate, hold_display=hold_display),
            stage=stage,
            title=title, subject=subject, subject_kind=subject_kind,
            payload=payload or {}, render_blocks=render_blocks or [],
            warnings=warnings or [], offers=offers or [], selectable=selectable,
            elapsed_ms=int((time.monotonic() - started) * 1000) if started else 0,
        )
        trace = take_backend_trace()
        if trace:
            event.backend_profile = str(trace.get("profile") or "")
            event.backend_model_id = str(trace.get("model_id") or "")
            event.backend_reasoning = str(trace.get("reasoning") or "")
            tokens = trace.get("reasoning_tokens")
            event.backend_reasoning_tokens = (
                tokens if isinstance(tokens, int) else None)
        self.last_event = event
        self.journal.emit(event, sidecars=sidecars)      # disk first — survives Ctrl-C
        check = self.gate.check
        try:
            decision = await check(
                event, force_gate=force_gate, hold_display=hold_display)
        except TypeError:
            decision = await check(event, force_gate=force_gate)
        self.journal.record_decision(event)              # also records any edits
        self._append_checkpoint(event)
        return decision

    def _next_screen(self, stage: StepStage, *, force_gate: bool,
                     hold_display: bool = False) -> int:
        """Visible-checkpoint number, counted from printed screens.

        Silent journal-only steps stay at 0. Consecutive ``hold_display``
        steps share one number so three journal artifacts can occupy one
        screen. The number is derived from how many visible screens have
        already been issued — it is not a table of stage→N.
        """
        policy = getattr(self.gate, "_policy", None)
        visible = True
        if policy is not None:
            visible = force_gate or policy.mode_for(stage) is not Mode.SILENT
        if not visible:
            return 0
        if hold_display:
            if not self._held_screen:
                self._screen += 1
                self._held_screen = self._screen
            return self._held_screen
        if self._held_screen:
            number = self._held_screen
            self._held_screen = 0
            return number
        self._screen += 1
        return self._screen

    def _append_checkpoint(self, event: StepEvent) -> None:
        """Human transcript of every step, including SILENT ones."""
        run_dir = getattr(self.journal, "run_dir", None)
        if run_dir is None:
            return
        path = Path(run_dir) / PathSettings().checkpoint_log
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._checkpoint_header_written:
            header = self.checkpoint_header or checkpoint_log_header(
                run_id=self.run_id)
            path.write_text(header if header.endswith("\n") else header + "\n",
                            encoding="utf-8")
            self._checkpoint_header_written = True
        body = render_screen([event], stable=True).rstrip()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(body + "\n")

    async def step_or_stop(self, stage: StepStage, **kw) -> Decision:
        """Like :meth:`step`, but raise :class:`RunStopped` on a STOP decision."""
        decision = await self.step(stage, **kw)
        if decision is Decision.STOP:
            raise RunStopped(
                stage=stage.value,
                index=self.last_event.index if self.last_event else 0,
                reason=f"stopped by user at {stage.value}",
            )
        return decision

    def progress(
        self,
        label: str,
        index: int | None = None,
        total: int | None = None,
        *,
        caption: str = "",
        started: float | None = None,
    ) -> None:
        """One discrete progress line. No-op unless the gate prints them."""
        sink = getattr(self.gate, "progress", None)
        if not callable(sink):
            return
        elapsed_s = (time.monotonic() - started) if started is not None else None
        sink(label, index, total, caption=caption, elapsed_s=elapsed_s)
