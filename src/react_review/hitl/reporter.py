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

from react_review.core.exceptions import RunStopped
from react_review.hitl.events import StepEvent, StepStage, SubjectKind
from react_review.hitl.gate import AutoContinue, CheckpointGate, Decision
from react_review.hitl.journal import NullJournal, RunJournal
from react_review.hitl.policy import Mode
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
    ) -> Decision:
        """Report one step and return the human's decision."""
        event = StepEvent(
            run_id=self.run_id, index=self.journal.next_index(),
            screen=self._next_screen(stage, force_gate=force_gate),
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
        decision = await self.gate.check(event, force_gate=force_gate)
        self.journal.record_decision(event)              # also records any edits
        return decision

    def _next_screen(self, stage: StepStage, *, force_gate: bool) -> int:
        """Visible-checkpoint number. Silent journal-only steps stay at 0."""
        policy = getattr(self.gate, "_policy", None)
        visible = True
        if policy is not None:
            visible = force_gate or policy.mode_for(stage) is not Mode.SILENT
        if not visible:
            return 0
        self._screen += 1
        return self._screen

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
