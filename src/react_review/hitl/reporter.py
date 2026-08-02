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
    ) -> Decision:
        """Report one step and return the human's decision."""
        event = StepEvent(
            run_id=self.run_id, index=self.journal.next_index(), stage=stage,
            title=title, subject=subject, subject_kind=subject_kind,
            payload=payload or {}, render_blocks=render_blocks or [],
            warnings=warnings or [], offers=offers or [], selectable=selectable,
            elapsed_ms=int((time.monotonic() - started) * 1000) if started else 0,
        )
        self.last_event = event
        self.journal.emit(event, sidecars=sidecars)      # disk first — survives Ctrl-C
        decision = await self.gate.check(event)
        self.journal.record_decision(event)              # also records any edits
        return decision

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
