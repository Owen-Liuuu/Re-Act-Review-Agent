"""Human-in-the-loop: step events, per-step artifacts, and the checkpoint gate.

The design premise, from the project's acceptance review: **a pipeline that runs
to completion and just prints a good result must be prevented.** Every stage
reports which file it read and the full content it produced, then asks whether
to continue.

Two halves, deliberately separate:

* the **journal** always writes — even unattended, a run leaves a per-step audit
  trail on disk (:class:`RunJournal`);
* the **gate** only blocks when someone is watching — the library default is
  :class:`AutoContinue`, so tests and eval runners are unaffected, while the CLI
  installs :class:`~react_review.hitl.console.ConsoleCheckpoint`.

Stages talk to both through one object, :class:`StepReporter`.
"""
from react_review.hitl.console import ConsoleCheckpoint
from react_review.hitl.events import StepEvent, StepStage, SubjectKind
from react_review.hitl.gate import (
    AutoContinue,
    CheckpointGate,
    Decision,
    ScriptedCheckpoint,
)
from react_review.hitl.journal import NullJournal, RunJournal
from react_review.hitl.policy import CheckpointPolicy, Mode
from react_review.hitl.render import (
    box_chars,
    render_event,
    render_progress,
    render_prompt,
    rule,
    safe_print,
    supports_unicode,
)
from react_review.hitl.reporter import StepReporter

__all__ = [
    "StepEvent", "StepStage", "SubjectKind",
    "Decision", "CheckpointGate", "AutoContinue", "ScriptedCheckpoint",
    "ConsoleCheckpoint",
    "RunJournal", "NullJournal",
    "CheckpointPolicy", "Mode",
    "safe_print", "render_event", "render_prompt", "render_progress", "rule", "box_chars",
    "supports_unicode",
    "StepReporter",
]
