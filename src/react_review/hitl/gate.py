"""The checkpoint gate — the ONLY place a pipeline waits for a human.

The gate is injected, never called globally: a stage does
``decision = await gate.check(event)`` and obeys the answer. The library default
is :class:`AutoContinue`, so tests, the eval runners, and any programmatic use
run start-to-finish unattended; only the CLI installs the interactive one
(``hitl.console.ConsoleCheckpoint``).

That inversion is what keeps "the pipeline must stop and ask" from turning into
an untestable ``input()`` buried in business logic.
"""
from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from react_review.hitl.events import StepEvent


class Decision(str, Enum):
    """What the human (or the default policy) decided at a checkpoint."""

    CONTINUE = "continue"
    STOP = "stop"
    RETRY = "retry"          # re-run this stage (e.g. re-extract the table)
    RETRY_ALT = "retry_alt"  # re-run it with the fallback model
    # Present in the vocabulary but NOT bound to a key unless --allow-skip:
    # findings are shown unconditionally until the pipeline has earned trust.
    SKIP_REST = "skip_rest"


def retry_offers(alt_backend: object | None) -> list[str]:
    """Retry keys for a checkpoint. Model 2 is offered only when a fallback exists."""
    offers = ["retry"]
    if alt_backend is not None:
        offers.append("retry_alt")
    return offers


def require_alt_backend(alt_backend, *, stage: str):
    """Raise rather than silently accepting when Model 2 was requested but missing."""
    if alt_backend is None:
        raise RuntimeError(
            f"{stage}: Retry with Model 2 was requested but no alt_backend "
            "is configured")
    return alt_backend


@runtime_checkable
class CheckpointGate(Protocol):
    """Decide whether a run may proceed past ``event``."""

    async def check(self, event: StepEvent, *, force_gate: bool = False) -> Decision:
        ...


class AutoContinue:
    """Never blocks — the library/CI default. Records the decision on the event."""

    async def check(self, event: StepEvent, *, force_gate: bool = False) -> Decision:
        event.decision = Decision.CONTINUE.value
        return Decision.CONTINUE


class ScriptedCheckpoint:
    """Replay a fixed list of decisions (tests). Falls back to CONTINUE."""

    def __init__(self, decisions: list[Decision] | None = None) -> None:
        self._queue = list(decisions or [])
        self.seen: list[StepEvent] = []

    async def check(self, event: StepEvent, *, force_gate: bool = False) -> Decision:
        self.seen.append(event)
        decision = self._queue.pop(0) if self._queue else Decision.CONTINUE
        event.decision = decision.value
        return decision
