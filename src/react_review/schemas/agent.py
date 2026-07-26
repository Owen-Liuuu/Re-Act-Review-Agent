"""Schemas for the bounded ReAct runtime (trajectory + processing record)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepRecord(BaseModel):
    """One Thought → Action → Observation step in an agent trajectory.

    Exactly one of ``observation`` / ``error`` is set for an executed action; a
    step whose tool was unknown records the ``error`` and no observation.
    """

    index: int
    thought: str = ""
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    observation: Any = None
    error: str = ""


class AgentRun(BaseModel):
    """The full record of a bounded agent run — the ProcessingRecord.

    Attributes:
        agent: which agent produced this run.
        task: the input the agent was given.
        steps: the ordered Thought/Action/Observation trajectory.
        status: ``finished`` (agent stopped itself) | ``max_steps`` (bounded
            cap hit) | ``error`` (unrecoverable).
        final: the agent's final output when it finished.
    """

    agent: str
    task: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)
    status: str = "finished"
    final: Any = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)
