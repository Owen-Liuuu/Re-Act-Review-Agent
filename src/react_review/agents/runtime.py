"""The bounded ReAct runtime.

An agent runs a Thought → Action → Observation loop over a SMALL exposed subset
of the tool catalogue (AgentBench: tool-selection accuracy degrades as the action
space grows), capped at ``max_steps``. The runtime owns the loop, tool dispatch,
and the trajectory record; the *policy* (what to think / which action to take)
is abstracted so the runtime is testable with a deterministic stub and swappable
for an LLM-driven policy (P2).

Determinism note: the runtime itself is deterministic — bounded, records every
step, converts tool errors and unknown-tool choices into recorded observations
rather than crashing, so a misbehaving policy cannot run away or abort the run.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from react_review.schemas.agent import AgentRun, StepRecord
from react_review.tools.base import Tool


class ProposedAction(BaseModel):
    """A policy's proposal for the next step.

    Either an action (``tool`` + ``args``) or a finish (``finish=True`` +
    ``final``). ``thought`` is the ReAct rationale recorded for observability.
    """

    thought: str = ""
    tool: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    finish: bool = False
    final: Any = None


class AgentPolicy(ABC):
    """Decides the next step given the task and the trajectory so far."""

    @abstractmethod
    async def propose(
        self,
        task: dict[str, Any],
        steps: list[StepRecord],
        tools: list[Tool],
    ) -> ProposedAction:
        """Return the next :class:`ProposedAction` (an action or a finish)."""


class BoundedReActAgent:
    """Runs a policy against an exposed tool subset, bounded by ``max_steps``."""

    def __init__(
        self,
        name: str,
        policy: AgentPolicy,
        tools: list[Tool],
        *,
        max_steps: int = 8,
    ) -> None:
        self._name = name
        self._policy = policy
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self._max_steps = max(1, max_steps)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    async def run(self, task: dict[str, Any]) -> AgentRun:
        steps: list[StepRecord] = []
        exposed = list(self._tools.values())

        for i in range(self._max_steps):
            action = await self._policy.propose(task, steps, exposed)

            if action.finish:
                return AgentRun(
                    agent=self._name, task=task, steps=steps,
                    status="finished", final=action.final,
                )

            tool = self._tools.get(action.tool)
            if tool is None:
                # Unknown tool: record and let the policy recover next step
                # (bounded, so it cannot loop forever).
                steps.append(StepRecord(
                    index=i, thought=action.thought, tool=action.tool,
                    args=action.args,
                    error=(
                        f"unknown tool {action.tool!r}; "
                        f"available: {self.tool_names}"
                    ),
                ))
                continue

            try:
                payload = tool.input_model(**action.args)
                result = await tool.run(payload)
                observation = (
                    result.model_dump(mode="json")
                    if isinstance(result, BaseModel) else result
                )
                steps.append(StepRecord(
                    index=i, thought=action.thought, tool=action.tool,
                    args=action.args, observation=observation,
                ))
            except Exception as exc:  # a bad call is an observation, not a crash
                steps.append(StepRecord(
                    index=i, thought=action.thought, tool=action.tool,
                    args=action.args, error=str(exc),
                ))

        return AgentRun(
            agent=self._name, task=task, steps=steps, status="max_steps", final=None,
        )
