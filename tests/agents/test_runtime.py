"""Tests for the bounded ReAct runtime (deterministic, stub policy)."""
from __future__ import annotations

import pytest

from react_review.agents.runtime import (
    AgentPolicy,
    BoundedReActAgent,
    ProposedAction,
)
from react_review.audit import ToleranceTable
from react_review.core.enums import AuditLabel
from react_review.schemas.agent import StepRecord
from react_review.tools.compare import CompareValuesTool


class ScriptedPolicy(AgentPolicy):
    """Emits a fixed list of actions in order (last one usually finishes)."""

    def __init__(self, actions: list[ProposedAction]) -> None:
        self._actions = actions
        self.seen: list[int] = []

    async def propose(self, task, steps, tools) -> ProposedAction:
        i = len(self.seen)
        self.seen.append(i)
        # If the script runs out, keep proposing a no-op tool call (never finish)
        # so the max_steps bound is exercised.
        if i < len(self._actions):
            return self._actions[i]
        return ProposedAction(tool="compare_values", args={"field_type": "x"})


def _compare_tool() -> CompareValuesTool:
    return CompareValuesTool(ToleranceTable())


@pytest.mark.asyncio
async def test_agent_runs_action_then_finishes():
    policy = ScriptedPolicy([
        ProposedAction(
            thought="compare the two values",
            tool="compare_values",
            args={"field_type": "eat_thickness", "review_value": "6.60",
                  "source_value": "6.60", "review_unit": "mm", "source_unit": "mm"},
        ),
        ProposedAction(thought="done", finish=True, final={"label": "match"}),
    ])
    agent = BoundedReActAgent("collector", policy, [_compare_tool()], max_steps=8)
    run = await agent.run({"goal": "audit one value"})

    assert run.status == "finished"
    assert run.n_steps == 1                       # the finish step is not recorded
    step = run.steps[0]
    assert step.tool == "compare_values"
    assert step.error == ""
    assert step.observation["label"] == AuditLabel.MATCH.value
    assert run.final == {"label": "match"}


@pytest.mark.asyncio
async def test_max_steps_bound_is_enforced():
    # A policy that never finishes must stop at max_steps.
    policy = ScriptedPolicy([])  # always proposes a no-op action
    agent = BoundedReActAgent("collector", policy, [_compare_tool()], max_steps=3)
    run = await agent.run({})
    assert run.status == "max_steps"
    assert run.n_steps == 3


@pytest.mark.asyncio
async def test_unknown_tool_is_recorded_not_crashed():
    policy = ScriptedPolicy([
        ProposedAction(thought="try a missing tool", tool="does_not_exist", args={}),
        ProposedAction(finish=True, final="ok"),
    ])
    agent = BoundedReActAgent("collector", policy, [_compare_tool()])
    run = await agent.run({})
    assert run.status == "finished"
    assert run.steps[0].error.startswith("unknown tool")
    assert run.steps[0].observation is None


@pytest.mark.asyncio
async def test_tool_error_is_recorded_not_crashed():
    # Bad args (missing required field_type) -> tool input validation error,
    # captured as a step error rather than aborting the run.
    policy = ScriptedPolicy([
        ProposedAction(tool="compare_values", args={"review_value": "1"}),  # no field_type
        ProposedAction(finish=True, final="recovered"),
    ])
    agent = BoundedReActAgent("collector", policy, [_compare_tool()])
    run = await agent.run({})
    assert run.status == "finished"
    assert run.steps[0].error != ""
    assert run.final == "recovered"


def test_agent_exposes_only_its_subset():
    agent = BoundedReActAgent("auditor", ScriptedPolicy([]), [_compare_tool()])
    assert agent.tool_names == ["compare_values"]
