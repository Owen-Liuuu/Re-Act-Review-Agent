"""Tests for the LLM ReAct policy (stub backend — deterministic, no network)."""
from __future__ import annotations

import pytest

from react_review.agents.llm_policy import LLMReActPolicy
from react_review.agents.runtime import BoundedReActAgent
from react_review.audit import ToleranceTable
from react_review.llm.base import LLMBackend
from react_review.tools.compare import CompareValuesTool


class QueueBackend(LLMBackend):
    """Returns queued responses in order; records the prompts it received."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else '{"finish": true, "final": null}'


class RaisingBackend(LLMBackend):
    def __init__(self) -> None:
        super().__init__()

    @property
    def model_id(self) -> str:
        return "raiser"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        raise RuntimeError("boom")


def _compare_tool() -> CompareValuesTool:
    return CompareValuesTool(ToleranceTable())


@pytest.mark.asyncio
async def test_parse_action_response():
    policy = LLMReActPolicy(QueueBackend([
        '{"thought": "t", "action": {"tool": "compare_values", "args": {"field_type": "bmi"}}}'
    ]))
    a = await policy.propose({}, [], [_compare_tool()])
    assert a.finish is False
    assert a.tool == "compare_values"
    assert a.args == {"field_type": "bmi"}
    assert a.thought == "t"


@pytest.mark.asyncio
async def test_parse_finish_response():
    policy = LLMReActPolicy(QueueBackend(['{"thought": "done", "finish": true, "final": {"x": 1}}']))
    a = await policy.propose({}, [], [_compare_tool()])
    assert a.finish is True
    assert a.final == {"x": 1}


@pytest.mark.asyncio
async def test_parse_tolerates_flat_action():
    policy = LLMReActPolicy(QueueBackend(['{"tool": "compare_values", "args": {"field_type": "n"}}']))
    a = await policy.propose({}, [], [_compare_tool()])
    assert a.tool == "compare_values" and a.args == {"field_type": "n"}


@pytest.mark.asyncio
async def test_unparseable_response_does_not_crash():
    policy = LLMReActPolicy(QueueBackend(["this is not json"]))
    a = await policy.propose({}, [], [_compare_tool()])
    assert a.finish is False
    assert a.tool == "__unparseable__"   # recorded as an error step downstream


@pytest.mark.asyncio
async def test_backend_error_finishes_gracefully():
    policy = LLMReActPolicy(RaisingBackend())
    a = await policy.propose({}, [], [_compare_tool()])
    assert a.finish is True
    assert "backend error" in a.final["error"]


@pytest.mark.asyncio
async def test_prompt_includes_tools_task_and_format():
    backend = QueueBackend(['{"finish": true, "final": null}'])
    policy = LLMReActPolicy(backend, instructions="You are the Collector.")
    await policy.propose({"goal": "find EAT"}, [], [_compare_tool()])
    p = backend.prompts[0]
    assert "You are the Collector." in p
    assert "compare_values" in p
    assert "find EAT" in p
    assert "RESPONSE FORMAT" in p


@pytest.mark.asyncio
async def test_policy_drives_runtime_end_to_end():
    # The LLM (stubbed) calls compare_values, sees the observation, then finishes.
    backend = QueueBackend([
        '{"thought": "compare", "action": {"tool": "compare_values", "args": '
        '{"field_type": "eat_thickness", "review_value": "6.60", "source_value": "6.60", '
        '"review_unit": "mm", "source_unit": "mm"}}}',
        '{"thought": "match found", "finish": true, "final": {"label": "match"}}',
    ])
    agent = BoundedReActAgent("collector", LLMReActPolicy(backend), [_compare_tool()], max_steps=6)
    run = await agent.run({"goal": "audit one value"})

    assert run.status == "finished"
    assert run.final == {"label": "match"}
    assert run.n_steps == 1
    assert run.steps[0].tool == "compare_values"
    assert run.steps[0].observation["label"] == "match"
    # The second prompt fed the observation back to the model.
    assert "Observation:" in backend.prompts[1]
