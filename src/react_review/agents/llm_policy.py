"""An LLM-driven ReAct policy for the bounded runtime.

Prompts the LLM in Thought/Action/Observation form: it sees the task, the small
exposed tool subset (name + arg fields + one-line purpose), and the trajectory
so far, then returns the next step as a strict JSON object — either an action
(a tool call) or a finish (with the agent's final output).

The policy never raises: a malformed / unparseable response becomes a recorded
error step (the runtime feeds it back next turn) so the agent can self-correct,
bounded by ``max_steps``.
"""
from __future__ import annotations

import json
from typing import Any

import structlog

from react_review.agents.runtime import AgentPolicy, ProposedAction
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.schemas.agent import StepRecord
from react_review.tools.base import Tool

logger = structlog.get_logger(__name__)

_RESPONSE_SPEC = """Respond with ONE JSON object and nothing else, in exactly one of these two shapes:

To call a tool:
{"thought": "why this step", "action": {"tool": "<tool_name>", "args": {<arg>: <value>, ...}}}

To finish (when you have the answer):
{"thought": "why you are done", "finish": true, "final": <your result as JSON>}
"""


def _tool_signature(tool: Tool) -> str:
    """One line: name(arg: type, ...) — first line of the tool's docstring."""
    fields = getattr(tool.input_model, "model_fields", {})
    params = ", ".join(
        f"{name}: {getattr(f.annotation, '__name__', str(f.annotation))}"
        for name, f in fields.items()
    )
    doc = (type(tool).__doc__ or "").strip().splitlines()
    purpose = doc[0].strip() if doc else ""
    return f"- {tool.name}({params}) — {purpose}"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


class LLMReActPolicy(AgentPolicy):
    """Drives the runtime by asking an LLM for the next ReAct step."""

    def __init__(
        self,
        backend: LLMBackend,
        instructions: str = "",
        *,
        max_obs_chars: int = 800,
    ) -> None:
        self._backend = backend
        self._instructions = instructions.strip()
        self._max_obs = max_obs_chars

    async def propose(
        self,
        task: dict[str, Any],
        steps: list[StepRecord],
        tools: list[Tool],
    ) -> ProposedAction:
        prompt = self._build_prompt(task, steps, tools)
        try:
            raw = await self._backend.complete(prompt)
        except Exception as exc:  # network/API failure — end gracefully, no crash
            logger.warning("llm_policy_backend_error", error=str(exc))
            return ProposedAction(
                thought="backend error", finish=True,
                final={"error": f"backend error: {exc}"},
            )
        return self._parse(raw)

    # ------------------------------------------------------------------
    def _build_prompt(
        self, task: dict[str, Any], steps: list[StepRecord], tools: list[Tool]
    ) -> str:
        parts: list[str] = []
        if self._instructions:
            parts.append(self._instructions)
        parts.append("## TASK\n" + json.dumps(task, ensure_ascii=False))
        parts.append(
            "## AVAILABLE TOOLS (you may ONLY use these)\n"
            + "\n".join(_tool_signature(t) for t in tools)
        )
        if steps:
            parts.append("## TRAJECTORY SO FAR\n" + self._render_steps(steps))
        parts.append("## RESPONSE FORMAT\n" + _RESPONSE_SPEC)
        return "\n\n".join(parts)

    def _render_steps(self, steps: list[StepRecord]) -> str:
        lines: list[str] = []
        for s in steps:
            if s.thought:
                lines.append(f"Thought: {s.thought}")
            lines.append(f"Action: {s.tool}({json.dumps(s.args, ensure_ascii=False)})")
            if s.error:
                lines.append(f"Observation (error): {_truncate(s.error, self._max_obs)}")
            else:
                obs = json.dumps(s.observation, ensure_ascii=False, default=str)
                lines.append(f"Observation: {_truncate(obs, self._max_obs)}")
        return "\n".join(lines)

    def _parse(self, raw: str) -> ProposedAction:
        try:
            data = parse_llm_response(raw, self._backend.model_id)
        except Exception:
            logger.warning("llm_policy_unparseable", raw=_truncate(raw, 200))
            # Recorded as an error step (tool name is unknown) so the model can
            # correct next turn; bounded by the runtime's max_steps.
            return ProposedAction(
                thought="previous response was not valid JSON; retrying",
                tool="__unparseable__",
            )

        thought = str(data.get("thought", ""))
        if data.get("finish"):
            return ProposedAction(thought=thought, finish=True, final=data.get("final"))

        action = data.get("action")
        if not isinstance(action, dict):
            action = data  # tolerate a flat {"tool":..., "args":...}
        tool = str(action.get("tool", "")).strip()
        args = action.get("args")
        if not isinstance(args, dict):
            args = {}
        return ProposedAction(thought=thought, tool=tool, args=args)
