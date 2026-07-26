"""The tool registry: register, look up, filter by stage, and take subsets.

``subset`` is what a P2 bounded ReAct agent uses to expose only the 3-5 tools
its stage needs (AgentBench: tool-selection accuracy degrades as the action
space grows), while the registry as a whole is the typed catalogue.
"""
from __future__ import annotations

from react_review.tools.base import Tool, ToolStage


class ToolRegistry:
    """A name-keyed collection of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        """Register a tool; raises on a missing or duplicate name."""
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} has no name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        """Return the tool registered under ``name`` (KeyError if absent)."""
        return self._tools[name]

    def names(self) -> list[str]:
        """All registered tool names, sorted."""
        return sorted(self._tools)

    def by_stage(self, stage: ToolStage) -> list[Tool]:
        """All tools in a given stage, in registration order."""
        return [t for t in self._tools.values() if t.stage == stage]

    def subset(self, names: list[str]) -> list[Tool]:
        """Return the named tools (for bounded agent exposure). KeyError if any missing."""
        return [self._tools[n] for n in names]

    def describe(self) -> list[dict[str, str]]:
        """A serialisable catalogue listing."""
        return [t.describe() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
