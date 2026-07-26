"""The typed tool contract.

Every tool declares its stage and its Pydantic input/output models, and exposes
a single async ``run``. Tools wrap the reused step implementations behind a
uniform, validated interface so the deterministic orchestrator (and, in P2, the
bounded ReAct agents) can call them the same way and only ever see typed I/O.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

from pydantic import BaseModel


class ToolStage(str, Enum):
    """The pipeline stage a tool belongs to (Proposal: Search/Verify/Extract/Compare)."""

    SEARCH = "search"
    VERIFY = "verify"
    EXTRACT = "extract"
    COMPARE = "compare"


class Tool(ABC):
    """Base class for a typed, single-purpose tool.

    Subclasses set ``name``, ``stage``, ``input_model``, ``output_model`` and
    implement ``run``. ``name`` may be set per-instance (e.g. the count tool is
    registered once per database).
    """

    name: str = ""
    stage: ClassVar[ToolStage]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, payload: BaseModel) -> BaseModel:
        """Execute the tool. ``payload`` is an ``input_model`` instance."""

    def describe(self) -> dict[str, str]:
        """A small, serialisable description (name / stage / I/O model names)."""
        return {
            "name": self.name,
            "stage": self.stage.value,
            "input": self.input_model.__name__,
            "output": self.output_model.__name__,
        }
