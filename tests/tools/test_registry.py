"""Tests for the tool contract + registry (deterministic, no network)."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from react_review.tools.base import Tool, ToolStage
from react_review.tools.registry import ToolRegistry


class _Out(BaseModel):
    ok: bool = True


class _In(BaseModel):
    x: int = 0


class _DummyTool(Tool):
    stage = ToolStage.COMPARE
    input_model = _In
    output_model = _Out

    def __init__(self, name: str) -> None:
        self.name = name

    async def run(self, payload: _In) -> _Out:
        return _Out(ok=payload.x > 0)


def test_register_and_get():
    reg = ToolRegistry()
    reg.register(_DummyTool("a"))
    assert "a" in reg
    assert reg.get("a").name == "a"
    assert len(reg) == 1


def test_duplicate_name_rejected():
    reg = ToolRegistry()
    reg.register(_DummyTool("a"))
    with pytest.raises(ValueError, match="duplicate"):
        reg.register(_DummyTool("a"))


def test_missing_name_rejected():
    reg = ToolRegistry()
    t = _DummyTool("")
    with pytest.raises(ValueError, match="no name"):
        reg.register(t)


def test_by_stage_and_subset():
    reg = ToolRegistry()
    reg.register(_DummyTool("a"))
    reg.register(_DummyTool("b"))
    assert reg.names() == ["a", "b"]
    assert {t.name for t in reg.by_stage(ToolStage.COMPARE)} == {"a", "b"}
    assert reg.by_stage(ToolStage.SEARCH) == []
    subset = reg.subset(["b"])
    assert [t.name for t in subset] == ["b"]


@pytest.mark.asyncio
async def test_tool_run_and_describe():
    t = _DummyTool("a")
    assert (await t.run(_In(x=5))).ok is True
    assert (await t.run(_In(x=0))).ok is False
    d = t.describe()
    assert d["name"] == "a" and d["stage"] == "compare"
    assert d["input"] == "_In" and d["output"] == "_Out"
