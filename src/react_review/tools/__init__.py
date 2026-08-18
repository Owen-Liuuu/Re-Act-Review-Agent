"""Typed tool catalogue (Search / Verify / Extract / Compare).

Each tool wraps a reused step implementation behind a validated input/output
contract; :class:`ToolRegistry` holds them and supports ``subset`` for bounded
agent exposure. Build the default catalogue with :func:`build_catalogue`.
"""
from react_review.tools.base import Tool, ToolStage
from react_review.tools.registry import ToolRegistry
from react_review.tools.catalogue import build_catalogue
from react_review.tools.compare import CompareValuesTool
from react_review.tools.search import CountResultsTool, SearchPubMedTool
from react_review.tools.verify import VerifyReferenceTool
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.models import (
    CompareInput,
    CountInput,
    CountResult,
    FetchResult,
)

__all__ = [
    "Tool",
    "ToolStage",
    "ToolRegistry",
    "build_catalogue",
    "CompareValuesTool",
    "SearchPubMedTool",
    "CountResultsTool",
    "VerifyReferenceTool",
    "FetchFullTextTool",
    "CompareInput",
    "CountInput",
    "CountResult",
    "FetchResult",
]
