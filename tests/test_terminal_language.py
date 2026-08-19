"""Terminal-facing modules must not ship CJK string literals.

Comments and docstrings are exempt. The scanner is the regression lock: a
Chinese spec that is copied into user-visible output will fail this test.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from react_review.contracts import repo_root

_CJK = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
_SRC = repo_root() / "src" / "react_review"
_SCAN = [
    _SRC / "hitl",
    _SRC / "orchestrator" / "audit_pipeline.py",
    _SRC / "parser" / "review_parser.py",
    _SRC / "parser" / "review_extraction" / "pipeline.py",
    _SRC / "tools" / "search" / "reconciler.py",
]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    found: set[int] = set()

    def mark(node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            found.add(id(first.value))

    mark(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            mark(node)
    return found


def cjk_string_literals(source: str) -> list[tuple[int, str]]:
    """Line number + value for non-docstring string literals that contain CJK."""
    tree = ast.parse(source)
    skip = _docstring_nodes(tree)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        if _CJK.search(node.value):
            hits.append((getattr(node, "lineno", 0), node.value))
    return hits


def _python_files() -> list[Path]:
    files: list[Path] = []
    for path in _SCAN:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            files.append(path)
    return files


def test_cjk_literal_scanner_flags_an_inserted_chinese_string():
    """Tamper check: a Chinese literal in scanned source must fail."""
    assert cjk_string_literals('msg = "hello"\n') == []
    assert cjk_string_literals('"""模块说明"""\nmsg = "ok"\n') == []
    hits = cjk_string_literals('msg = "检索失败"\n')
    assert hits and "检索失败" in hits[0][1]


def test_terminal_modules_have_no_cjk_string_literals():
    failures: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        hits = cjk_string_literals(source)
        for lineno, value in hits:
            preview = value.replace("\n", " ")[:80]
            failures.append(f"{path.relative_to(repo_root())}:{lineno}: {preview!r}")
    assert not failures, "CJK in terminal-facing string literals:\n" + "\n".join(failures)
