"""Detect prompt-skeleton descriptions echoed back as model output.

JSON examples in prompts must use ``{placeholders}``, ``<angle descriptions>``,
or real literals. A bare English phrase in a value slot is what models copy
into ``page_hint`` and similar unvalidated fields.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable

import structlog

logger = structlog.get_logger(__name__)

_KV = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"((?:\\.|[^"\\])*)"')
_PLACEHOLDER = re.compile(r"^\{[A-Za-z_][A-Za-z0-9_]*\}$")
_IDENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")

_PROMPT_MODULES = (
    "react_review.parser.review_extraction.prompts",
    "react_review.parser.table_capture_contract",
    "react_review.parser.review_parser",
    "react_review.tools.forest_ocr",
    "react_review.tools.extract_source",
    "react_review.tools.batch_prompt",
    "react_review.tools.semantic_compare",
    "react_review.dkb.agent",
    "react_review.agents.llm_policy",
)


def is_allowed_skeleton_value(value: str) -> bool:
    """True when a JSON-example string is a placeholder, angle-desc, or literal."""
    text = (value or "").strip()
    if not text:
        return True
    if _PLACEHOLDER.fullmatch(text):
        return True
    if len(text) >= 2 and text.startswith("<") and text.endswith(">"):
        return True
    if _NUMBER.fullmatch(text):
        return True
    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if parts and all(_IDENT.fullmatch(p) for p in parts):
            return True
    return bool(_IDENT.fullmatch(text))


def skeleton_string_values(template: str) -> list[tuple[str, str]]:
    """``(key, value)`` pairs from JSON-example strings in a prompt template."""
    body = (template or "").replace("{{", "{").replace("}}", "}")
    out: list[tuple[str, str]] = []
    for match in _KV.finditer(body):
        key, raw = match.group(1), match.group(2)
        try:
            value = ast.literal_eval(f'"{raw}"')
        except (SyntaxError, ValueError):
            value = raw.replace(r"\"", '"')
        out.append((key, value))
    return out


def collect_prompt_templates() -> list[tuple[str, str]]:
    """Named prompt-string constants across extraction / capture / tools."""
    import importlib

    found: list[tuple[str, str]] = []
    seen: set[int] = set()
    for mod_name in _PROMPT_MODULES:
        module = importlib.import_module(mod_name)
        for name, text in vars(module).items():
            if not isinstance(text, str) or "\n" not in text:
                continue
            if id(text) in seen:
                continue
            if not skeleton_string_values(text):
                continue
            seen.add(id(text))
            found.append((f"{mod_name}.{name}", text))
        templates = getattr(module, "PROMPT_TEMPLATES", None)
        if isinstance(templates, dict):
            for name, text in templates.items():
                if not isinstance(text, str) or id(text) in seen:
                    continue
                if not skeleton_string_values(text):
                    continue
                seen.add(id(text))
                found.append((f"{mod_name}.PROMPT_TEMPLATES[{name}]", text))
    return found


def bare_skeleton_hits(
    templates: Iterable[tuple[str, str]] | None = None,
) -> list[tuple[str, str, str]]:
    """``(source, key, value)`` for JSON-example strings that are bare descriptions."""
    hits: list[tuple[str, str, str]] = []
    for source, text in (templates if templates is not None else collect_prompt_templates()):
        for key, value in skeleton_string_values(text):
            if not is_allowed_skeleton_value(value):
                hits.append((source, key, value))
    return hits


@lru_cache(maxsize=1)
def bare_description_index() -> dict[str, frozenset[str]]:
    """JSON key → instructional phrases that must not appear as model output."""
    grouped: dict[str, set[str]] = defaultdict(set)
    for _source, key, value in bare_skeleton_hits():
        grouped[key].add(value)
    return {key: frozenset(values) for key, values in grouped.items()}


# Few-shot tokens from frozen v1/v3 / forest_ocr_v1 examples. A transcription
# that emits one of these when the source material does not contain it has
# copied the prompt, not the paper.
EXAMPLE_LEAK_TOKENS: tuple[str, ...] = (
    "Ahmad",
    "Egypt",
    "T1DM",
    "Li J 2015",
    "6.60",
)


def example_leak_tokens(output: str, source_text: str) -> list[str]:
    """Example tokens present in ``output`` but absent from the source material."""
    blob = output or ""
    source = source_text or ""
    return [token for token in EXAMPLE_LEAK_TOKENS if token in blob and token not in source]


def warn_echoed_placeholders(payload: Any, *, model_id: str = "") -> None:
    """Log when a parsed value equals the prompt's instructional phrase for that key."""
    index = bare_description_index()
    if not index:
        return

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value in index.get(str(key), ()):
                    logger.warning(
                        "prompt_placeholder_echoed",
                        key=str(key),
                        value=value[:80],
                        model=model_id,
                    )
                else:
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(payload)
