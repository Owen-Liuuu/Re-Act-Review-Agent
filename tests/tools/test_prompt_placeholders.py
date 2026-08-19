"""JSON-example value slots must not grow new bare instructional phrases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.core.exceptions import LLMError
from react_review.llm.base import parse_llm_response
from react_review.llm.prompt_placeholders import (
    bare_skeleton_hits,
    example_leak_tokens,
    is_allowed_skeleton_value,
)

EXEMPTIONS = Path(__file__).with_name("prompt_placeholder_exemptions.json")


def _row(source: str, key: str, value: str) -> tuple[str, str, str]:
    return (source, key, value)


def test_allowed_skeleton_forms():
    assert is_allowed_skeleton_value("{caption}")
    assert is_allowed_skeleton_value(
        "<printed page number, digits only, or empty string>")
    assert is_allowed_skeleton_value("table_1")
    assert is_allowed_skeleton_value("0.0")
    assert is_allowed_skeleton_value("pdf_table | forest_plot | other")
    assert is_allowed_skeleton_value("same|review_broader|review_narrower")
    assert not is_allowed_skeleton_value("printed page or empty")
    assert not is_allowed_skeleton_value(
        "one row per study; the cohort split is a column pair")


def test_bare_skeleton_hits_match_the_exemption_list():
    actual = set(bare_skeleton_hits())
    frozen = {
        _row(row["source"], row["key"], row["value"])
        for row in json.loads(EXEMPTIONS.read_text(encoding="utf-8"))
    }
    new = actual - frozen
    stale = frozen - actual
    assert not new, (
        "new bare JSON-example phrases (use {placeholder}, <angle desc>, "
        "or a real literal; or add an exemption if this is frozen prompt debt):\n"
        + "\n".join(sorted(f"{s} {k}={v!r}" for s, k, v in new))
    )
    assert not stale, (
        "stale exemptions (the phrase is gone — remove it from the list):\n"
        + "\n".join(sorted(f"{s} {k}={v!r}" for s, k, v in stale))
    )


def test_parse_llm_response_warns_when_page_hint_echoes_skeleton(monkeypatch):
    seen: list[tuple[str, dict]] = []

    def _capture(event: str, **kwargs) -> None:
        seen.append((event, kwargs))

    monkeypatch.setattr(
        "react_review.llm.prompt_placeholders.logger.warning", _capture)
    parse_llm_response('{"page_hint": "printed page or empty"}', "stub")
    echoes = [kw for event, kw in seen if event == "prompt_placeholder_echoed"]
    assert echoes and echoes[0]["key"] == "page_hint"
    assert echoes[0]["model"] == "stub"


def test_numeric_page_hint_is_not_an_echo(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        "react_review.llm.prompt_placeholders.logger.warning",
        lambda event, **kwargs: seen.append(event),
    )
    parse_llm_response('{"page_hint": "10"}', "stub")
    assert "prompt_placeholder_echoed" not in seen


def test_example_leak_tokens_require_absence_from_source():
    assert example_leak_tokens('{"rows":[["Ahmad 2022","Egypt"]]}', "unrelated") == [
        "Ahmad", "Egypt"]
    assert example_leak_tokens(
        '{"rows":[["Ahmad 2022","Egypt"]]}', "Ahmad 2022 Egypt T1DM") == []


def test_copied_few_shot_row_is_rejected():
    with pytest.raises(LLMError, match="prompt-example token"):
        parse_llm_response(
            '{"rows":[["Li J 2015","23","58","32","54"]]}',
            "stub",
            source_text="Capovilla 2023 19 58",
        )


def test_tokens_that_are_in_the_source_are_not_rejected():
    data = parse_llm_response(
        '{"rows":[["Li J 2015","23","58"]]}',
        "stub",
        source_text="Li J 2015 23 58 32 54 Total (Wald)",
    )
    assert data["rows"][0][0] == "Li J 2015"
