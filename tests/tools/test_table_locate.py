"""table_locate_v1: unique table to locate, not a 20k excerpt. lean_v8 stays frozen."""
from __future__ import annotations

import hashlib
import json

import pytest

from react_review.contracts import repo_root
from react_review.schemas.table import CapturedTable
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
)
from react_review.tools.extraction_profile import TABLE_LOCATE_V1, prompt_version
from tests.tools.test_extraction_profile import _RecordingBackend, _pinned_payload, _rendered


def _ambiguous_mie_table() -> CapturedTable:
    """Unique table for age, but two MIE columns so B1 refuses the cell."""
    return CapturedTable(
        table_id="Tab1",
        caption="Table 1 Patient characteristics",
        header_rows=[["", "Before PSM / MIE", "After PSM / MIE"]],
        rows=[
            ["Age, years", "", ""],
            ["median (range)", "73(70-85)", "73(70-83)"],
        ],
    )


def _table_payload(**overrides) -> ExtractSourceValueInput:
    table = _ambiguous_mie_table()
    document = PaperDocument(
        paper_id="pin",
        reference=ReferenceEntry(title="pinned paper"),
        full_text="Age, years 73(70-85) 73(70-83)",
        tables=[table],
    )
    fixture = dict(
        document=document, field_type="age", group="mie", concept="age",
        concept_variants=["age", "age (years)"], raw_field_name="Age (years)",
        unit_hint="years", research_context="pinned context",
        cohort_display="MIE",
        cohorts={"mie": ["MIE"], "oe": ["OE"]},
        outcome="overall complications",
        extraction_profile="table_locate_v1",
        attempt=0)
    return ExtractSourceValueInput(**{**fixture, **overrides})


@pytest.mark.asyncio
async def test_no_tables_renders_the_same_bytes_as_lean_v8():
    """Empty tables keep the 20k excerpt path. The new template is not used."""
    v8 = await _rendered("lean_v8", outcome="overall complications")
    v9 = await _rendered("table_locate_v1", outcome="overall complications")
    assert v8 == v9
    assert "## SOURCE TABLE" not in v9


@pytest.mark.asyncio
async def test_lean_v8_does_not_switch_to_the_table_template():
    """targeted_v7 / lean_v8 stay frozen even when a unique table is present."""
    backend = _RecordingBackend()
    payload = _table_payload(extraction_profile="lean_v8")
    await ExtractSourceValueTool(backend).run(payload)
    assert backend.prompts
    assert "## SOURCE TABLE" not in backend.prompts[0]
    assert "Before PSM / MIE" not in backend.prompts[0] or "PAPER TEXT" in backend.prompts[0]


@pytest.mark.asyncio
async def test_unique_table_is_the_locate_input_not_a_20k_excerpt():
    backend = _RecordingBackend()
    await ExtractSourceValueTool(backend).run(_table_payload())
    assert len(backend.prompts) == 1
    prompt = backend.prompts[0]
    assert "## SOURCE TABLE" in prompt
    assert "Before PSM / MIE" in prompt
    assert "After PSM / MIE" in prompt
    assert "73(70-85)" in prompt
    assert len(prompt) < 8000
    assert "x" * 1000 not in prompt


@pytest.mark.asyncio
async def test_a_second_table_is_not_sent():
    other = CapturedTable(
        table_id="Tab2", caption="Table 2 Lab values",
        header_rows=[["", "Group A", "Group B"]],
        rows=[["BMI", "24.1", "26.0"]],
    )
    payload = _table_payload()
    payload = payload.model_copy(
        update={"document": payload.document.model_copy(
            update={"tables": [_ambiguous_mie_table(), other]})})
    backend = _RecordingBackend()
    await ExtractSourceValueTool(backend).run(payload)
    prompt = backend.prompts[0]
    assert "BMI" not in prompt
    assert "## SOURCE TABLE" in prompt


def test_table_locate_version_string():
    assert prompt_version("table_locate_v1") == TABLE_LOCATE_V1
    assert TABLE_LOCATE_V1 != prompt_version("lean_v8")


def test_table_locate_contract_pins_the_rendered_prompt():
    body = json.loads(
        (repo_root() / "configs/prompt_contracts/table_locate_v1.json"
         ).read_text(encoding="utf-8"))
    assert body["extraction_profile"] == "table_locate_v1"
    assert body["prompt_version"] == TABLE_LOCATE_V1
    assert body["rendered_prompt_sha256"] != "PENDING"


@pytest.mark.asyncio
async def test_table_locate_contract_hash_matches_what_the_tool_sends():
    body = json.loads(
        (repo_root() / "configs/prompt_contracts/table_locate_v1.json"
         ).read_text(encoding="utf-8"))
    backend = _RecordingBackend()
    await ExtractSourceValueTool(backend).run(_table_payload())
    digest = hashlib.sha256(backend.prompts[0].encode("utf-8")).hexdigest().upper()
    assert digest == body["rendered_prompt_sha256"]
