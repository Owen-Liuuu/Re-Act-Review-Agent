"""Tests for the built tool catalogue (mock mode) and the compare tool.

The mock catalogue wires the reused Mock* implementations, so every tool runs
end-to-end without network or LLM. The compare tool is additionally checked
against a real benchmark row.
"""
from __future__ import annotations

import pytest

from react_review.audit import ToleranceTable
from react_review.core.config import AppConfig
from react_review.core.enums import AuditLabel
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools import build_catalogue
from react_review.tools.base import ToolStage
from react_review.tools.compare import CompareValuesTool
from react_review.tools.models import (
    CompareInput,
    CountInput,
    ExtractInput,
)
from react_review.pipeline.schemas import EvidenceFieldSchema
from react_review.steps.data_extraction.schemas import PaperDocument


@pytest.fixture
def catalogue():
    return build_catalogue(AppConfig(mock_mode=True))


def test_catalogue_registers_expected_tools(catalogue):
    assert set(catalogue.names()) == {
        "search_pubmed",
        "count_pubmed",
        "count_europepmc",
        "count_openalex",
        "verify_reference",
        "fetch_fulltext",
        "extract_fields",
        "extract_source_value",
        "resolve_reference",
        "compare_values",
    }
    assert len(catalogue) == 10
    # Stage grouping is correct.
    assert len(catalogue.by_stage(ToolStage.SEARCH)) == 5   # + resolve_reference
    assert len(catalogue.by_stage(ToolStage.VERIFY)) == 1
    assert len(catalogue.by_stage(ToolStage.EXTRACT)) == 3
    assert len(catalogue.by_stage(ToolStage.COMPARE)) == 1


@pytest.mark.asyncio
async def test_compare_tool_matches_benchmark_case():
    tool = CompareValuesTool(ToleranceTable())
    # Keles: numbers equal but mm vs cm -> unit_mismatch (the real A022 case).
    out = await tool.run(CompareInput(
        field_type="eat_thickness",
        review_value="0.7 (0.6–0.9)", source_value="0.7 (0.6–0.9)",
        review_unit="mm", source_unit="cm",
    ))
    assert out.label == AuditLabel.UNIT_MISMATCH


@pytest.mark.asyncio
async def test_count_tool_runs(catalogue):
    out = await catalogue.get("count_pubmed").run(CountInput(query="diabetes"))
    assert out.database == "PubMed"
    assert out.count == 142


@pytest.mark.asyncio
async def test_verify_tool_runs(catalogue):
    out = await catalogue.get("verify_reference").run(
        ReferenceEntry(title="A paper", doi="10.1/x")
    )
    assert out.status.value == "verified"


@pytest.mark.asyncio
async def test_fetch_then_extract(catalogue):
    ref = ReferenceEntry(title="A paper", doi="10.1/x")
    fetched = await catalogue.get("fetch_fulltext").run(ref)
    assert fetched.retrieved is True
    assert fetched.document is not None

    schema = [EvidenceFieldSchema(student_field_name="sample_size",
                                  canonical_concept="sample_size", type="numeric")]
    table = await catalogue.get("extract_fields").run(
        ExtractInput(document=fetched.document, evidence_schema=schema,
                     research_context="test")
    )
    assert table.fields[0].field_name == "sample_size"


def test_real_mode_catalogue_builds():
    # Constructs the real impls (no calls made) — just ensures wiring imports OK.
    reg = build_catalogue(AppConfig(mock_mode=False))
    assert len(reg) == 10
    assert "fetch_fulltext" in reg
    assert "extract_source_value" in reg
    assert "resolve_reference" in reg


@pytest.mark.asyncio
async def test_default_catalogue_uses_shipped_tolerance():
    # No explicit tolerance -> should load configs/tolerances.yaml (1% / 3%),
    # not fall back to arbitrary defaults (R3 regression guard).
    reg = build_catalogue(AppConfig(mock_mode=True))
    out = await reg.get("compare_values").run(
        CompareInput(field_type="x", review_value="6.65", source_value="6.60")
    )
    assert out.tolerance_pct == 1.0
    assert out.sd_tolerance_pct == 3.0
    assert out.label == AuditLabel.MATCH
