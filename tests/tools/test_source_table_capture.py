"""source_table_capture_v1: sealed like SourceQuery. No review_* / value slots."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from react_review.contracts import repo_root
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extraction_cache import ExtractionCache
from react_review.tools.source_table_capture import (
    PROMPT_ID,
    PROMPT_VERSION,
    SourceTableCapturer,
    SourceTableQuery,
    render_source_table_capture_prompt,
    sha256_rendered_prompt,
)
from react_review.audit.semantic_cache import SemanticCache
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry


_REF = ReferenceEntry(title="t", doi="10.1000/a2")


def test_envelope_forbids_review_and_value_slots():
    fields = set(SourceTableQuery.model_fields)
    assert fields == {"text"}
    for name in ("review_value", "review_item", "value", "selected", "research_context"):
        assert name not in fields
    with pytest.raises(ValidationError):
        SourceTableQuery(text="x", review_value="6.60")
    with pytest.raises(ValidationError):
        SourceTableQuery(text="x", value="6.60")


def test_rendered_prompt_only_interpolates_paper_text():
    prompt = render_source_table_capture_prompt(SourceTableQuery(text="CELL 73"))
    assert "CELL 73" in prompt
    assert "## SOURCE PAPER TEXT" in prompt
    assert "review_value" not in prompt
    assert "REVIEW TEXT" not in prompt
    assert "SELECTED DISPLAYS" not in prompt


def test_contract_file_pins_the_rendered_prompt():
    body = json.loads(
        (repo_root() / "configs/prompt_contracts/source_table_capture_v1.json"
         ).read_text(encoding="utf-8"))
    assert body["prompt_id"] == PROMPT_ID
    assert body["prompt_version"] == PROMPT_VERSION
    rendered = render_source_table_capture_prompt(
        SourceTableQuery(text=body["fixture_inputs"]["text"]),
    )
    assert sha256_rendered_prompt(rendered) == body["rendered_prompt_sha256"]


class _RecordingBackend:
    model_id = "scripted"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return json.dumps({
            "tables": [{
                "table_id": "table_1",
                "caption": "Table 1",
                "header_rows": [["Age", "Total"]],
                "rows": [["years", "73"]],
            }],
        })


@pytest.mark.asyncio
async def test_capturer_returns_grids_and_never_asks_for_a_review_value():
    backend = _RecordingBackend()
    tables = await SourceTableCapturer(backend).capture("Age Total 73")
    assert len(tables) == 1
    assert tables[0].rows[0][1] == "73"
    assert "## SOURCE PAPER TEXT" in backend.prompts[0]
    assert "REVIEW TEXT" not in backend.prompts[0]
    assert "review_value" not in backend.prompts[0]


class _LocalRetriever(PaperRetriever):
    def __init__(self, document, tables=None) -> None:
        self.document = document
        self.captured_tables = list(tables or [])

    async def retrieve(self, reference: ReferenceEntry):
        return self.document


@pytest.mark.asyncio
async def test_fetch_falls_back_to_the_source_profile_when_pdf_tables_are_empty(
    tmp_path,
):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    document = PaperDocument(
        paper_id="p", reference=_REF, full_text="Age Total 73",
        document_scope=DocumentScope.FULL_TEXT,
        metadata={"source": "local_pdf", "path": str(pdf)},
    )
    backend = _RecordingBackend()
    result = await FetchFullTextTool(
        _LocalRetriever(document),
        source_tables=SourceTableCapturer(backend),
    ).run(_REF)
    assert result.tables
    assert result.document.tables[0].header_rows[0][0] == "Age"
    assert "## SOURCE PAPER TEXT" in backend.prompts[0]


@pytest.mark.asyncio
async def test_fetch_does_not_call_the_model_when_the_document_is_not_an_upload():
    document = PaperDocument(
        paper_id="p", reference=_REF, full_text="pmc text",
        document_scope=DocumentScope.FULL_TEXT,
        metadata={"source": "pmc", "pmc_id": "PMC1"},
    )
    backend = _RecordingBackend()
    result = await FetchFullTextTool(
        _LocalRetriever(document),
        source_tables=SourceTableCapturer(backend),
    ).run(_REF)
    assert result.tables == []
    assert backend.prompts == []


def test_extraction_cache_marked_private_is_not_shareable(tmp_path):
    cache = ExtractionCache(tmp_path / "extraction_cache.json")
    cache.mark_private()
    cache.put("k", {"ok": True}, model_id="m")
    body = json.loads((tmp_path / "extraction_cache.json").read_text(encoding="utf-8"))
    assert body["shareable"] is False
    assert "never redistribute" in body["shareable_reason"]


def test_semantic_cache_marked_private_is_not_shareable(tmp_path):
    cache = SemanticCache(tmp_path / "semantic_cache.json")
    cache.mark_private()
    cache.save()
    body = json.loads((tmp_path / "semantic_cache.json").read_text(encoding="utf-8"))
    assert body["shareable"] is False
