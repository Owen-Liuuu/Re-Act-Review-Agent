"""Opt-in real-LLM test for one explicit TableCapturer prompt profile.

The live test is skipped unless ``RUN_LIVE_LLM=1`` is set.  When enabled it
uses the configured production backend, sends the exact prompt rendered by
``TableCapturer``, and preserves all inputs/outputs under ``output/live_tests``
for manual inspection.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest

from react_review.core.config import load_config
from react_review.hitl import StepReporter
from react_review.llm.base import LLMBackend
from react_review.parser.review_parser import _pdf_text
from react_review.parser.table_capture import TableCapturer
from react_review.parser.table_capture_contract import (
    PROMPT_TEMPLATES,
    TableCapturePromptContract,
    render_table_capture_prompt,
)
from react_review.parser.table_render import render_table_set
from react_review.pipeline.factory import _create_llm_backend


ROOT = Path(__file__).resolve().parents[2]
LIVE_ENABLED = os.getenv("RUN_LIVE_LLM", "").strip().lower() in {
    "1", "true", "yes", "on",
}


def _redact_url_query(value: str) -> str:
    """Keep a useful endpoint identity without persisting query credentials."""
    parts = urlsplit(str(value or ""))
    if not parts.scheme:
        return ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _explicit_live_profile() -> str:
    profile = os.getenv("TABLE_CAPTURE_PROMPT_PROFILE", "").strip()
    assert profile in PROMPT_TEMPLATES, (
        "set TABLE_CAPTURE_PROMPT_PROFILE explicitly to table_capture_v1 or "
        "table_capture_v2"
    )
    return profile


def _backend_metadata(inner: Any) -> dict[str, Any]:
    settings = getattr(inner, "_settings", None)
    base_url = getattr(inner, "_base_url", "") or getattr(settings, "base_url", "")
    return {
        "endpoint": _redact_url_query(base_url),
        "max_tokens": getattr(settings, "max_tokens", None),
        "temperature": getattr(settings, "temperature", None),
    }


class RecordingLiveBackend(LLMBackend):
    """Delegate to a real backend while retaining its exact I/O."""

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner
        self.prompts: list[str] = []
        self.responses: list[str] = []
        self.seeds: list[int] = []

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        self.seeds.append(seed)
        response = await self.inner.complete(prompt, seed=seed)
        self.responses.append(response)
        return response


def _normalise(text: str) -> str:
    value = str(text or "")

    # 只消除 PDF 行尾断词：
    # echocar-\ndiography → echocardiography
    value = re.sub(
        r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)",
        "",
        value,
    )

    return re.sub(r"\s+", " ", value).strip().casefold()


def _contains_term(text: str, term: str) -> bool:
    """Whole-token containment so EAT does not match inside treatment."""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _unanchored_cells(table_set: Any, review_text: str) -> list[dict[str, Any]]:
    """Cells the model returned that are not present in the PDF text layer."""
    source = _normalise(review_text)
    missing: list[dict[str, Any]] = []
    for table in table_set.tables:
        for section, rows in (("header", table.header_rows), ("row", table.rows)):
            for row_index, row in enumerate(rows):
                for column_index, cell in enumerate(row):
                    value = str(cell or "").strip()
                    if value and _normalise(value) not in source:
                        missing.append({
                            "table_id": table.table_id,
                            "section": section,
                            "row": row_index,
                            "column": column_index,
                            "value": value,
                        })
        for label in table.cohort_labels_seen:
            if label and _normalise(label) not in source:
                missing.append({
                    "table_id": table.table_id,
                    "section": "cohort_labels_seen",
                    "value": label,
                })
    return missing


def _write_artifacts(
    *, pdf: Path, config: Path, model_id: str, review_text: str,
    prompt: str, response: str, context: str, table_set: Any,
    unanchored: list[dict[str, Any]], prompt_profile: str,
    backend_metadata: dict[str, Any],
) -> Path:
    base = Path(os.getenv(
        "TABLE_CAPTURE_ARTIFACT_DIR",
        str(ROOT / "output" / "live_tests" / "table_capture"),
    ))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = base / prompt_profile / f"{pdf.stem}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "review_text.txt").write_text(review_text, encoding="utf-8")
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "raw_response.txt").write_text(response, encoding="utf-8")
    (run_dir / "rendered_tables.txt").write_text(
        render_table_set(table_set.tables), encoding="utf-8")
    (run_dir / "capture.json").write_text(json.dumps({
        "research_context": context,
        **table_set.model_dump(mode="json"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "diagnostics.json").write_text(json.dumps({
        "unanchored_cells": unanchored,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    contract = TableCapturePromptContract.load(prompt_profile)
    (run_dir / "metadata.json").write_text(json.dumps({
        "pdf": str(pdf.resolve()),
        "config_file": config.name,
        "model_id": model_id,
        "prompt_id": contract.prompt_id,
        "prompt_version": contract.prompt_version,
        "prompt_contract_sha256": contract.rendered_prompt_sha256,
        "review_text_chars": len(review_text),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "seed": 42,
        **backend_metadata,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def test_v2_candidate_has_no_frozen_domain_example_while_v1_stays_frozen() -> None:
    """The experiment compares a faithful baseline with one neutral candidate."""
    forbidden = (
        "T1DM", "EAT", "Ahmad 2022", "6.60 ± 0.71", "epicardial adipose",
    )
    for term in forbidden:
        assert not _contains_term(PROMPT_TEMPLATES["table_capture_v2"], term), (
            f"TableCapture v2 contains frozen example {term!r}"
        )
    assert _contains_term(PROMPT_TEMPLATES["table_capture_v1"], "T1DM")


def test_live_artifact_url_redaction_drops_query_parameters() -> None:
    assert _redact_url_query(
        "https://provider.example/v1?api_key=secret&trace=yes"
    ) == "https://provider.example/v1"


@pytest.mark.skipif(
    not LIVE_ENABLED,
    reason="set RUN_LIVE_LLM=1 to call the configured real LLM",
)
@pytest.mark.asyncio
async def test_live_llm_uses_the_explicit_table_capture_prompt() -> None:
    """Call the real model through the actual, explicitly selected unit."""
    prompt_profile = _explicit_live_profile()
    pdf = Path(os.getenv("TABLE_CAPTURE_PDF", str(ROOT / "SRMA.pdf")))
    config_path = Path(os.getenv(
        "TABLE_CAPTURE_CONFIG", str(ROOT / "configs" / "config.local.yaml")))
    assert pdf.is_file(), f"TABLE_CAPTURE_PDF does not exist: {pdf}"
    assert config_path.is_file(), f"TABLE_CAPTURE_CONFIG does not exist: {config_path}"

    # Match ReviewParser exactly: PyMuPDF text, first 50,000 characters.
    review_text = _pdf_text(pdf)[:50000]
    inner = _create_llm_backend(load_config(config_path))
    backend = RecordingLiveBackend(inner)

    table_set, context = await TableCapturer(
        backend, prompt_profile=prompt_profile).capture(
        review_text,
        reporter=StepReporter(),
        pdf_path=str(pdf.resolve()),
    )

    assert len(backend.prompts) == 1, "TableCapturer unexpectedly called the LLM more than once"
    assert len(backend.responses) == 1
    assert backend.seeds == [42]

    actual_prompt = backend.prompts[0]
    raw_response = backend.responses[0]
    expected_prompt = render_table_capture_prompt(prompt_profile, text=review_text)
    unanchored = _unanchored_cells(table_set, review_text)

    run_dir = _write_artifacts(
        pdf=pdf,
        config=config_path,
        model_id=backend.model_id,
        review_text=review_text,
        prompt=actual_prompt,
        response=raw_response,
        context=context,
        table_set=table_set,
        unanchored=unanchored,
        prompt_profile=prompt_profile,
        backend_metadata=_backend_metadata(inner),
    )
    print(f"\n[live table capture] profile={prompt_profile} model={backend.model_id}")
    print(f"[live table capture] tables={len(table_set.tables)}")
    print(f"[live table capture] artifacts={run_dir.resolve()}")

    # This is the central contract: the test observes the exact production
    # prompt, rather than duplicating or approximating it in a test fixture.
    assert actual_prompt == expected_prompt
    assert table_set.tables, "the real LLM returned no parseable data table"

    placeholders = (
        "<string or empty string>", "<exact cell text>",
        "<exact printed header>", "<specific transcription uncertainty>",
    )
    for placeholder in placeholders:
        assert placeholder not in raw_response, (
            f"the model copied output-schema placeholder {placeholder!r}"
        )

    # The neutral candidate may not introduce the baseline's example terms when
    # they are absent. V1 is intentionally frozen and scored rather than edited.
    frozen_terms = (
        "T1DM", "EAT", "Ahmad 2022", "6.60 ± 0.71", "epicardial adipose",
    )
    for term in frozen_terms:
        if prompt_profile == "table_capture_v2" and not _contains_term(review_text, term):
            assert not _contains_term(raw_response, term), (
                f"the model introduced absent frozen-domain term {term!r}; "
                f"inspect {run_dir / 'raw_response.txt'}"
            )

    # The prompt requires rectangular transcription and verbatim cells.
    for table in table_set.tables:
        width = max((len(row) for row in table.header_rows), default=0)
        assert width > 0, f"{table.table_id} has no header cells"
        assert all(len(row) == width for row in table.header_rows), (
            f"{table.table_id} has ragged header rows")
        assert all(len(row) == width for row in table.rows), (
            f"{table.table_id} has rows that do not match header width {width}")

    assert not unanchored, (
        f"the LLM returned {len(unanchored)} cell/cohort value(s) not found "
        f"verbatim in the PDF text; inspect {run_dir / 'diagnostics.json'}"
    )
