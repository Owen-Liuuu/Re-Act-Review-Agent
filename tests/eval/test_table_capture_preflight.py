"""The A/B gate refuses stale contracts before any paid call can start."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.table_capture_preflight import PreflightError, preflight


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval" / "table_capture_ab_v1.json"


def _changed_manifest(tmp_path: Path, **changes) -> Path:
    body = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body.update(changes)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_public_preflight_pins_two_documents_two_prompts_and_four_calls():
    report = preflight(MANIFEST)
    assert report["ready"] is True
    assert report["planned_live_calls"] == 4
    assert report["document_count"] == 2
    assert report["prompt_count"] == 2
    assert report["gold_counts"] == {"tables": 2, "rows": 25, "cells": 201}


def test_stale_prompt_hash_is_refused_before_live(tmp_path):
    body = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body["prompt_profiles"][1]["rendered_prompt_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(PreflightError, match="prompt hash"):
        preflight(path)


def test_call_count_must_equal_documents_times_prompts(tmp_path):
    with pytest.raises(PreflightError, match="planned_live_calls"):
        preflight(_changed_manifest(tmp_path, planned_live_calls=3))


@pytest.mark.skipif(
    not (ROOT / "SRMA.pdf").is_file()
    or not (ROOT / "output/gold/table_capture_v1/eat_t1dm_review_2025.jsonl").is_file(),
    reason="private PDF/gold are intentionally absent from a public clone",
)
def test_private_preflight_verifies_the_current_pdf_and_gold_hashes():
    report = preflight(MANIFEST, require_private=True)
    assert report["private_inputs_verified"] is True
    assert len(report["documents"]) == 2
