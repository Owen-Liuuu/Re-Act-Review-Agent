"""Offline checks for frozen-benchmark gates and stratified diagnostics."""
from __future__ import annotations

import hashlib
import json

from react_review.eval_accuracy import RowResult
from react_review.eval_benchmark import benchmark_diagnostics, validate_frozen_benchmark


def _row(audit_id: str, expected: str, predicted: str, *, mode: str,
         observed_mode: str | None = None, gap: str = "") -> RowResult:
    return RowResult(
        study_id="study", group="group", field_type="field",
        expected_label=expected, predicted_label=predicted,
        expected_source="source", extracted_source="source", found=True,
        outcome="found", extraction_correct=True, audit_id=audit_id,
        expected_match_mode=mode, match_mode=observed_mode or mode,
        known_gap=gap,
    )


def test_diagnostics_keep_declared_gaps_in_raw_failures():
    rows = [
        _row("S1", "match", "match", mode="semantic"),
        _row("K1", "mismatch", "match", mode="structured",
             gap="confidence_level_not_modeled"),
        _row("U1", "match", "not_comparable", mode="numeric"),
    ]

    result = benchmark_diagnostics(rows)

    assert result["status"] == "fail_unexpected_differences"
    assert result["differences"]["total"] == 2
    assert result["differences"]["declared_known_gap_count"] == 1
    assert result["differences"]["unexpected_count"] == 1
    assert result["silent_releases"]["total"] == 1
    assert result["silent_releases"]["declared_known_gap_count"] == 1
    assert result["silent_releases"]["unexpected_count"] == 0
    assert result["semantic"]["escalation_gate_passed"] is True


def test_frozen_gate_checks_hashes_contract_and_semantic_rows(tmp_path):
    review = tmp_path / "review.pdf"
    source = tmp_path / "source.pdf"
    review.write_bytes(b"review")
    source.write_bytes(b"source")
    (tmp_path / "contract.csv").write_text("x\n", encoding="utf-8")
    preflight = {"capture": {"frozen_whitelist": ["table_1"]}}
    (tmp_path / "preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
    manifest = {
        "benchmark_id": "fixture",
        "review": {"path": "review.pdf",
                   "sha256": hashlib.sha256(b"review").hexdigest()},
        "selected_source": {"path": "source.pdf",
                            "sha256": hashlib.sha256(b"source").hexdigest()},
        "table_whitelist": ["table_1"],
        "contract_files": ["contract.csv", "preflight.json"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [{"audit_id": "S1", "expected_match_mode": "semantic",
             "known_gap": ""}]

    result = validate_frozen_benchmark(tmp_path, rows)

    assert result["status"] == "pass"
    assert result["checks"]["review_sha256_match"] is True
    assert result["checks"]["selected_source_sha256_match"] is True
    assert result["checks"]["semantic_expected_rows"] == ["S1"]


def test_frozen_gate_fails_closed_when_local_source_changes(tmp_path):
    review = tmp_path / "review.pdf"
    source = tmp_path / "source.pdf"
    review.write_bytes(b"review")
    source.write_bytes(b"changed")
    (tmp_path / "preflight.json").write_text(
        json.dumps({"capture": {"frozen_whitelist": ["t"]}}), encoding="utf-8")
    manifest = {
        "review": {"path": "review.pdf",
                   "sha256": hashlib.sha256(b"review").hexdigest()},
        "selected_source": {"path": "source.pdf",
                            "sha256": hashlib.sha256(b"original").hexdigest()},
        "table_whitelist": ["t"],
        "contract_files": ["preflight.json"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_frozen_benchmark(
        tmp_path, [{"audit_id": "S", "expected_match_mode": "semantic"}])

    assert result["status"] == "failed"
    assert result["checks"]["selected_source_sha256_match"] is False
    assert any("hash differs" in error for error in result["errors"])
