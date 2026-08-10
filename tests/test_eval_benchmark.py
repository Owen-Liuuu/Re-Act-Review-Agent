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


# --- manifest v2: many sources, declared routes (P8-0 U11) ---

def _v2_manifest(tmp_path, **overrides):
    import hashlib, json
    (tmp_path / "raw").mkdir(exist_ok=True)
    files = {"raw/review.pdf": b"%PDF review", "raw/a.pdf": b"%PDF a",
             "raw/b.pdf": b"%PDF b", "audit_template.csv": b"audit_id\nA1\n"}
    artifacts = []
    for rel, body in files.items():
        (tmp_path / rel).write_bytes(body)
        role = ("review" if "review" in rel else
                "contract" if rel.endswith(".csv") else "source")
        artifacts.append({"path": rel, "role": role,
                          "sha256": hashlib.sha256(body).hexdigest().upper()})
    manifest = {"schema_version": 2, "benchmark_id": "many_sources",
                "artifacts": artifacts, "required_routes": []}
    manifest.update(overrides)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_a_benchmark_with_several_source_papers_can_be_pinned(tmp_path):
    """v1 could name one source. Nine papers could not be frozen at all."""
    gate = validate_frozen_benchmark(_v2_manifest(tmp_path), [{"audit_id": "A1"}])
    assert gate["status"] == "pass"
    assert gate["checks"]["artifacts_by_role"] == {"review": 1, "source": 2,
                                                  "contract": 1}
    assert gate["checks"]["artifacts_verified"] == 4


def test_a_mutated_source_paper_fails_the_gate(tmp_path):
    root = _v2_manifest(tmp_path)
    (root / "raw" / "b.pdf").write_bytes(b"%PDF edited")
    gate = validate_frozen_benchmark(root, [{"audit_id": "A1"}])
    assert gate["status"] == "failed"
    assert any("hash differs" in e for e in gate["errors"])


def test_a_declared_route_the_answer_key_never_exercises_fails(tmp_path):
    root = _v2_manifest(tmp_path, required_routes=["semantic"])
    gate = validate_frozen_benchmark(root, [{"audit_id": "A1"}])
    assert gate["status"] == "failed"
    assert any("requires a semantic route" in e for e in gate["errors"])


def test_a_numbers_only_benchmark_is_allowed_to_say_so(tmp_path):
    """The old rule was wired into the checker; now it is in the file."""
    root = _v2_manifest(tmp_path, required_routes=["numeric"])
    gate = validate_frozen_benchmark(
        root, [{"audit_id": "A1", "expected_match_mode": "numeric"}])
    assert gate["status"] == "pass"


def test_the_eat_benchmark_is_now_pinned_like_the_melanoma_one():
    import csv
    from pathlib import Path

    eat = Path(__file__).resolve().parents[1] / "eval" / "benchmark"
    with open(eat / "audit_template.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    gate = validate_frozen_benchmark(eat, rows)
    assert gate["status"] == "pass"          # no longer "legacy_unfrozen"
    assert gate["checks"]["artifacts_by_role"]["source"] == 9


def test_the_eat_manifest_states_the_context_its_recordings_used():
    """The research context is part of the prompt, so it is part of the freeze.

    A manifest that described the domain in nicer words would change every
    prompt and invalidate every recording — which is what happened the first
    time this manifest was written, and is why the string is pinned here.
    """
    import json
    from pathlib import Path

    eat = Path(__file__).resolve().parents[1] / "eval" / "benchmark"
    manifest = json.loads((eat / "manifest.json").read_text(encoding="utf-8-sig"))
    assert manifest["domain"] == "EAT thickness/volume in T1DM vs healthy controls"
