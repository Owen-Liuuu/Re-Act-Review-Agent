"""Frozen-benchmark validation and stratified evaluation diagnostics.

The ordinary accuracy metrics must stay honest: declared limitations are not
removed from the headline score.  This module adds a second view that explains
which failures were frozen in advance and which appeared only in the live run.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _within(base: Path, relative: str) -> Path:
    path = (base / relative).resolve()
    if path != base.resolve() and base.resolve() not in path.parents:
        raise ValueError(f"benchmark path escapes its directory: {relative}")
    return path


def validate_frozen_benchmark(
    benchmark: Path, rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Validate a frozen benchmark's local entry contract.

    Legacy benchmarks without ``manifest.json`` remain usable and are marked as
    ``legacy_unfrozen``.  A frozen benchmark fails closed on a missing/mutated
    review or source PDF, a changed whitelist, missing contract files, or no
    declared semantic row.
    """
    benchmark = benchmark.resolve()
    manifest_path = benchmark / "manifest.json"
    if not manifest_path.is_file():
        return {
            "status": "legacy_unfrozen",
            "benchmark": str(benchmark),
            "checks": {},
            "errors": [],
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    checks: dict[str, Any] = {}

    for name in ("review", "selected_source"):
        spec = manifest.get(name) or {}
        relative = str(spec.get("path") or "")
        try:
            path = _within(benchmark, relative) if relative else benchmark
        except ValueError as exc:
            errors.append(str(exc))
            checks[f"{name}_exists"] = False
            checks[f"{name}_sha256_match"] = False
            continue
        exists = bool(relative) and path.is_file()
        checks[f"{name}_exists"] = exists
        expected_hash = str(spec.get("sha256") or "").upper()
        actual_hash = _sha256(path) if exists else ""
        matches = bool(expected_hash) and actual_hash == expected_hash
        checks[f"{name}_sha256_match"] = matches
        if not exists:
            errors.append(f"{name} PDF is missing: {relative}")
        elif not matches:
            errors.append(f"{name} PDF hash differs from manifest: {relative}")

    missing_contract = [
        name for name in (manifest.get("contract_files") or [])
        if not _within(benchmark, str(name)).is_file()
    ]
    checks["missing_contract_files"] = missing_contract
    if missing_contract:
        errors.append("missing contract files: " + ", ".join(missing_contract))

    manifest_whitelist = list(manifest.get("table_whitelist") or [])
    preflight_path = benchmark / "preflight.json"
    preflight = (json.loads(preflight_path.read_text(encoding="utf-8-sig"))
                 if preflight_path.is_file() else {})
    frozen_whitelist = list(
        ((preflight.get("capture") or {}).get("frozen_whitelist") or []))
    checks["table_whitelist"] = manifest_whitelist
    checks["table_whitelist_matches_preflight"] = (
        bool(manifest_whitelist) and manifest_whitelist == frozen_whitelist)
    if not checks["table_whitelist_matches_preflight"]:
        errors.append("manifest table whitelist differs from frozen preflight")

    semantic_rows = [
        str(row.get("audit_id") or "") for row in rows
        if (row.get("expected_match_mode") or "").strip() == "semantic"
    ]
    checks["semantic_expected_rows"] = semantic_rows
    if not semantic_rows:
        errors.append("no answer-key row expects semantic comparison")

    checks["declared_known_gaps"] = [
        {"audit_id": str(row.get("audit_id") or ""),
         "gap": str(row.get("known_gap") or "")}
        for row in rows if (row.get("known_gap") or "").strip()
    ]
    return {
        "status": "pass" if not errors else "failed",
        "benchmark_id": str(manifest.get("benchmark_id") or benchmark.name),
        "benchmark": str(benchmark),
        "checks": checks,
        "errors": errors,
    }


def benchmark_diagnostics(results: Iterable[Any]) -> dict[str, Any]:
    """Separate frozen limitations from unexpected live-run differences."""
    rows = list(results)
    with_contract = [r for r in rows if r.expected_match_mode]
    if not with_contract:
        return {"status": "not_applicable"}

    differences = [r for r in rows if r.predicted_label != r.expected_label]
    known = [r for r in differences if r.known_gap]
    unexpected = [r for r in differences if not r.known_gap]
    declared_gap_rows = [r for r in rows if r.known_gap]
    semantic_expected = [r for r in rows if r.expected_match_mode == "semantic"]
    semantic_reached = [r for r in semantic_expected if r.match_mode == "semantic"]
    mode_diffs = [
        r for r in with_contract if r.match_mode != r.expected_match_mode
    ]
    relation_diffs = [
        r for r in semantic_expected
        if r.expected_semantic_relation
        and r.semantic_relation != r.expected_semantic_relation
    ]
    review_required_diffs = [
        r for r in rows
        if r.expected_match_mode
        and r.review_required != r.expected_review_required
    ]
    silent = [
        r for r in rows
        if r.expected_label in {"mismatch", "unit_mismatch"}
        and r.predicted_label == "match"
        and not r.review_required
    ]

    def row_summary(r: Any) -> dict[str, Any]:
        return {
            "audit_id": r.audit_id,
            "study_id": r.study_id,
            "field_type": r.field_type,
            "expected_label": r.expected_label,
            "predicted_label": r.predicted_label,
            "expected_match_mode": r.expected_match_mode,
            "observed_match_mode": r.match_mode,
            "expected_semantic_relation": r.expected_semantic_relation,
            "observed_semantic_relation": r.semantic_relation,
            "known_gap": r.known_gap,
            "reason": r.match_reason,
            "extraction_correct": r.extraction_correct,
            "expected_review_required": r.expected_review_required,
            "observed_review_required": r.review_required,
        }

    gap_outcomes = []
    for row in declared_gap_rows:
        if row.predicted_label != row.expected_label:
            outcome = "exposed"
        elif not row.extraction_correct:
            outcome = "masked_by_incomplete_or_wrong_extraction"
        else:
            outcome = "not_reproduced"
        gap_outcomes.append({**row_summary(row), "outcome": outcome})

    semantic_gate_passed = bool(semantic_reached)
    unexpected_contract_ids = sorted({
        r.audit_id for r in [*unexpected, *mode_diffs, *relation_diffs,
                             *review_required_diffs]
    })
    status = ("fail_unexpected_differences" if unexpected_contract_ids
              else "fail_semantic_not_reached" if not semantic_gate_passed
              else "pass_with_declared_gaps" if declared_gap_rows
              else "pass")
    return {
        "status": status,
        "expected_mode_counts": dict(Counter(r.expected_match_mode for r in with_contract)),
        "observed_mode_counts": dict(Counter(r.match_mode for r in with_contract)),
        "semantic": {
            "expected_count": len(semantic_expected),
            "reached_count": len(semantic_reached),
            "reached_audit_ids": [r.audit_id for r in semantic_reached],
            "not_reached_audit_ids": [r.audit_id for r in semantic_expected
                                      if r.match_mode != "semantic"],
            "relation_differences": [row_summary(r) for r in relation_diffs],
            "escalation_gate_passed": semantic_gate_passed,
        },
        "mode_differences": [row_summary(r) for r in mode_diffs],
        "differences": {
            "total": len(differences),
            "declared_known_gap_count": len(known),
            "declared_known_gaps": [row_summary(r) for r in known],
            "unexpected_count": len(unexpected),
            "unexpected": [row_summary(r) for r in unexpected],
            "unexpected_contract_audit_ids": unexpected_contract_ids,
        },
        "declared_gap_outcomes": gap_outcomes,
        "review_required_differences": [row_summary(r) for r in review_required_diffs],
        "silent_releases": {
            "total": len(silent),
            "declared_known_gap_count": sum(bool(r.known_gap) for r in silent),
            "unexpected_count": sum(not r.known_gap for r in silent),
            "rows": [row_summary(r) for r in silent],
        },
    }


def format_benchmark_diagnostics(diagnostics: dict[str, Any]) -> str:
    if diagnostics.get("status") == "not_applicable":
        return ""
    semantic = diagnostics["semantic"]
    differences = diagnostics["differences"]
    silent = diagnostics["silent_releases"]
    return "\n".join([
        "",
        "============= frozen benchmark =============",
        f"status                 : {diagnostics['status']}",
        f"expected modes         : {diagnostics['expected_mode_counts']}",
        f"observed modes         : {diagnostics['observed_mode_counts']}",
        f"semantic escalations   : {semantic['reached_count']}/"
        f"{semantic['expected_count']} expected rows",
        f"declared gap failures  : {differences['declared_known_gap_count']}",
        f"unexpected differences : {differences['unexpected_count']}",
        f"semantic relation diffs: {len(semantic['relation_differences'])}",
        f"review-required diffs  : {len(diagnostics['review_required_differences'])}",
        f"silent releases        : {silent['total']} "
        f"(declared={silent['declared_known_gap_count']} "
        f"unexpected={silent['unexpected_count']})",
        "============================================",
    ])
