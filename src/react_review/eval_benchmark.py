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
    review or source PDF, a changed whitelist, missing contract files, or a
    route it declares and its answer key does not exercise.

    Two manifest schemas. **v1** names one review and one ``selected_source``,
    and requires a semantic row — the shape the melanoma checkpoint was frozen
    in. **v2** takes ``artifacts`` with a role each, so a benchmark with nine
    source papers can be pinned at all, and states its ``required_routes``
    instead of having "there must be a semantic row" wired into this function:
    a benchmark that audits only numbers is a legitimate benchmark, and the
    rule it is held to belongs in its own file where it can be reviewed.
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

    version = int(manifest.get("schema_version") or 1)
    checks["manifest_schema_version"] = version
    if version >= 2:
        _check_artifacts(benchmark, manifest, checks, errors)
    else:
        _check_v1_pdfs(benchmark, manifest, checks, errors)

    _check_contract_files(benchmark, manifest, checks, errors)
    _check_whitelist(benchmark, manifest, checks, errors)
    _check_routes(manifest, rows, checks, errors, version=version)

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


def _check_artifacts(benchmark: Path, manifest: dict, checks: dict,
                     errors: list[str]) -> None:
    """v2: every declared artifact exists and still hashes to what it did."""
    by_role: dict[str, int] = {}
    verified = 0
    for spec in (manifest.get("artifacts") or []):
        relative = str(spec.get("path") or "")
        role = str(spec.get("role") or "unknown")
        by_role[role] = by_role.get(role, 0) + 1
        try:
            path = _within(benchmark, relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"{role} artifact is missing: {relative}")
            continue
        declared = str(spec.get("sha256") or "").upper()
        if not declared:
            errors.append(f"{role} artifact declares no sha256: {relative}")
        elif _sha256(path) != declared:
            errors.append(f"{role} artifact hash differs from manifest: {relative}")
        else:
            verified += 1
    checks["artifacts_by_role"] = by_role
    checks["artifacts_verified"] = verified
    if not by_role.get("review"):
        errors.append("the manifest declares no review artifact")
    if not by_role.get("source"):
        errors.append("the manifest declares no source artifact")


def _check_v1_pdfs(benchmark: Path, manifest: dict, checks: dict,
                   errors: list[str]) -> None:
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


def _check_contract_files(benchmark: Path, manifest: dict, checks: dict,
                          errors: list[str]) -> None:
    missing_contract = [
        name for name in (manifest.get("contract_files") or [])
        if not _within(benchmark, str(name)).is_file()
    ]
    checks["missing_contract_files"] = missing_contract
    if missing_contract:
        errors.append("missing contract files: " + ", ".join(missing_contract))


def _check_whitelist(benchmark: Path, manifest: dict, checks: dict,
                     errors: list[str]) -> None:
    manifest_whitelist = list(manifest.get("table_whitelist") or [])
    preflight_path = benchmark / "preflight.json"
    preflight = (json.loads(preflight_path.read_text(encoding="utf-8-sig"))
                 if preflight_path.is_file() else {})
    frozen_whitelist = list(
        ((preflight.get("capture") or {}).get("frozen_whitelist") or []))
    checks["table_whitelist"] = manifest_whitelist
    if not manifest_whitelist and not frozen_whitelist:
        # A benchmark whose answer key was not built from a frozen table capture
        # has no whitelist to agree with. Silence on both sides is consistent;
        # only a disagreement is a failure.
        checks["table_whitelist_matches_preflight"] = "not_declared"
        return
    checks["table_whitelist_matches_preflight"] = (
        manifest_whitelist == frozen_whitelist and bool(manifest_whitelist))
    if checks["table_whitelist_matches_preflight"] is not True:
        errors.append("manifest table whitelist differs from frozen preflight")


def _check_routes(manifest: dict, rows: list[dict[str, str]], checks: dict,
                  errors: list[str], *, version: int) -> None:
    """Every route the benchmark says it exercises must appear in its key.

    v1 had one route hard-coded here ("there must be a semantic row"), which
    made a numbers-only benchmark impossible to freeze. v2 declares its own
    routes, so the rule is visible in the file being judged — and declaring
    none is a statement a reader can see and dispute, not a hidden exemption.
    """
    modes = {(row.get("expected_match_mode") or "").strip()
             for row in rows} - {""}
    checks["observed_expected_modes"] = sorted(modes)
    checks["semantic_expected_rows"] = [
        str(row.get("audit_id") or "") for row in rows
        if (row.get("expected_match_mode") or "").strip() == "semantic"]

    required = ([str(r) for r in (manifest.get("required_routes") or [])]
                if version >= 2 else ["semantic"])
    checks["required_routes"] = required
    for route in required:
        if route not in modes:
            errors.append(
                f"the manifest requires a {route} route, which no answer-key "
                "row expects")


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
