"""Recompute the TableCapture v1/v2 release decision from public metrics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from react_review.contracts import read_json_object, repo_root, sha256_file


class ResultError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResultError(message)


def _control_fingerprint(controls: dict[str, Any]) -> str:
    body = {key: controls.get(key) for key in (
        "provider", "model", "temperature", "seed", "max_tokens")}
    material = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest().upper()


def _failure_reasons(v1: dict[str, Any], v2: dict[str, Any]) -> list[str]:
    one = v1["metrics"]
    two = v2["metrics"]
    reasons: list[str] = []
    if float(two["table_recall"]) < 1.0:
        reasons.append("expected_table_missed")
    if float(two["row_recall"]) < float(one["row_recall"]):
        reasons.append("row_recall_regressed")
    if (float(two["exact_cell_accuracy"]) < float(one["exact_cell_accuracy"])
            or float(two["normalized_cell_accuracy"])
            < float(one["normalized_cell_accuracy"])):
        reasons.append("cell_accuracy_regressed")
    if int(two["unanchored_cells"]) > int(one["unanchored_cells"]):
        reasons.append("unanchored_cells_increased")
    if int(two["hallucinated_cells"]) > int(one["hallucinated_cells"]):
        reasons.append("hallucinated_cells_increased")
    if float(two["json_success_rate"]) < 1.0:
        reasons.append("json_failed")
    if float(two["schema_success_rate"]) < 1.0:
        reasons.append("schema_failed")
    if int(v2.get("forbidden_domain_terms_introduced") or 0):
        reasons.append("frozen_domain_terms_introduced")
    return reasons


def check_result(path: Path | str) -> dict[str, Any]:
    root = repo_root()
    result_path = Path(path)
    if not result_path.is_absolute():
        result_path = root / result_path
    body = read_json_object(result_path, kind="TableCapture A/B result")
    _require(body.get("schema_version") == 1, "unsupported result schema")

    parent_path = root / str(body.get("parent_manifest") or "")
    parent = read_json_object(parent_path, kind="TableCapture A/B manifest")
    _require(sha256_file(parent_path) == str(
        body.get("parent_manifest_sha256") or "").upper(),
        "parent manifest hash does not match")

    controls = body.get("controls")
    _require(isinstance(controls, dict), "result has no controls")
    fingerprint = _control_fingerprint(controls)
    _require(fingerprint == controls.get("control_fingerprint"),
             "control fingerprint does not match model/seed/temperature/max_tokens")
    _require(controls.get("execution_mode") == "live", "result is not from live calls")
    _require(controls.get("actual_cost_available") is False,
             "result must not invent a currency cost without usage/pricing")

    documents = {str(d["document_id"]): d for d in parent.get("documents") or []}
    prompts = {str(p["prompt_id"]): p for p in parent.get("prompt_profiles") or []}
    expected_matrix = {(document_id, prompt_id)
                       for document_id in documents for prompt_id in prompts}
    calls = body.get("calls")
    _require(isinstance(calls, list), "result calls are not a list")
    actual_matrix = {(str(c.get("document_id")), str(c.get("prompt_id")))
                     for c in calls if isinstance(c, dict)}
    _require(len(calls) == len(actual_matrix) == int(parent["planned_live_calls"]),
             "result does not contain exactly one row per planned call")
    _require(actual_matrix == expected_matrix, "result call matrix differs from preregistration")
    _require(int(controls.get("calls") or 0) == len(calls),
             "controls call count differs from result rows")

    response_hashes: set[str] = set()
    by_document: dict[str, dict[str, dict[str, Any]]] = {}
    for call in calls:
        document_id = str(call["document_id"])
        prompt_id = str(call["prompt_id"])
        document = documents[document_id]
        prompt = prompts[prompt_id]
        _require(call.get("control_fingerprint") == fingerprint,
                 f"{document_id}/{prompt_id} did not use the paired controls")
        _require(call.get("response_reused") is False,
                 f"{document_id}/{prompt_id} declares a reused response")
        for field in ("rendered_prompt_sha256", "response_sha256", "capture_sha256",
                      "review_text_sha256"):
            value = str(call.get(field) or "")
            _require(len(value) == 64 and all(c in "0123456789ABCDEF" for c in value),
                     f"{document_id}/{prompt_id} has invalid {field}")
        response_hashes.add(str(call["response_sha256"]))
        _require(call.get("pdf_sha256") == document.get("pdf_sha256"),
                 f"{document_id}/{prompt_id} PDF hash differs from preregistration")
        _require(call.get("gold_sha256") == document.get("gold_sha256"),
                 f"{document_id}/{prompt_id} gold hash differs from preregistration")
        _require(call.get("prompt_contract_sha256")
                 == prompt.get("rendered_prompt_sha256"),
                 f"{document_id}/{prompt_id} prompt contract hash differs")
        _require(isinstance(call.get("metrics"), dict),
                 f"{document_id}/{prompt_id} has no metrics")
        by_document.setdefault(document_id, {})[prompt_id] = call
    _require(len(response_hashes) == len(calls), "response hashes are not unique")

    document_failures: dict[str, list[str]] = {}
    candidate_terms = 0
    for document_id, pair in by_document.items():
        _require(pair["table_capture_v1"]["review_text_sha256"]
                 == pair["table_capture_v2"]["review_text_sha256"],
                 f"{document_id} prompts did not receive the same PDF text")
        reasons = _failure_reasons(
            pair["table_capture_v1"], pair["table_capture_v2"])
        document_failures[document_id] = reasons
        candidate_terms += int(
            pair["table_capture_v2"].get("forbidden_domain_terms_introduced") or 0)

    failed_documents = sum(bool(reasons) for reasons in document_failures.values())
    if failed_documents == 0:
        diagnosis = "passed"
        candidate = "promoted"
        production_default = "table_capture_v2"
    elif failed_documents == len(document_failures):
        diagnosis = "regressed"
        candidate = "not_promoted"
        production_default = "table_capture_v1"
    else:
        diagnosis = "mixed"
        candidate = "not_promoted"
        production_default = "table_capture_v1"

    decision = body.get("decision") or {}
    _require(decision.get("diagnosis") == diagnosis, "decision diagnosis contradicts metrics")
    _require(decision.get("candidate_v2") == candidate,
             "decision candidate_v2 contradicts metrics")
    _require(decision.get("production_default") == production_default,
             "decision production_default contradicts metrics")
    recorded = decision.get("document_results") or {}
    _require(set(recorded) == set(document_failures),
             "decision document results are incomplete")
    for document_id, reasons in document_failures.items():
        _require(set(recorded[document_id]) == set(reasons),
                 f"decision reasons contradict metrics for {document_id}")

    transitions = body.get("error_type_transitions") or []
    published = {(row["document_id"], row["error_type"]): row
                 for row in transitions}
    expected_transitions: dict[tuple[str, str], tuple[int, int]] = {}
    for document_id, pair in by_document.items():
        v1_errors = pair["table_capture_v1"]["metrics"].get("error_counts") or {}
        v2_errors = pair["table_capture_v2"]["metrics"].get("error_counts") or {}
        for error_type in set(v1_errors) | set(v2_errors):
            expected_transitions[(document_id, error_type)] = (
                int(v1_errors.get(error_type, 0)), int(v2_errors.get(error_type, 0)))
        expected_transitions[(document_id, "text_layer_unanchored_cells")] = (
            int(pair["table_capture_v1"].get("text_layer_unanchored_cells") or 0),
            int(pair["table_capture_v2"].get("text_layer_unanchored_cells") or 0))
    # Zero-to-zero transitions add no information and need not be published.
    expected_transitions = {key: value for key, value in expected_transitions.items()
                            if value != (0, 0)}
    _require(set(published) == set(expected_transitions),
             "error transition rows are incomplete or contain an unknown row")
    for key, (v1, v2) in expected_transitions.items():
        row = published[key]
        transition = "improved" if v2 < v1 else "regressed" if v2 > v1 else "unchanged"
        _require((int(row["v1"]), int(row["v2"]), row["transition"])
                 == (v1, v2, transition), f"error transition {key} contradicts metrics")

    policy = body.get("artifact_policy") or {}
    _require(policy.get("raw_responses_committed") is False
             and policy.get("review_text_committed") is False
             and policy.get("full_gold_committed") is False,
             "private artifacts are not excluded by the result policy")

    return {
        "calls": len(calls),
        "call_matrix": actual_matrix,
        "candidate_v2": candidate,
        "production_default": production_default,
        "diagnosis": diagnosis,
        "documents": document_failures,
        "paired_controls_verified": True,
        "response_hashes_unique": len(response_hashes) == len(calls),
        "candidate_domain_terms_introduced": candidate_terms,
    }
