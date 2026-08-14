"""Preflight the TableCapture A/B gate without making a model call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from eval.table_capture_score import load_gold
from react_review.contracts import read_json_object, repo_root, sha256_file
from react_review.parser.table_capture_contract import TableCapturePromptContract


DEFAULT_MANIFEST = Path("eval/table_capture_ab_v1.json")


class PreflightError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _redact_url(value: str) -> str:
    parts = urlsplit(str(value or ""))
    if not parts.scheme:
        return ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _config_summary(path: Path | None, calls: int) -> dict[str, Any]:
    if path is None:
        return {}
    from react_review.core.config import load_config

    settings = load_config(path).llm
    return {
        "config_file": path.name,
        "provider": settings.provider,
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens_per_call": settings.max_tokens,
        "maximum_output_tokens": calls * settings.max_tokens,
        "endpoint": _redact_url(settings.base_url),
        "api_key_recorded": False,
        "cost_estimate": {
            "status": "not_available",
            "reason": "the repository has no versioned provider pricing table",
        },
    }


def preflight(manifest_path: Path | str = DEFAULT_MANIFEST, *,
              require_private: bool = False,
              config_path: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root / path
    body = read_json_object(path, kind="TableCapture A/B manifest")
    _require(body.get("schema_version") == 1, "unsupported A/B manifest schema")

    schema = root / str(body.get("gold_schema") or "")
    _require(schema.is_file(), f"gold schema is missing: {schema}")
    _require(sha256_file(schema) == str(body.get("gold_schema_sha256") or "").upper(),
             "gold schema hash does not match the manifest")

    prompts = body.get("prompt_profiles")
    documents = body.get("documents")
    _require(isinstance(prompts, list) and prompts, "manifest pins no prompt profiles")
    _require(isinstance(documents, list) and documents, "manifest names no documents")
    prompt_ids: set[str] = set()
    for entry in prompts:
        _require(isinstance(entry, dict), "prompt profile entry is not an object")
        prompt_id = str(entry.get("prompt_id") or "")
        _require(prompt_id not in prompt_ids, f"duplicate prompt id {prompt_id!r}")
        prompt_ids.add(prompt_id)
        contract = TableCapturePromptContract.load(prompt_id)
        _require(contract.path.resolve() == (
            root / str(entry.get("prompt_contract") or "")).resolve(),
            f"{prompt_id} points at the wrong prompt contract file")
        _require(not contract.drifts(), f"{prompt_id} prompt contract has drifted")
        _require(contract.rendered_prompt_sha256 == str(
            entry.get("rendered_prompt_sha256") or "").upper(),
            f"{prompt_id} prompt hash does not match the A/B manifest")

    planned_calls = int(body.get("planned_live_calls") or 0)
    _require(planned_calls == len(prompts) * len(documents),
             "planned_live_calls must equal documents times prompt profiles")

    public_counts = {
        "tables": sum(int(d.get("table_count") or 0) for d in documents),
        "rows": sum(int(d.get("row_count") or 0) for d in documents),
        "cells": sum(int(d.get("cell_count") or 0) for d in documents),
    }
    verified_documents: list[dict[str, Any]] = []
    if require_private:
        for document in documents:
            document_id = str(document.get("document_id") or "")
            pdf = root / str(document.get("pdf") or "")
            gold_path = root / str(document.get("gold") or "")
            _require(pdf.is_file(), f"private/source PDF is missing for {document_id}: {pdf}")
            _require(gold_path.is_file(), f"private gold is missing for {document_id}: {gold_path}")
            _require(sha256_file(pdf) == str(document.get("pdf_sha256") or "").upper(),
                     f"{document_id} PDF hash does not match")
            _require(sha256_file(gold_path) == str(
                document.get("gold_sha256") or "").upper(),
                f"{document_id} gold hash does not match")
            records = load_gold(gold_path)
            _require({r["document_id"] for r in records} == {document_id},
                     f"{document_id} gold contains another document identity")
            table_count = len({r["table_id"] for r in records})
            row_ids = {(r["table_id"], r["row_id"]) for r in records}
            data_rows = {key for key in row_ids if key[1].startswith("data_")}
            counts = {
                "table_count": table_count,
                "row_count": len(row_ids),
                "data_row_count": len(data_rows),
                "cell_count": len(records),
                "value_cell_count": sum(r["blank_kind"] == "value" for r in records),
            }
            for key, actual in counts.items():
                _require(actual == int(document.get(key) or 0),
                         f"{document_id} {key} is {actual}, manifest says {document.get(key)}")
            verified_documents.append({
                "document_id": document_id,
                "pdf_sha256": sha256_file(pdf),
                "gold_sha256": sha256_file(gold_path),
                **counts,
            })

    return {
        "ready": True,
        "dataset_id": body.get("dataset_id"),
        "manifest_sha256": sha256_file(path),
        "document_count": len(documents),
        "prompt_count": len(prompts),
        "planned_live_calls": planned_calls,
        "seed": int(body.get("seed") or 0),
        "gold_counts": public_counts,
        "private_inputs_verified": require_private,
        "documents": verified_documents,
        "prompt_profiles": [
            {"prompt_id": p["prompt_id"],
             "rendered_prompt_sha256": p["rendered_prompt_sha256"]}
            for p in prompts
        ],
        "artifact_root": body.get("artifact_root"),
        "raw_response_retained": True,
        "config": _config_summary(config_path, planned_calls),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-private", action="store_true")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        report = preflight(
            args.manifest, require_private=args.require_private,
            config_path=args.config)
    except Exception as exc:  # noqa: BLE001 - command must fail closed
        print(json.dumps({"ready": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
