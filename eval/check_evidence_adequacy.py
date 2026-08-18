"""Validate the frozen evidence-adequacy transition contract.

This checker initially guards the preimplementation adjudication itself.  The
comparison-gate commit extends it to join these rows to produced results; the
expectations below must not move in response to implementation output.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSITION = ROOT / "eval" / "benchmark_1" / "evidence_adequacy_transition_v1.csv"

COLUMNS = (
    "audit_id",
    "old_label",
    "expected_new_label",
    "expected_adequacy",
    "required_failed_axis",
    "document_scope",
    "reason",
    "adjudicator",
)
HARD_NOT_COMPARABLE = {
    "A028", "A030", "A031", "A037", "A038",
    "A041", "A042", "A044", "A045",
}
TRUE_DIFFERENCES = {"A010", "A013"}
REMAINDER = {
    "A_06", "B_07", "C_07", "D_06",
    "D_07", "E_07", "F_07", "G_07",
}
EXPECTED_IDS = HARD_NOT_COMPARABLE | TRUE_DIFFERENCES | REMAINDER
POSITIVE_ABSTRACT_CONTROLS = {"A034", "A035"}


def load_transition(path: Path = DEFAULT_TRANSITION) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(
                f"transition columns must be {COLUMNS!r}; got {reader.fieldnames!r}"
            )
        rows = [{key: (value or "").strip() for key, value in row.items()}
                for row in reader]
    return rows


def validate_transition(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    ids = [row["audit_id"] for row in rows]
    if len(rows) != 19:
        errors.append(f"expected 19 frozen accusations; got {len(rows)}")
    if len(ids) != len(set(ids)):
        errors.append("audit_id values must be unique")
    if set(ids) != EXPECTED_IDS:
        errors.append(
            f"transition identities changed: missing={sorted(EXPECTED_IDS - set(ids))} "
            f"extra={sorted(set(ids) - EXPECTED_IDS)}"
        )

    by_id = {row["audit_id"]: row for row in rows}
    for audit_id in sorted(HARD_NOT_COMPARABLE):
        row = by_id.get(audit_id)
        if row and (
            row["expected_new_label"] != "not_comparable"
            or row["expected_adequacy"] != "insufficient"
            or not row["required_failed_axis"]
        ):
            errors.append(f"{audit_id} must freeze insufficient/not_comparable with a failed axis")

    for audit_id in sorted(TRUE_DIFFERENCES):
        row = by_id.get(audit_id)
        if row and (
            row["expected_new_label"] != "unit_mismatch"
            or row["expected_adequacy"] != "sufficient"
            or row["required_failed_axis"]
        ):
            errors.append(f"{audit_id} must preserve the verified unit mismatch")

    for row in rows:
        if not row["reason"] or not row["adjudicator"]:
            errors.append(f"{row['audit_id'] or '<blank>'} lacks reason/adjudicator")
        if row["document_scope"] not in {"full_text", "abstract_only", "metadata_only"}:
            errors.append(f"{row['audit_id']} has invalid document_scope")
        if row["expected_adequacy"] not in {"sufficient", "insufficient", "unknown"}:
            errors.append(f"{row['audit_id']} has invalid expected_adequacy")
        if row["expected_new_label"] not in {
            "match", "mismatch", "unit_mismatch", "not_comparable"
        }:
            errors.append(f"{row['audit_id']} has invalid expected_new_label")
        if row["expected_new_label"] == "not_comparable" and (
            row["expected_adequacy"] == "sufficient"
        ):
            errors.append(f"{row['audit_id']} cannot be sufficient and not_comparable")
    return errors


def _field(value, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else default


def _result_rows(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("results JSON must be a list or object")
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    report = payload.get("report") or {}
    if isinstance(report, dict) and isinstance(report.get("results"), list):
        return report["results"]
    if isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError("results JSON has no rows, results, or report.results list")


def validate_results(
    transition: list[dict[str, str]], produced: list[dict]
) -> list[str]:
    """Join frozen expectations to produced rows by claim identity."""
    errors: list[str] = []
    by_id: dict[str, dict] = {}
    for row in produced:
        audit_id = str(row.get("audit_id") or row.get("review_data_id") or "")
        if not audit_id:
            continue
        if audit_id in by_id:
            errors.append(f"produced results repeat audit_id {audit_id}")
        by_id[audit_id] = row

    for expected in transition:
        audit_id = expected["audit_id"]
        actual = by_id.get(audit_id)
        if actual is None:
            errors.append(f"produced results are missing {audit_id}")
            continue
        label = str(actual.get("predicted_label") or actual.get("label") or "")
        if label != expected["expected_new_label"]:
            errors.append(
                f"{audit_id} label {label!r} != "
                f"{expected['expected_new_label']!r}")
        adequacy = actual.get("evidence_adequacy") or {}
        status = str(actual.get("evidence_adequacy_status")
                     or _field(adequacy, "status", ""))
        scope = str(actual.get("document_scope")
                    or _field(adequacy, "document_scope", ""))
        if status != expected["expected_adequacy"]:
            errors.append(
                f"{audit_id} adequacy {status!r} != "
                f"{expected['expected_adequacy']!r}")
        if scope != expected["document_scope"]:
            errors.append(
                f"{audit_id} document_scope {scope!r} != "
                f"{expected['document_scope']!r}")
        failed_axis = expected["required_failed_axis"]
        if failed_axis:
            axis = (_field(adequacy, "axis_results", {}) or {}).get(failed_axis) or {}
            axis_status = str(_field(axis, "status", ""))
            if axis_status not in {"fail", "unknown"}:
                errors.append(
                    f"{audit_id} required axis {failed_axis!r} was not refused")

    for audit_id in sorted(POSITIVE_ABSTRACT_CONTROLS):
        actual = by_id.get(audit_id)
        if actual is None:
            errors.append(f"produced results are missing positive control {audit_id}")
            continue
        label = str(actual.get("predicted_label") or actual.get("label") or "")
        adequacy = actual.get("evidence_adequacy") or {}
        status = str(actual.get("evidence_adequacy_status")
                     or _field(adequacy, "status", ""))
        if label != "match" or status != "sufficient":
            errors.append(
                f"{audit_id} must remain sufficient/match; got {status}/{label}")

    if len(produced) == 77:
        labels = Counter(
            str(row.get("predicted_label") or row.get("label") or "")
            for row in produced)
        expected_counts = {
            "match": 30, "mismatch": 0,
            "unit_mismatch": 2, "not_comparable": 45,
        }
        if any(labels[label] != count for label, count in expected_counts.items()):
            errors.append(
                f"77-row label counts changed: got {dict(labels)}, "
                f"expected {expected_counts}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("transition", nargs="?", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--results", type=Path,
                        help="produced JSON rows or EvidencePackage to validate")
    args = parser.parse_args(argv)
    rows = load_transition(args.transition)
    errors = validate_transition(rows)
    if args.results:
        payload = json.loads(args.results.read_text(encoding="utf-8-sig"))
        errors.extend(validate_results(rows, _result_rows(payload)))
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    suffix = (" and produced results satisfy the gate"
              if args.results else "")
    print(
        "PASS: 19 accusations frozen "
        f"(17 -> not_comparable; 2 verified unit mismatches preserved){suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
