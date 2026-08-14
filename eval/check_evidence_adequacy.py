"""Validate the frozen evidence-adequacy transition contract.

This checker initially guards the preimplementation adjudication itself.  The
comparison-gate commit extends it to join these rows to produced results; the
expectations below must not move in response to implementation output.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSITION = ROOT / "eval" / "benchmark" / "evidence_adequacy_transition_v1.csv"

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("transition", nargs="?", type=Path, default=DEFAULT_TRANSITION)
    args = parser.parse_args(argv)
    rows = load_transition(args.transition)
    errors = validate_transition(rows)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print(
        "PASS: 19 accusations frozen "
        "(17 -> not_comparable; 2 verified unit mismatches preserved)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
