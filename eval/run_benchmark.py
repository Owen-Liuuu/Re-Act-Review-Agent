"""Score the deterministic audit core against the hand-labelled benchmark.

Runs ``compare_values`` over:
  - eval/benchmark_1/audit_template.csv   -> reproduce expected_label (axis 3)
  - eval/benchmark_1/seeded_discrepancies.csv -> recall/precision (axis 5)

Exits non-zero if either axis is not perfectly reproduced, so it can gate CI.

Usage:  python eval/run_benchmark.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from react_review.audit import ToleranceTable, compare_values

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "eval" / "benchmark_1"
TOL_CFG = ROOT / "configs" / "tolerances.yaml"


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _label_for(row: dict[str, str], tol: ToleranceTable) -> str:
    field_type = row["field_type"]
    result = compare_values(
        field_type=field_type,
        review_value=row.get("review_value"),
        source_value=row.get("source_value"),
        review_unit=row.get("unit", ""),
        source_unit=row.get("source_unit", ""),
        rel_tolerance=tol.rel_tolerance(field_type),
        sd_rel_tolerance=tol.sd_rel_tolerance(field_type),
        audit_id=row.get("audit_id") or row.get("seed_id", ""),
        study_id=row.get("study_id", ""),
        group=row.get("group", "-"),
    )
    return result.label.value


def score_audit(tol: ToleranceTable) -> bool:
    rows = _load(BENCH / "audit_template.csv")
    wrong = []
    for r in rows:
        got = _label_for(r, tol)
        if got != r["expected_label"].strip():
            wrong.append((r["audit_id"], r["expected_label"], got))
    dist = Counter(_label_for(r, tol) for r in rows)
    print(f"[axis 3] audit_template: {len(rows)} rows, "
          f"{len(rows) - len(wrong)} correct, {len(wrong)} wrong")
    print(f"          predicted distribution: {dict(dist)}")
    for aid, exp, got in wrong:
        print(f"          WRONG {aid}: expected {exp}, got {got}")
    return not wrong


def score_seeds(tol: ToleranceTable) -> bool:
    rows = _load(BENCH / "seeded_discrepancies.csv")
    tp = fp = tn = fn = 0
    wrong = []
    for r in rows:
        got = _label_for(r, tol)
        got_flag = got != "match"
        want_flag = r["should_flag"].strip().lower() == "yes"
        label_ok = got == r["expected_label"].strip()
        if got_flag and want_flag:
            tp += 1
        elif got_flag and not want_flag:
            fp += 1
        elif not got_flag and not want_flag:
            tn += 1
        else:
            fn += 1
        if not label_ok:
            wrong.append((r["seed_id"], r["expected_label"], got))
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    print(f"[axis 5] seeded_discrepancies: {len(rows)} rows | "
          f"TP={tp} FP={fp} TN={tn} FN={fn} | "
          f"recall={recall:.0%} precision={precision:.0%}")
    for sid, exp, got in wrong:
        print(f"          WRONG {sid}: expected {exp}, got {got}")
    return not wrong


def main() -> int:
    tol = ToleranceTable.from_yaml(TOL_CFG)
    print(f"tolerance: default {tol.rel_tolerance('*') * 100:.2f}%\n")
    ok_audit = score_audit(tol)
    print()
    ok_seeds = score_seeds(tol)
    print()
    if ok_audit and ok_seeds:
        print("BENCHMARK PASS — audit core reproduces the answer key exactly.")
        return 0
    print("BENCHMARK FAIL — see WRONG rows above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
