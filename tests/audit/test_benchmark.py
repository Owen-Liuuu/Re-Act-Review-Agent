"""CI gate: the audit core must reproduce the benchmark answer key exactly.

Loads the hand-labelled benchmark CSVs and asserts compare_values reproduces
every ``expected_label`` (axis 3) and catches every seeded positive without
false positives (axis 5).
"""
from __future__ import annotations

import csv
from pathlib import Path

from react_review.audit import ToleranceTable, compare_values

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "eval" / "benchmark_1"
TOL = ToleranceTable.from_yaml(ROOT / "configs" / "tolerances.yaml")


def _load(name: str) -> list[dict[str, str]]:
    with (BENCH / name).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _predict(row: dict[str, str]) -> str:
    ft = row["field_type"]
    return compare_values(
        field_type=ft,
        review_value=row.get("review_value"),
        source_value=row.get("source_value"),
        review_unit=row.get("unit", ""),
        source_unit=row.get("source_unit", ""),
        rel_tolerance=TOL.rel_tolerance(ft),
        sd_rel_tolerance=TOL.sd_rel_tolerance(ft),
    ).label.value


def test_audit_template_reproduced_exactly():
    rows = _load("audit_template.csv")
    wrong = [
        (r["audit_id"], r["expected_label"], _predict(r))
        for r in rows
        if _predict(r) != r["expected_label"].strip()
    ]
    assert not wrong, f"mislabelled audit rows: {wrong}"
    # Known distribution under the dual band (mean 1% + SD 3%).
    labels = [_predict(r) for r in rows]
    assert labels.count("match") == 52
    assert labels.count("mismatch") == 1
    assert labels.count("unit_mismatch") == 4
    # A003 is the SD-driven mismatch (mean identical, SD 1.7 vs 1.77 = 3.95%).
    a003 = next(r for r in rows if r["audit_id"] == "A003")
    assert _predict(a003) == "mismatch"


def test_seeded_discrepancies_full_recall_and_precision():
    rows = _load("seeded_discrepancies.csv")
    tp = fp = fn = 0
    for r in rows:
        got_flag = _predict(r) != "match"
        want_flag = r["should_flag"].strip().lower() == "yes"
        if got_flag and want_flag:
            tp += 1
        elif got_flag and not want_flag:
            fp += 1
        elif not got_flag and want_flag:
            fn += 1
    assert fn == 0, "missed a seeded positive (recall < 100%)"
    assert fp == 0, "flagged a negative control (precision < 100%)"
    assert tp == 8

    # Every seed's exact label must also match.
    wrong = [
        (r["seed_id"], r["expected_label"], _predict(r))
        for r in rows
        if _predict(r) != r["expected_label"].strip()
    ]
    assert not wrong, f"mislabelled seeds: {wrong}"
