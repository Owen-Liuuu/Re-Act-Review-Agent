"""End-to-end deterministic pipeline on the benchmark.

Treats ``audit_template.csv`` as two separate tables (the review side and the
source side, as the parser/collector would produce them), feeds them to the
AuditOrchestrator, and prints the aggregated report. This exercises the full P1
chain: match (4-tuple join) -> compare_values tool -> aggregate.

Usage:  python eval/run_pipeline.py
"""
from __future__ import annotations

import asyncio
import csv
from pathlib import Path

from react_review.audit import ToleranceTable
from react_review.core.config import AppConfig
from react_review.orchestrator import AuditOrchestrator
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.tools import build_catalogue

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "eval" / "benchmark_1"
TOL_CFG = ROOT / "configs" / "tolerances.yaml"


def load_tables(path: Path) -> tuple[list[ReviewDataItem], list[SourceEvidenceItem]]:
    """Split audit_template.csv into a review table and a source table."""
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    review = [
        ReviewDataItem(
            review_data_id=r.get("audit_id", ""),
            study_id=r["study_id"],
            group=r.get("group", "-"),
            field_type=r["field_type"],
            value=r.get("review_value"),
            unit=r.get("unit", ""),
        )
        for r in rows
    ]
    source = [
        SourceEvidenceItem(
            study_id=r["study_id"],
            group=r.get("group", "-"),
            field_type=r["field_type"],
            source_value=r.get("source_value"),
            source_unit=r.get("source_unit", ""),
            source_quote=r.get("source_quote", ""),
            source_location_in_paper=r.get("source_location_in_paper", ""),
        )
        for r in rows
    ]
    return review, source


async def main() -> int:
    tol = ToleranceTable.from_yaml(TOL_CFG)
    catalogue = build_catalogue(AppConfig(mock_mode=True), tolerance=tol)
    orch = AuditOrchestrator(catalogue)

    review, source = load_tables(BENCH / "audit_template.csv")
    report = await orch.run(review, source, run_id="bench")

    print(report.summary)
    print(f"  verdict: {report.verdict.value}")
    print(f"  pairs compared: {len(report.results)} "
          f"(unmatched: {len(report.unmatched_review)} review / "
          f"{len(report.unmatched_source)} source)")
    for r in report.results:
        if r.label.value != "match":
            print(f"  [{r.label.value}] {r.study_id}/{r.group}/{r.field_type}: {r.reason}")

    ok = (report.n_match, report.n_mismatch, report.n_unit_mismatch) == (52, 1, 4)
    print("\nPIPELINE PASS" if ok else "\nPIPELINE FAIL (expected 52/1/4)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
