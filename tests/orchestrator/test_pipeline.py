"""Tests for the deterministic audit orchestrator (incl. end-to-end benchmark)."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from react_review.audit import ToleranceTable
from react_review.core.config import AppConfig
from react_review.core.enums import ReportVerdict
from react_review.orchestrator import AuditOrchestrator
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.tools import build_catalogue

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "eval" / "benchmark"


def _orch() -> AuditOrchestrator:
    tol = ToleranceTable.from_yaml(ROOT / "configs" / "tolerances.yaml")
    return AuditOrchestrator(build_catalogue(AppConfig(mock_mode=True), tolerance=tol))


def _rv(study, group, ft, value, unit=""):
    return ReviewDataItem(study_id=study, group=group, field_type=ft, value=value, unit=unit)


def _sv(study, group, ft, value, unit=""):
    return SourceEvidenceItem(study_id=study, group=group, field_type=ft,
                              source_value=value, source_unit=unit)


@pytest.mark.asyncio
async def test_verdict_fail_on_mismatch():
    report = await _orch().run(
        [_rv("s1", "t1dm", "sample_size", "85")],
        [_sv("s1", "t1dm", "sample_size", "100")],
    )
    assert report.verdict == ReportVerdict.FAIL
    assert report.n_mismatch == 1


@pytest.mark.asyncio
async def test_verdict_partial_on_unit_only():
    report = await _orch().run(
        [_rv("s1", "t1dm", "eat_thickness", "0.7", unit="mm")],
        [_sv("s1", "t1dm", "eat_thickness", "0.7", unit="cm")],
    )
    assert report.verdict == ReportVerdict.PARTIAL
    assert report.n_unit_mismatch == 1


@pytest.mark.asyncio
async def test_missing_source_value_precedes_residual_source_unit():
    report = await _orch().run(
        [_rv("s1", "t1dm", "eat_thickness", "0.7", unit="mm")],
        [_sv("s1", "t1dm", "eat_thickness", None, unit="cm")],
    )
    assert report.n_unit_mismatch == 0
    assert report.n_not_comparable == 1
    assert report.results[0].reason == (
        "the source value is missing; units were not compared")


@pytest.mark.asyncio
async def test_verdict_pass_when_all_match():
    report = await _orch().run(
        [_rv("s1", "t1dm", "age", "34", unit="years")],
        [_sv("s1", "t1dm", "age", "34", unit="years")],
    )
    assert report.verdict == ReportVerdict.PASS
    assert report.n_match == 1


@pytest.mark.asyncio
async def test_all_not_comparable_is_incomplete_not_pass():
    # Both values unparseable -> not_comparable. Nothing was verified, so the
    # verdict must NOT be PASS (regression guard for the B1 bug).
    report = await _orch().run(
        [_rv("s1", "t1dm", "age", "not reported")],
        [_sv("s1", "t1dm", "age", "also missing")],
    )
    assert report.n_match == 0 and report.n_not_comparable == 1
    assert report.verdict == ReportVerdict.INCOMPLETE


@pytest.mark.asyncio
async def test_partial_when_some_pairs_not_comparable():
    report = await _orch().run(
        [_rv("s1", "t1dm", "age", "34"), _rv("s1", "t1dm", "bmi", "N/R")],
        [_sv("s1", "t1dm", "age", "34"), _sv("s1", "t1dm", "bmi", "missing")],
    )
    assert report.n_match == 1 and report.n_not_comparable == 1
    assert report.verdict == ReportVerdict.PARTIAL


@pytest.mark.asyncio
async def test_unmatched_items_are_flagged_not_dropped():
    report = await _orch().run(
        [_rv("s1", "t1dm", "age", "34")],       # no source
        [_sv("s1", "t1dm", "bmi", "24")],       # no review
    )
    assert report.verdict == ReportVerdict.INCOMPLETE  # nothing comparable
    assert len(report.unmatched_review) == 1
    assert len(report.unmatched_source) == 1
    assert any("no source evidence" in f for f in report.flags)


@pytest.mark.asyncio
async def test_end_to_end_reproduces_benchmark():
    with (BENCH / "audit_template.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    review = [_rv(r["study_id"], r["group"], r["field_type"], r["review_value"], r.get("unit", ""))
              for r in rows]
    source = [_sv(r["study_id"], r["group"], r["field_type"], r["source_value"], r.get("source_unit", ""))
              for r in rows]

    report = await _orch().run(review, source, run_id="bench")
    assert not report.unmatched_review and not report.unmatched_source
    assert (report.n_match, report.n_mismatch, report.n_unit_mismatch) == (52, 1, 4)
    assert report.verdict == ReportVerdict.FAIL
