"""Tests for the CSV loaders and the `audit` CLI subcommand (end-to-end)."""
from __future__ import annotations

from pathlib import Path

from react_review.cli import _audit_main
from react_review.core.enums import ReportVerdict
from react_review.csv_io import load_review_items, load_source_items
from react_review.store import EvidencePackageStore

BENCH = Path(__file__).resolve().parents[1] / "eval" / "benchmark" / "audit_template.csv"


def test_csv_loaders_read_benchmark():
    review = load_review_items(BENCH)
    source = load_source_items(BENCH)
    assert len(review) == 57
    assert len(source) == 57
    # review side reads review_value/unit; source side reads source_value/source_unit
    a003_r = next(r for r in review if r.review_data_id == "A003")
    assert a003_r.field_type == "bmi" and a003_r.value == "20.57 ± 1.7"
    a022_s = source[[i for i, s in enumerate(source)
                     if s.study_id == "keles_2016" and s.group == "t1dm"
                     and s.field_type == "eat_thickness"][0]]
    assert a022_s.review_data_id == "A022"
    assert a022_s.source_unit == "cm"


def test_audit_cli_end_to_end(tmp_path, capsys):
    # Point both review and source at the benchmark table (loaders split it).
    _audit_main([
        str(BENCH), str(BENCH),
        "--out", str(tmp_path), "--run-id", "t1",
    ])
    out = capsys.readouterr().out
    assert "52 match" in out and "1 mismatch" in out and "4 unit_mismatch" in out

    # The evidence package was persisted and round-trips.
    pkg = EvidencePackageStore(tmp_path).load("t1")
    assert pkg.report.verdict == ReportVerdict.FAIL       # a mismatch present
    assert pkg.report.n_match == 52
    assert pkg.report.n_mismatch == 1
    assert pkg.report.n_unit_mismatch == 4
    assert len(pkg.review_items) == 57 and len(pkg.source_items) == 57
