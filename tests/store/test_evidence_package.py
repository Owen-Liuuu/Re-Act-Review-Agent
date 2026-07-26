"""Tests for the JSON Evidence Package Store."""
from __future__ import annotations

import pytest

from react_review.core.enums import AuditLabel, ReportVerdict
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.audit import MatchResult
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.package import EvidencePackage
from react_review.schemas.report import AuditReport
from react_review.store import EvidencePackageStore


def _package(run_id: str = "run-001") -> EvidencePackage:
    return EvidencePackage(
        run_id=run_id,
        review_items=[
            ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                           field_type="eat_thickness", value="6.60 ± 0.71", unit="mm"),
        ],
        source_items=[
            SourceEvidenceItem(study_id="ahmad_2022", group="t1dm",
                               field_type="eat_thickness", source_value="6.60 ± 0.71",
                               source_unit="mm", source_quote="EFT was 6.60 ± 0.71 mm"),
        ],
        report=AuditReport(
            run_id=run_id, verdict=ReportVerdict.PASS, n_match=1,
            results=[MatchResult(study_id="ahmad_2022", field_type="eat_thickness",
                                 label=AuditLabel.MATCH)],
            summary="[PASS] 1 compared: 1 match",
        ),
        processing_records=[
            AgentRun(agent="collector",
                     steps=[StepRecord(index=0, thought="fetch", tool="fetch_fulltext",
                                       observation={"retrieved": True})],
                     status="finished"),
        ],
    )


def test_save_then_load_round_trips(tmp_path):
    store = EvidencePackageStore(tmp_path)
    pkg = _package()
    path = store.save(pkg)
    assert path.is_file()

    loaded = store.load("run-001")
    assert loaded.run_id == "run-001"
    assert loaded.review_items[0].value == "6.60 ± 0.71"
    assert loaded.source_items[0].source_quote == "EFT was 6.60 ± 0.71 mm"
    assert loaded.report.verdict == ReportVerdict.PASS
    assert loaded.report.results[0].label == AuditLabel.MATCH
    assert loaded.processing_records[0].agent == "collector"
    assert loaded.processing_records[0].steps[0].observation == {"retrieved": True}


def test_exists_and_list_runs(tmp_path):
    store = EvidencePackageStore(tmp_path)
    assert store.list_runs() == []
    assert not store.exists("run-001")
    store.save(_package("run-001"))
    store.save(_package("run-002"))
    assert store.exists("run-001")
    assert store.list_runs() == ["run-001", "run-002"]


def test_load_missing_raises(tmp_path):
    store = EvidencePackageStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("nope")


def test_save_is_run_scoped(tmp_path):
    store = EvidencePackageStore(tmp_path)
    store.save(_package("run-001"))
    assert (tmp_path / "run-001" / "package.json").is_file()
    # No stray temp file left behind after the atomic write.
    assert not list((tmp_path / "run-001").glob("*.tmp"))


def test_overwrite_replaces_cleanly(tmp_path):
    store = EvidencePackageStore(tmp_path)
    store.save(_package("run-001"))
    pkg2 = _package("run-001")
    pkg2.report.summary = "updated"
    store.save(pkg2)
    assert store.load("run-001").report.summary == "updated"
    assert store.list_runs() == ["run-001"]
