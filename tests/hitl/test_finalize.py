"""Stopping a run records WHY and keeps the evidence already collected."""
from __future__ import annotations

import json

from react_review.cli import _finalize_partial
from react_review.schemas.evidence import SourceEvidenceItem
from react_review.schemas.package import EvidencePackage
from react_review.store import EvidencePackageStore


def _store_with_partial(tmp_path) -> EvidencePackageStore:
    store = EvidencePackageStore(tmp_path)
    store.save_partial(EvidencePackage(
        run_id="r1", status="in_progress",
        source_items=[SourceEvidenceItem(study_id="ahmad_2022", field_type="bmi",
                                         source_value="24")],
    ))
    return store


def test_stopping_stamps_the_status_and_keeps_the_evidence(tmp_path, capsys):
    store = _store_with_partial(tmp_path)
    _finalize_partial(store, "r1", "stopped_by_user", "review_table_capture",
                      "stopped by user at review_table_capture")

    data = json.loads((tmp_path / "r1" / "package.partial.json").read_text(encoding="utf-8"))
    assert data["status"] == "stopped_by_user"
    assert data["stopped_at_stage"] == "review_table_capture"
    assert len(data["source_items"]) == 1          # the evidence is still there

    out = capsys.readouterr().out
    assert "stopped_by_user" in out and "package.partial.json" in out


def test_interrupt_is_recorded_distinctly_from_a_deliberate_stop(tmp_path, capsys):
    store = _store_with_partial(tmp_path)
    _finalize_partial(store, "r1", "interrupted", "", "interrupted (Ctrl-C)")
    data = json.loads((tmp_path / "r1" / "package.partial.json").read_text(encoding="utf-8"))
    assert data["status"] == "interrupted" and data["stopped_at_stage"] == ""


def test_finalize_without_a_partial_still_reports_where_to_look(tmp_path, capsys):
    # Stopped at the very first checkpoint: no evidence yet, but the journal is there.
    _finalize_partial(EvidencePackageStore(tmp_path), "r1", "stopped_by_user",
                      "review_pdf_loaded", "stopped before any collection")
    out = capsys.readouterr().out
    assert "ARTIFACTS" in out and "PARTIAL" not in out
