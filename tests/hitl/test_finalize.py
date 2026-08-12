"""Stopping a run records WHY, keeps the evidence, and happens exactly once.

Four ways out — finished, stopped, interrupted, failed — and the reason a
session owns all four is that they were separately written and separately
incomplete: the stop path recorded no telemetry, the reason was printed and
never written down, and a stop before the first paper left the directory with
nothing in it to distinguish a run that ended early from one that never ran.
"""
from __future__ import annotations

import json

import pytest

from react_review.production import FinalisationFailed, ProductionSession
from react_review.schemas.evidence import SourceEvidenceItem
from react_review.schemas.package import EvidencePackage
from react_review.schemas.telemetry import RunTelemetry
from react_review.store import EvidencePackageStore


def _store_with_partial(tmp_path) -> EvidencePackageStore:
    store = EvidencePackageStore(tmp_path)
    store.save_partial(EvidencePackage(
        run_id="r1", status="in_progress",
        source_items=[SourceEvidenceItem(study_id="ahmad_2022", field_type="bmi",
                                         source_value="24")],
    ))
    return store


def _session(store, telemetry=None) -> ProductionSession:
    return ProductionSession(store, "r1", telemetry=telemetry or RunTelemetry())


def _partial(tmp_path) -> dict:
    return json.loads((tmp_path / "r1" / "package.partial.json").read_text(
        encoding="utf-8"))


def test_stopping_stamps_the_status_and_keeps_the_evidence(tmp_path, capsys):
    _session(_store_with_partial(tmp_path)).finalise_stopped(
        stage="review_table_capture", reason="stopped by user at review_table_capture")

    data = _partial(tmp_path)
    assert data["status"] == "stopped_by_user"
    assert data["stopped_at_stage"] == "review_table_capture"
    assert len(data["source_items"]) == 1          # the evidence is still there

    out = capsys.readouterr().out
    assert "stopped_by_user" in out and "package.partial.json" in out


def test_the_reason_is_written_down_not_only_printed(tmp_path, capsys):
    """A day later the terminal is gone and the directory is what is left."""
    _session(_store_with_partial(tmp_path)).finalise_stopped(
        stage="collect_study", reason="stopped: the wrong PDF was attached")
    assert _partial(tmp_path)["stop_reason"] == "stopped: the wrong PDF was attached"


def test_a_stopped_run_still_reports_what_it_spent(tmp_path):
    """The cost was incurred whether or not the run got to the end."""
    telemetry = RunTelemetry()
    telemetry.record_call(prompt="p", output="o", seconds=1.5)
    _session(_store_with_partial(tmp_path), telemetry).finalise_stopped(
        stage="collect_study", reason="stopped")
    assert _partial(tmp_path)["telemetry"]["backend_requests"] == 1


def test_interrupt_is_recorded_distinctly_from_a_deliberate_stop(tmp_path, capsys):
    _session(_store_with_partial(tmp_path)).finalise_interrupted()
    data = _partial(tmp_path)
    assert data["status"] == "interrupted" and data["stopped_at_stage"] == ""


def test_a_crash_is_an_outcome_and_says_so(tmp_path, capsys):
    """`in_progress` claims the run is still going, which it is not."""
    _session(_store_with_partial(tmp_path)).finalise_error(
        RuntimeError("the retriever gave up"))
    data = _partial(tmp_path)
    assert data["status"] == "error"
    assert "RuntimeError: the retriever gave up" in data["stop_reason"]


def test_stopping_before_the_first_paper_still_leaves_an_artifact(tmp_path, capsys):
    """Otherwise an interrupt during parsing is indistinguishable from no run.

    There is no evidence to keep, so nothing claims there is — but the run id,
    the contract it was running under, why it ended and what it had already
    spent are all things a reader coming back to the directory needs.
    """
    from react_review.run_profile import ExecutionMode, RunManifest, load_run_contract

    session = _session(EvidencePackageStore(tmp_path))
    session.manifest = RunManifest.of(
        load_run_contract("configs/run_profiles/legacy.json"),
        ExecutionMode(extraction_mode="live"), context_source="cli")
    session.finalise_stopped(stage="review_pdf_loaded",
                             reason="stopped before any collection")

    data = _partial(tmp_path)
    assert data["run_id"] == "r1" and data["source_items"] == []
    assert data["status"] == "stopped_by_user"
    assert data["stopped_at_stage"] == "review_pdf_loaded"
    assert data["stop_reason"] == "stopped before any collection"
    assert data["run_manifest"]["contract"]["profile_id"] == "legacy"

    out = capsys.readouterr().out
    assert "ARTIFACTS" in out
    # No claim of "evidence collected so far", because none was.
    assert "evidence collected so far" not in out


# --- finalisation happens once ---------------------------------------------

def test_a_second_finalise_does_not_overwrite_the_first_reason(tmp_path):
    """An error while handling a stop is ordinary; the stop is the true reason."""
    session = _session(_store_with_partial(tmp_path))
    session.finalise_stopped(stage="collect_study", reason="stopped by user")
    session.finalise_error(RuntimeError("and then the store went away"))

    data = _partial(tmp_path)
    assert data["status"] == "stopped_by_user"
    assert data["stop_reason"] == "stopped by user"


def test_finalising_twice_does_not_double_the_books(tmp_path):
    """Cache totals are a snapshot, so repeating one writes the same numbers."""

    class _Cache:
        hits, misses = 3, 1

        def save(self):
            return None

    telemetry = RunTelemetry()
    session = ProductionSession(tmp_path and _store_with_partial(tmp_path), "r1",
                                telemetry=telemetry, extraction_cache=_Cache())
    session.finalise_stopped(stage="collect_study", reason="stopped")
    first = (telemetry.cache_hits, telemetry.cache_misses)
    session.finalise_interrupted()
    assert (telemetry.cache_hits, telemetry.cache_misses) == first == (3, 1)


# --- the finished package ----------------------------------------------------

def test_the_finished_package_is_saved_once_and_returned_from_disk(tmp_path):
    """What the report renders is the file, so the file is what is returned."""
    store = EvidencePackageStore(tmp_path)
    telemetry = RunTelemetry()
    telemetry.record_batch(claims=4, failed=False)
    session = ProductionSession(store, "r1", telemetry=telemetry)

    loaded = session.finalise_success(EvidencePackage(run_id="r1"))
    assert loaded.telemetry.batch.claims == 4          # the run's own totals
    assert store.package_path("r1").is_file()
    assert session.outcome == "complete"


def test_the_time_the_run_took_is_in_the_file_not_only_in_the_object(tmp_path):
    """It was measured and then written before the clock stopped.

    The package was saved from inside the clock, so `wall_seconds` on disk was
    always exactly 0.0 — a run could report its wall time on the terminal and
    then persist a file saying it took no time at all.
    """
    store = EvidencePackageStore(tmp_path)
    telemetry = RunTelemetry()
    session = ProductionSession(store, "r1", telemetry=telemetry)
    with session:
        for _ in range(200_000):
            pass
        loaded = session.finalise_success(EvidencePackage(run_id="r1"))
    assert loaded.telemetry.wall_seconds > 0
    assert store.load("r1").telemetry.wall_seconds > 0


def test_the_clock_stops_before_the_report_is_rendered(tmp_path):
    """Rendering a finished package is not part of what the audit cost."""
    telemetry = RunTelemetry()
    session = ProductionSession(EvidencePackageStore(tmp_path), "r1",
                                telemetry=telemetry)
    with session:
        session.finalise_success(EvidencePackage(run_id="r1"))
        measured = telemetry.wall_seconds
        for _ in range(200_000):                       # stand-in for rendering
            pass
    assert telemetry.wall_seconds == measured


def test_a_package_that_cannot_be_read_back_is_not_reported_as_complete(tmp_path):
    """Saving is not the promise; being loadable again is."""

    class _Unreadable(EvidencePackageStore):
        def load(self, run_id):
            raise ValueError("claims_per_batch")

    session = ProductionSession(_Unreadable(tmp_path), "r1",
                                telemetry=RunTelemetry())
    with pytest.raises(FinalisationFailed, match="could not be read back"):
        session.finalise_success(EvidencePackage(run_id="r1"))
    assert session.outcome != "complete"


def test_a_failure_after_finalising_does_not_restate_a_good_run_as_failed(tmp_path):
    """A report that will not render is a report failure, not an audit failure."""
    store = EvidencePackageStore(tmp_path)
    session = ProductionSession(store, "r1", telemetry=RunTelemetry())
    session.finalise_success(EvidencePackage(run_id="r1"))

    session.finalise_error(RuntimeError("the HTML template is broken"))
    assert store.load("r1").status == "complete"
    assert not (tmp_path / "r1" / "package.partial.json").is_file()
