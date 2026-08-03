"""Tests for the JSON Evidence Package Store."""
from __future__ import annotations

import pytest

from react_review.checklist.schema import (
    ChecklistApplication,
    ChecklistAssessment,
    ChecklistGap,
    ChecklistItem,
)
from react_review.core.enums import AuditLabel, ReportVerdict
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.audit import MatchResult
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.knowledge import KnowledgeImportRecord
from react_review.schemas.package import EvidencePackage
from react_review.schemas.reason import ReasonRecord
from react_review.schemas.report import AuditReport
from react_review.schemas.resolution import (
    FieldResolutionRecord,
    ResolutionAttempt,
    ResolutionCellRef,
)
from react_review.store import EvidencePackageStore


def _package(run_id: str = "run-001") -> EvidencePackage:
    return EvidencePackage(
        run_id=run_id,
        review_items=[
            ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                           field_type="eat_thickness", value="6.60 ± 0.71", unit="mm",
                           resolution_key="resolution-1"),
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
        field_resolutions=[
            FieldResolutionRecord(
                resolution_key="resolution-1", raw_field_name="EFT/ EAT", unit="mm",
                field_type="eat_thickness", status="candidate", source="retrieval_llm",
                grounded_on=["eat_thickness"], confidence=0.91,
                checks={"grounding": True, "unit": True},
                reasons=[ReasonRecord(code="candidate_mapping", stage="field_resolution")],
                attempts=[ResolutionAttempt(
                    seed=42, model_id="fixture", field_type="eat_thickness",
                    prompt_sha256="prompt-hash", response_sha256="response-hash")],
                stability="stable", consensus_count=2,
                candidate_names=["eat_thickness"],
                affected_cells=[ResolutionCellRef(
                    table_id="table_1", cell_ref=(0, 2), study_id="ahmad_2022",
                    column_header="EFT/ EAT", status="candidate",
                    field_type="eat_thickness")],
                statuses_seen=["candidate"], field_types_seen=["eat_thickness"],
            ),
        ],
        knowledge_imports=[KnowledgeImportRecord(
            source="ontology:labs", path="configs/ontology/labs.json",
            sha256="ontology-hash", concepts_before=9, concepts_after=12,
            added=3, added_field_types=[
                "hba1c", "ldl_cholesterol", "fasting_glucose"],
        )],
        knowledge_fingerprint="effective-kb-hash",
        knowledge_concept_count=12,
        checklist=ChecklistApplication(
            name="clinical", version="1", source_file="checklist.yaml",
            sha256="checklist-hash",
            items=[ChecklistItem(
                id="risk", question="Risk assessed?", required=True,
                aliases=["risk of bias"])],
            assessments=[ChecklistAssessment(
                checklist_id="risk", question="Risk assessed?", required=True,
                status="missing_required", expected=1, found=0)],
            gaps=[ChecklistGap(
                checklist_id="risk", question="Risk assessed?", scope="review",
                reason="required checklist item was not found")],
        ),
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
    assert loaded.review_items[0].resolution_key == "resolution-1"
    resolution = loaded.field_resolutions[0]
    assert resolution.checks == {"grounding": True, "unit": True}
    assert resolution.attempts[0].prompt_sha256 == "prompt-hash"
    assert resolution.stability == "stable" and resolution.consensus_count == 2
    assert resolution.candidate_names == ["eat_thickness"]
    assert resolution.affected_cells[0].cell_ref == (0, 2)
    ontology = loaded.knowledge_imports[0]
    assert ontology.source == "ontology:labs"
    assert ontology.added == 3 and ontology.sha256 == "ontology-hash"
    assert loaded.knowledge_fingerprint == "effective-kb-hash"
    assert loaded.knowledge_concept_count == 12
    assert loaded.checklist.sha256 == "checklist-hash"
    assert loaded.checklist.items[0].aliases == ["risk of bias"]
    assert loaded.checklist.gaps[0].checklist_id == "risk"


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
