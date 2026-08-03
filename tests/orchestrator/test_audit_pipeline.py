"""End-to-end (mock) test: Collector -> Auditor -> Judge -> EvidencePackage."""
from __future__ import annotations

import json

import pytest

from react_review.checklist.schema import ChecklistApplication
from react_review.agents.collector import Collector
from react_review.audit import ToleranceTable
from react_review.core.enums import ReportVerdict
from react_review.llm.base import LLMBackend
from react_review.orchestrator import AuditOrchestrator, AuditPipeline, Judge
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.knowledge import KnowledgeImportRecord
from react_review.schemas.resolution import FieldResolutionRecord
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.store import EvidencePackageStore
from react_review.tools.compare import CompareValuesTool
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry


class _KeyedBackend(LLMBackend):
    """Returns a source value that depends on the field_type in the prompt."""

    def __init__(self, by_field: dict[str, dict]) -> None:
        super().__init__()
        self._by_field = by_field

    @property
    def model_id(self) -> str:
        return "keyed"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        for ft, payload in self._by_field.items():
            if ft in prompt:
                return json.dumps(payload)
        return json.dumps({"found": False, "value": None})


class _DocRetriever(PaperRetriever):
    async def retrieve(self, reference):
        return PaperDocument(paper_id=reference.doi or "x", reference=reference,
                             full_text=("source paper text. EFT 6.60 ± 0.71 mm. "
                                        "EFT 0.7 cm."))


def _pipeline(extract_backend):
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(_DocRetriever()))
    reg.register(ExtractSourceValueTool(extract_backend))
    reg.register(CompareValuesTool(ToleranceTable()))
    collector = Collector(reg)
    auditor = AuditOrchestrator(reg)
    return AuditPipeline(collector, auditor, Judge())


_REF = lambda sid: ReferenceEntry(title=sid, doi="10.1/" + sid)


@pytest.mark.asyncio
async def test_pipeline_match_pass(tmp_path):
    field_resolution = FieldResolutionRecord(
        resolution_key="resolution-eat", raw_field_name="EFT/ EAT", unit="mm",
        field_type="eat_thickness", status="authoritative", source="deterministic")
    knowledge_import = KnowledgeImportRecord(
        source="ontology:labs", path="labs.json", sha256="hash", added=3)
    checklist = ChecklistApplication(
        name="clinical", version="1", source_file="checklist.yaml", sha256="check-hash")
    review = [
        ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                       field_type="eat_thickness", value="6.60 ± 0.71", unit="mm",
                       resolution_key="resolution-eat"),
    ]
    backend = _KeyedBackend({"eat_thickness": {
        "found": True, "value": "6.60 ± 0.71", "unit": "mm",
        "quote": "EFT 6.60 ± 0.71 mm", "source_field_name": "EFT", "location": "Table 2"}})
    store = EvidencePackageStore(tmp_path)
    pipe = _pipeline(backend)
    pipe._store = store

    pkg = await pipe.run(
        review, _REF, research_context="EAT in T1DM", run_id="r1",
        field_resolutions=[field_resolution], knowledge_imports=[knowledge_import],
        knowledge_fingerprint="effective-kb-hash", knowledge_concept_count=12,
        checklist=checklist)

    assert pkg.report.n_match == 1
    assert pkg.final_verification.verdict == ReportVerdict.PASS
    assert pkg.final_verification.human_review_flags == []
    assert pkg.source_items[0].source_value == "6.60 ± 0.71"
    assert pkg.field_resolutions[0].resolution_key == "resolution-eat"
    assert pkg.knowledge_imports[0].source == "ontology:labs"
    assert pkg.knowledge_fingerprint == "effective-kb-hash"
    assert pkg.knowledge_concept_count == 12
    assert pkg.checklist.sha256 == "check-hash"
    # persisted + round-trips
    loaded = store.load("r1")
    assert loaded.final_verification.verdict == ReportVerdict.PASS
    assert loaded.field_resolutions[0].field_type == "eat_thickness"
    assert loaded.knowledge_imports[0].added == 3
    assert loaded.knowledge_fingerprint == "effective-kb-hash"
    assert loaded.knowledge_concept_count == 12
    assert loaded.checklist.name == "clinical"
    # a collector processing record is attached
    assert any(r.agent == "collector" for r in pkg.processing_records)


@pytest.mark.asyncio
async def test_pipeline_unit_mismatch_flags_human_review():
    review = [
        ReviewDataItem(study_id="keles_2016", group="t1dm",
                       field_type="eat_thickness", value="0.7", unit="mm"),
    ]
    backend = _KeyedBackend({"eat_thickness": {
        "found": True, "value": "0.7", "unit": "cm",   # unit differs -> unit_mismatch
        "quote": "EFT 0.7 cm", "source_field_name": "EFT", "location": "Table 2"}})
    pkg = await _pipeline(backend).run(review, _REF, run_id="r2")

    assert pkg.report.n_unit_mismatch == 1
    assert pkg.final_verification.verdict == ReportVerdict.PARTIAL
    flags = pkg.final_verification.human_review_flags
    assert len(flags) == 1
    assert flags[0].label == "unit_mismatch"
    assert flags[0].study_id == "keles_2016"


@pytest.mark.asyncio
async def test_pipeline_missing_source_flags_review():
    review = [
        ReviewDataItem(study_id="x_2020", group="t1dm",
                       field_type="bmi", value="24.0", unit="kg/m2"),
    ]
    backend = _KeyedBackend({"bmi": {
        "found": False, "value": None, "unit": "kg/m3",
        "not_found_reason": "the paper does not report BMI",
    }})
    pkg = await _pipeline(backend).run(review, _REF, run_id="r3")

    # source not found -> not_comparable -> flagged for human review
    assert pkg.source_items[0].source_value is None
    assert pkg.source_items[0].source_unit == "kg/m3"  # retained provenance, not a verdict
    assert pkg.report.n_unit_mismatch == 0
    assert pkg.report.n_not_comparable == 1
    flags = pkg.final_verification.human_review_flags
    assert len(flags) == 1
    assert flags[0].label == "missing_source"
