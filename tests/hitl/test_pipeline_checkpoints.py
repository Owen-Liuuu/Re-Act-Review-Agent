"""The audit pipeline gates per SOURCE PAPER and persists progress as it goes."""
from __future__ import annotations

import json

import pytest

from react_review.checklist.schema import ChecklistApplication
from react_review.audit import ToleranceTable
from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.core.exceptions import RunStopped
from react_review.hitl import (
    Decision,
    RunJournal,
    ScriptedCheckpoint,
    StepReporter,
    StepStage,
    SubjectKind,
)
from react_review.orchestrator import AuditOrchestrator, AuditPipeline, Judge
from react_review.schemas.agent import AgentRun
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.knowledge import KnowledgeImportRecord
from react_review.schemas.resolution import FieldResolutionRecord
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.store import EvidencePackageStore
from react_review.tools.compare import CompareValuesTool
from react_review.tools.registry import ToolRegistry


class _StubCollector:
    """Echoes the review value back as the source value (always a match)."""

    def __init__(self) -> None:
        self.calls = 0

    async def collect(self, review_item, reference, *, research_context=""):
        self.calls += 1
        from react_review.agents.collector import CollectResult
        return CollectResult(
            source_item=SourceEvidenceItem(
                study_id=review_item.study_id, group=review_item.group,
                field_type=review_item.field_type, source_value=review_item.value,
                source_unit=review_item.unit, collection_outcome=CollectionOutcome.FOUND,
            ),
            record=AgentRun(agent="collector"),
            decision=ReflectionDecision.ACCEPT,
        )


def _items() -> list[ReviewDataItem]:
    # two studies, two claims each — interleaved to prove grouping preserves order
    return [
        ReviewDataItem(study_id="ahmad_2022", group="t1dm", field_type="bmi", value="24"),
        ReviewDataItem(study_id="keles_2016", group="t1dm", field_type="bmi", value="25"),
        ReviewDataItem(study_id="ahmad_2022", group="control", field_type="bmi", value="22"),
    ]


def _pipeline(tmp_path, decisions):
    reg = ToolRegistry()
    reg.register(CompareValuesTool(ToleranceTable()))
    gate = ScriptedCheckpoint(decisions)
    reporter = StepReporter("run1", gate=gate, journal=RunJournal(tmp_path / "run1"))
    store = EvidencePackageStore(tmp_path)
    pipe = AuditPipeline(_StubCollector(), AuditOrchestrator(reg), Judge(),
                         store=store, reporter=reporter)
    return pipe, gate, store


@pytest.mark.asyncio
async def test_every_paper_reports_but_the_batch_is_reviewed_once(tmp_path):
    pipe, gate, _ = _pipeline(tmp_path, [])
    await pipe.run(_items(), lambda sid: ReferenceEntry(title=sid), run_id="run1")

    stages = [e.stage for e in gate.seen]
    # every paper still REPORTS in full (2 studies, 3 claims)…
    assert stages.count(StepStage.COLLECT_STUDY) == 2
    # …but the collection is reviewed once, after all of it is in
    assert stages.count(StepStage.COLLECTION_REVIEW) == 1
    assert stages[-3:] == [StepStage.COLLECTION_REVIEW,
                           StepStage.AUDIT_SUMMARY, StepStage.JUDGE_FLAGS]
    # grouping preserves first-appearance order
    assert [e.payload["study_id"] for e in gate.seen
            if e.stage is StepStage.COLLECT_STUDY] == ["ahmad_2022", "keles_2016"]


async def _blocking_stages(n_studies: int) -> list[StepStage]:
    """Which stages the console actually PAUSES on for a review of n papers."""
    from react_review.hitl import CheckpointPolicy, ConsoleCheckpoint

    gate = ConsoleCheckpoint(CheckpointPolicy.key_stages())
    gate._read_key = lambda: "c"                       # type: ignore[method-assign]
    blocked: list[StepStage] = []
    ask = gate._ask                                    # wrap the INSTANCE, not the class

    async def counting_ask(event):
        blocked.append(event.stage)
        return await ask(event)

    gate._ask = counting_ask                           # type: ignore[method-assign]

    reg = ToolRegistry()
    reg.register(CompareValuesTool(ToleranceTable()))
    pipe = AuditPipeline(_StubCollector(), AuditOrchestrator(reg), Judge(),
                         reporter=StepReporter("r", gate=gate))
    items = [ReviewDataItem(study_id=f"s{i}", group="t1dm", field_type="bmi", value="24")
             for i in range(n_studies)]
    await pipe.run(items, lambda sid: ReferenceEntry(title=sid), run_id="r")
    return blocked


@pytest.mark.asyncio
async def test_pause_count_does_not_grow_with_the_number_of_papers(capsys):
    # A review may include 9 source papers or 80; the reviewer presses the same
    # number of keys either way. This is the generalisation requirement.
    few = await _blocking_stages(2)
    many = await _blocking_stages(20)
    capsys.readouterr()                                   # discard the rendered blocks

    assert few == many
    assert StepStage.COLLECT_STUDY not in few             # shown, never gated
    assert few == [StepStage.COLLECTION_REVIEW, StepStage.AUDIT_SUMMARY,
                   StepStage.JUDGE_FLAGS]


@pytest.mark.asyncio
async def test_stopping_at_a_study_keeps_the_evidence_already_collected(tmp_path):
    pipe, _, store = _pipeline(tmp_path, [Decision.CONTINUE, Decision.STOP])
    field_resolution = FieldResolutionRecord(
        resolution_key="resolution-bmi", raw_field_name="BMI", field_type="bmi",
        status="authoritative", source="deterministic")
    knowledge_import = KnowledgeImportRecord(
        source="ontology:labs", path="labs.json", sha256="hash", added=3)
    checklist = ChecklistApplication(
        name="clinical", version="1", source_file="checklist.yaml", sha256="check-hash")
    with pytest.raises(RunStopped):
        await pipe.run(
            _items(), lambda sid: ReferenceEntry(title=sid), run_id="run1",
            field_resolutions=[field_resolution], knowledge_imports=[knowledge_import],
            knowledge_fingerprint="effective-kb-hash", knowledge_concept_count=12,
            checklist=checklist)

    partial = tmp_path / "run1" / "package.partial.json"
    assert partial.is_file()
    data = json.loads(partial.read_text(encoding="utf-8"))
    assert data["status"] == "in_progress"
    assert len(data["source_items"]) == 3          # both groups were collected
    assert data["field_resolutions"][0]["resolution_key"] == "resolution-bmi"
    assert data["knowledge_imports"][0]["source"] == "ontology:labs"
    assert data["knowledge_fingerprint"] == "effective-kb-hash"
    assert data["knowledge_concept_count"] == 12
    assert data["checklist"]["sha256"] == "check-hash"
    assert not (tmp_path / "run1" / "package.json").is_file()   # never finalised


@pytest.mark.asyncio
async def test_full_run_writes_the_final_package(tmp_path):
    pipe, _, store = _pipeline(tmp_path, [])
    pkg = await pipe.run(_items(), lambda sid: ReferenceEntry(title=sid), run_id="run1")
    assert pkg.status == "complete"
    assert store.exists("run1")
    assert pkg.report.n_match == 3


@pytest.mark.asyncio
async def test_collect_event_names_the_paper_and_shows_every_value(tmp_path):
    pipe, gate, _ = _pipeline(tmp_path, [])
    await pipe.run(_items(), lambda sid: ReferenceEntry(title=sid, doi="10.1/x"),
                   run_id="run1")
    first = next(e for e in gate.seen if e.stage is StepStage.COLLECT_STUDY)
    assert first.subject == "doi:10.1/x"                     # which paper
    assert first.subject_kind is SubjectKind.SOURCE_PDF
    assert first.title.startswith("ahmad_2022 ·")
    assert "unknown source" in first.title
    assert len(first.payload["evidence"]) == 2               # full content, both claims
    assert "review '24'" in first.render_blocks[0]


def test_collect_title_distinguishes_pmc_abstract_and_unretrieved():
    from types import SimpleNamespace

    from react_review.core.enums import CollectionOutcome
    from react_review.orchestrator.audit_pipeline import AuditPipeline
    from react_review.steps.data_extraction.schemas import DocumentScope

    pmc_item = SourceEvidenceItem(
        study_id="svanteson_2019", group="t1dm", field_type="bmi",
        retriever_kind="pmc", document_scope=DocumentScope.FULL_TEXT,
        collection_outcome=CollectionOutcome.FOUND)
    pmc_source = SimpleNamespace(
        retrieved=True,
        document=SimpleNamespace(full_text="x" * 17619, document_scope=DocumentScope.FULL_TEXT),
        provenance={"retriever_kind": "pmc"},
    )
    assert AuditPipeline._collect_title(
        "svanteson_2019", [pmc_item], pmc_source
    ) == "svanteson_2019 · pmc · 17,619 chars · full_text     (hit:PMC esearch by DOI)"

    abs_item = SourceEvidenceItem(
        study_id="ahmad_2022", group="t1dm", field_type="bmi",
        retriever_kind="pubmed_abstract", document_scope=DocumentScope.ABSTRACT_ONLY,
        collection_outcome=CollectionOutcome.FOUND)
    abs_source = SimpleNamespace(
        retrieved=True,
        document=SimpleNamespace(
            full_text="y" * 2041, document_scope=DocumentScope.ABSTRACT_ONLY),
        provenance={"retriever_kind": "pubmed_abstract"},
    )
    assert AuditPipeline._collect_title(
        "ahmad_2022", [abs_item], abs_source
    ) == "ahmad_2022 · pubmed_abstract · 2,041 chars · abstract_only     (fallback: DOI miss → title search)"

    abs_doi_source = SimpleNamespace(
        retrieved=True,
        document=SimpleNamespace(
            full_text="y" * 2041, document_scope=DocumentScope.ABSTRACT_ONLY),
        provenance={"retriever_kind": "pubmed_abstract", "source_doi": "10.1/x"},
    )
    assert "fallback: DOI miss → PubMed abstract" in AuditPipeline._collect_title(
        "ahmad_2022", [abs_item], abs_doi_source)

    failed = SourceEvidenceItem(
        study_id="elbaky_2023", group="t1dm", field_type="bmi",
        collection_outcome=CollectionOutcome.SOURCE_ACCESS_FAILED)
    failed_source = SimpleNamespace(
        retrieved=False, document=None, provenance={})
    assert AuditPipeline._collect_title(
        "elbaky_2023", [failed], failed_source
    ) == "elbaky_2023 · NOT RETRIEVED     (all four retrieval tiers failed)"

    unresolved = SourceEvidenceItem(
        study_id="capovilla_2023", group="mie", field_type="event_count",
        collection_outcome=CollectionOutcome.UNRESOLVED_SOURCE)
    unresolved_source = SimpleNamespace(
        retrieved=False, document=None, provenance={},
        outcome=CollectionOutcome.UNRESOLVED_SOURCE,
        reference=SimpleNamespace(
            title="Capovilla G. Front Oncol. 2023;13:1104109.",
            authors=[], year=2023, journal="Front Oncol"),
    )
    from react_review.tools.search.reconciler import _UNRESOLVED_NOTES, _query_key
    from react_review.tools.search.models import ReferenceQuery
    key = _query_key(ReferenceQuery(
        citation="Capovilla G. Front Oncol. 2023;13:1104109.",
        title="Capovilla G. Front Oncol. 2023;13:1104109.",
        year=2023, journal="Front Oncol"))
    _UNRESOLVED_NOTES[key] = "retrieved 3 candidates, none matched the citation"
    title = AuditPipeline._collect_title(
        "capovilla_2023", [unresolved], unresolved_source)
    assert "四层全失败" not in title
    assert "all four retrieval tiers failed" not in title
    assert "all tiers failed" not in title.lower()
    assert "retrieved 3 candidates, none matched the citation" in title


def test_collect_title_failure_paths_do_not_blank_or_crash():
    from types import SimpleNamespace

    from react_review.core.enums import CollectionOutcome
    from react_review.orchestrator.audit_pipeline import AuditPipeline

    empty = SourceEvidenceItem(
        study_id="x", group="t1dm", field_type="bmi",
        collection_outcome=CollectionOutcome.FOUND)
    title = AuditPipeline._collect_title("x", [empty], None)
    assert "unknown source" in title
    assert title.strip()

    local = SourceEvidenceItem(
        study_id="ahmad_2022", group="t1dm", field_type="bmi",
        source_file=r"C:\papers\Ahmad 2022.pdf",
        retriever_kind="local_pdf",
        collection_outcome=CollectionOutcome.FOUND)
    local_source = SimpleNamespace(
        retrieved=True,
        document=SimpleNamespace(full_text="z" * 100, document_scope=None),
        provenance={"source_file": r"C:\papers\Ahmad 2022.pdf",
                    "retriever_kind": "local_pdf"},
    )
    titled = AuditPipeline._collect_title("ahmad_2022", [local], local_source)
    assert "Ahmad 2022.pdf" in titled
    assert "(hit:local PDF)" in titled
    assert "10.xxxx" not in titled
    assert "doi:" not in titled

    no_scope = AuditPipeline._collect_title(
        "ahmad_2022", [empty],
        SimpleNamespace(
            retrieved=True,
            document=SimpleNamespace(full_text="hi", document_scope=None),
            provenance={"retriever_kind": "pmc"},
        ),
    )
    assert no_scope == "ahmad_2022 · pmc · 2 chars     (hit:PMC esearch by DOI)"
    assert "unknown" not in no_scope

