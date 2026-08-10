"""Full audit pipeline: Collector → Auditor → Judge → EvidencePackage.

Sequences the P2 agents deterministically (the orchestrator owns control flow):
for each review claim the Collector finds the source value; the Auditor
(AuditOrchestrator) matches + compares review vs source; the Judge adjudicates
and flags items for human review. The whole run is persisted as an
EvidencePackage. The review items themselves come from the ReviewParser (run by
the caller / CLI) or from a CSV.

Collection is grouped BY SOURCE PAPER rather than by claim: each group reports
every value it read from that paper and then offers a checkpoint, so a reviewer
sees a paper's full evidence at once instead of being asked 60 times. Progress
is written to ``package.partial.json`` after every group, so a run that is
stopped — or interrupted — keeps the evidence it had already collected.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Callable

import structlog

from react_review.checklist.schema import ChecklistApplication
from react_review.hitl.events import StepStage, SubjectKind
from react_review.hitl.reporter import StepReporter
from react_review.normalize.cohorts import CohortRegistry
from react_review.orchestrator.judge import Judge
from react_review.orchestrator.pipeline import AuditOrchestrator
from react_review.schemas.agent import AgentRun
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.knowledge import KnowledgeImportRecord
from react_review.schemas.package import EvidencePackage
from react_review.schemas.resolution import FieldResolutionRecord
from react_review.schemas.table import CapturedTableSet
from react_review.store.evidence_package import EvidencePackageStore

if TYPE_CHECKING:  # avoid an agents <-> orchestrator import cycle at runtime
    from react_review.agents.collector import Collector
    from react_review.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)

ReferenceResolver = Callable[[str], "ReferenceEntry"]


def _group_by_study(items: list[ReviewDataItem]) -> list[tuple[str, list[ReviewDataItem]]]:
    """Group claims by study_id, preserving first-appearance order."""
    order: list[str] = []
    groups: dict[str, list[ReviewDataItem]] = {}
    for item in items:
        if item.study_id not in groups:
            groups[item.study_id] = []
            order.append(item.study_id)
        groups[item.study_id].append(item)
    return [(sid, groups[sid]) for sid in order]


class AuditPipeline:
    """Collect source evidence, audit it, adjudicate, and persist."""

    def __init__(
        self,
        collector: Collector,
        auditor: AuditOrchestrator,
        judge: Judge,
        *,
        store: EvidencePackageStore | None = None,
        reporter: StepReporter | None = None,
        run_manifest=None,
    ) -> None:
        self._collector = collector
        self._auditor = auditor
        self._judge = judge
        self._store = store
        # Recorded on every package this run writes, partial ones included: a
        # run that stopped halfway still has to say what rules it was applying.
        self._run_manifest = run_manifest
        # Default reporter never blocks and never writes — library/CI behaviour.
        self._reporter = reporter or StepReporter()

    async def run(
        self,
        review_items: list[ReviewDataItem],
        reference_for: ReferenceResolver,
        *,
        research_context: str = "",
        run_id: str | None = None,
        parser_record: AgentRun | None = None,
        captured_tables: CapturedTableSet | None = None,
        cohorts: CohortRegistry | None = None,
        field_resolutions: list[FieldResolutionRecord] | None = None,
        knowledge_imports: list[KnowledgeImportRecord] | None = None,
        knowledge_fingerprint: str = "",
        knowledge_concept_count: int = 0,
        checklist: ChecklistApplication | None = None,
    ) -> EvidencePackage:
        run_id = run_id or uuid.uuid4().hex[:12]
        self._reporter.run_id = self._reporter.run_id or run_id
        logger.info("audit_pipeline_start", run_id=run_id, n_review=len(review_items))

        source_items = []
        records: list[AgentRun] = [parser_record] if parser_record else []
        groups = _group_by_study(review_items)
        per_study: list[dict] = []

        for study_id, claims in groups:
            reference = reference_for(study_id)
            for item in claims:
                result = await self._collector.collect(
                    item, reference, research_context=research_context
                )
                source_items.append(result.source_item)
                records.append(result.record)

            # Progress survives a crash or a Ctrl-C: written after every paper.
            if self._store is not None:
                self._store.save_partial(EvidencePackage(
                    run_id=run_id, run_manifest=self._run_manifest,
                    review_items=review_items, source_items=source_items,
                    processing_records=records,
                    captured_tables=captured_tables or CapturedTableSet(),
                    cohorts=cohorts or CohortRegistry(),
                    field_resolutions=field_resolutions or [],
                    knowledge_imports=knowledge_imports or [],
                    knowledge_fingerprint=knowledge_fingerprint,
                    knowledge_concept_count=knowledge_concept_count,
                    checklist=checklist,
                    status="in_progress",
                ))

            collected = source_items[-len(claims):]
            subject = self._subject_for(collected, reference)
            warnings = self._warn_for(collected)
            per_study.append({
                "study_id": study_id, "subject": subject,
                "n_claims": len(claims), "n_found": len(claims) - len(warnings),
                "warnings": warnings,
            })
            # Shown in full, not gated — see StepStage.COLLECT_STUDY.
            await self._reporter.step_or_stop(
                StepStage.COLLECT_STUDY,
                title=f"Source evidence · {study_id}",
                subject=subject,
                subject_kind=SubjectKind.SOURCE_PDF,
                payload={"study_id": study_id,
                         "claims": [i.model_dump(mode="json") for i in claims],
                         "evidence": [s.model_dump(mode="json") for s in collected]},
                render_blocks=[self._render_study(study_id, claims, collected)],
                warnings=warnings,
            )

        # ONE checkpoint for the whole collection, however many papers there were.
        await self._reporter.step_or_stop(
            StepStage.COLLECTION_REVIEW,
            title="Source evidence collected",
            payload={"papers": per_study,
                     "evidence": [s.model_dump(mode="json") for s in source_items]},
            render_blocks=[self._render_collection(per_study, len(source_items))],
            warnings=[w for p in per_study for w in p["warnings"]],
        )

        report = await self._auditor.run(review_items, source_items, run_id=run_id,
                                         research_context=research_context)
        await self._reporter.step_or_stop(
            StepStage.AUDIT_SUMMARY, title="Audit result",
            payload=report.model_dump(mode="json"),
            render_blocks=[report.summary],
        )

        final = self._judge.adjudicate(
            report, source_items, review_items, checklist=checklist)
        await self._reporter.step_or_stop(
            StepStage.JUDGE_FLAGS, title="Flagged for human review",
            payload=final.model_dump(mode="json"),
            render_blocks=[self._render_flags(final)],
        )

        package = EvidencePackage(
            run_id=run_id,
            run_manifest=self._run_manifest,
            review_items=review_items,
            source_items=source_items,
            report=report,
            final_verification=final,
            processing_records=records,
            captured_tables=captured_tables or CapturedTableSet(),
            cohorts=cohorts or CohortRegistry(),
            field_resolutions=field_resolutions or [],
            knowledge_imports=knowledge_imports or [],
            knowledge_fingerprint=knowledge_fingerprint,
            knowledge_concept_count=knowledge_concept_count,
            checklist=checklist,
        )
        if self._store is not None:
            self._store.save(package)

        logger.info(
            "audit_pipeline_done", run_id=run_id, verdict=final.verdict.value,
            flags=len(final.human_review_flags),
        )
        return package

    # --- rendering helpers (strings only; the gate does the printing) ---

    @staticmethod
    def _subject_for(collected: list, reference) -> str:
        """Which file this study's evidence came from (advisor requirement #1)."""
        for s in collected:
            path = getattr(s, "source_file", "")
            if path:
                return path
        doi = getattr(reference, "doi", "") or ""
        return f"doi:{doi}" if doi else getattr(reference, "title", "") or ""

    @staticmethod
    def _warn_for(collected: list) -> list[str]:
        out = []
        for s in collected:
            outcome = getattr(s.collection_outcome, "value", str(s.collection_outcome))
            if outcome != "found":
                out.append(f"{s.group}/{s.field_type}: {outcome}")
        return out

    @staticmethod
    def _render_study(study_id: str, claims: list, collected: list) -> str:
        lines = [f"  {study_id}: {len(claims)} claim(s)"]
        for claim, src in zip(claims, collected):
            outcome = getattr(src.collection_outcome, "value", str(src.collection_outcome))
            got = src.source_value if src.source_value is not None else f"— ({outcome})"
            lines.append(
                f"    {claim.group}/{claim.field_type or claim.raw_field_name}: "
                f"review {claim.value!r} vs source {got!r}")
            if src.source_quote:
                lines.append(f"        “{src.source_quote[:110]}”")
        return "\n".join(lines)

    @staticmethod
    def _render_collection(per_study: list[dict], n_evidence: int) -> str:
        """The one screen a reviewer reads before letting the audit proceed."""
        found = sum(p["n_found"] for p in per_study)
        lines = [f"  {len(per_study)} paper(s) · {n_evidence} claim(s) · "
                 f"{found} found · {n_evidence - found} missing", ""]
        width = max((len(p["study_id"]) for p in per_study), default=10)
        for p in per_study:
            missing = p["n_claims"] - p["n_found"]
            mark = "ok" if not missing else f"! {missing} missing"
            lines.append(f"    {p['study_id']:<{width}}  {p['n_found']}/{p['n_claims']}"
                         f"  {mark}")
            if p["subject"]:
                lines.append(f"      {p['subject']}")
        return "\n".join(lines)

    @staticmethod
    def _render_flags(final) -> str:
        if not final.human_review_flags:
            return "  (no items flagged)"
        return "\n".join(
            f"  [{f.label}] {f.study_id}/{f.group}/{f.field_type}: {f.reason}"
            for f in final.human_review_flags[:40])
