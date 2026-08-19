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

import time
import uuid
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import structlog

from react_review.checklist.schema import ChecklistApplication
from react_review.core.enums import CollectionOutcome
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
from react_review.steps.data_extraction.schemas import DocumentScope
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
        if not item.study_id:
            continue
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
        telemetry=None,
        owns_final_save: bool = True,
    ) -> None:
        self._collector = collector
        self._auditor = auditor
        self._judge = judge
        self._store = store
        # Recorded on every package this run writes, partial ones included: a
        # run that stopped halfway still has to say what rules it was applying.
        self._run_manifest = run_manifest
        # What the run cost, written into the package it produces — including
        # the partial one, where a reader most needs to know what had already
        # been spent when it stopped.
        self._telemetry = telemetry
        # Whether saving the FINISHED package is this object's job. It is for a
        # library caller, who has nothing else; it is not for a production run,
        # which has a session that closes the books first. Saving here in that
        # case wrote a package whose telemetry stopped before the last paper,
        # and left the caller to save a second time over the same filename.
        self._owns_final_save = owns_final_save
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
        batch_records = []
        seen_readings: set[str] = set()
        groups = _group_by_study(review_items)
        per_study: list[dict] = []
        n_papers = len(groups)
        batch_started = time.monotonic()

        for i, (study_id, claims) in enumerate(groups, start=1):
            reference = reference_for(study_id)
            # Opened once for the whole study: every claim about this paper is
            # then answered from the same retrieval, and the cost of an audit
            # scales with papers rather than with cells.
            opener = getattr(self._collector, "open_study", None)
            source = await opener(reference) if opener is not None else None
            collect_study = getattr(self._collector, "collect_study", None)
            if collect_study is not None:
                # One pass per paper. The claims come back in the order they
                # went in, so nothing here has to know they were grouped.
                produced = await collect_study(
                    claims, reference, research_context=research_context,
                    **({"source": source} if source is not None else {}))
                source_items.extend(produced.source_items)
                records.extend(produced.records)
                for record in produced.batch_records:
                    # Deduplicated by execution id: one reading is one record
                    # however many claims name it, and a run that resumed could
                    # otherwise write the same reading twice.
                    persistent = record.persistent()
                    if persistent.execution_id not in seen_readings:
                        seen_readings.add(persistent.execution_id)
                        batch_records.append(persistent)
            else:
                for item in claims:
                    result = await self._collector.collect(
                        item, reference, research_context=research_context,
                        **({"source": source} if source is not None else {})
                    )
                    source_items.append(result.source_item)
                    records.append(result.record)

            # Progress survives a crash or a Ctrl-C: written after every paper.
            if self._store is not None:
                self._store.save_partial(EvidencePackage(
                    run_id=run_id, run_manifest=self._run_manifest,
                    review_items=review_items, source_items=source_items,
                    processing_records=records, batch_records=batch_records,
                    telemetry=self._telemetry,
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
                "document_scope": self._scope_for(source),
                "adequacy": dict(Counter(
                    (s.evidence_adequacy.status.value
                     if s.evidence_adequacy is not None else "not_assessed")
                    for s in collected)),
                "warnings": warnings,
            })
            # Shown in full, not gated — see StepStage.COLLECT_STUDY.
            await self._reporter.step_or_stop(
                StepStage.COLLECT_STUDY,
                title=self._collect_title(study_id, collected, source),
                subject=subject,
                subject_kind=SubjectKind.SOURCE_PDF,
                payload={"study_id": study_id,
                         "claims": [i.model_dump(mode="json") for i in claims],
                         "evidence": [s.model_dump(mode="json") for s in collected]},
                render_blocks=[self._render_study(study_id, claims, collected)],
                warnings=warnings,
            )
            if i % 10 == 0 or i == n_papers:
                self._reporter.progress(
                    "paper", i, n_papers, caption=study_id, started=batch_started)
                batch_started = time.monotonic()

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
            batch_records=batch_records,
            telemetry=self._telemetry,
            captured_tables=captured_tables or CapturedTableSet(),
            cohorts=cohorts or CohortRegistry(),
            field_resolutions=field_resolutions or [],
            knowledge_imports=knowledge_imports or [],
            knowledge_fingerprint=knowledge_fingerprint,
            knowledge_concept_count=knowledge_concept_count,
            checklist=checklist,
        )
        if self._store is not None and self._owns_final_save:
            self._store.save(package)

        logger.info(
            "audit_pipeline_done", run_id=run_id, verdict=final.verdict.value,
            flags=len(final.human_review_flags),
        )
        return package

    # --- rendering helpers (strings only; the gate does the printing) ---

    _TITLE_MAX = 72
    _ACCESS_FAILED = frozenset({
        CollectionOutcome.SOURCE_ACCESS_FAILED.value,
        CollectionOutcome.UNRESOLVED_SOURCE.value,
    })

    @classmethod
    def _collect_title(cls, study_id: str, collected: list, source) -> str:
        """One glance: which paper, which tier, how much text, which scope."""
        if cls._retrieval_failed(collected, source):
            mismatch = cls._unresolved_note(source)
            suffix = mismatch or "all four retrieval tiers failed"
            return cls._clip_title(f"{study_id} · NOT RETRIEVED") + f"     ({suffix})"

        ident = cls._source_ident(collected, source)
        parts = [study_id, ident]
        n_chars = cls._source_chars(source)
        if n_chars:
            parts.append(f"{n_chars:,} chars")
        scope = cls._scope_for(source)
        if scope in {"", DocumentScope.UNKNOWN.value} and collected:
            item_scope = getattr(collected[0], "document_scope", None)
            scope = getattr(item_scope, "value", str(item_scope or ""))
        if scope and scope != DocumentScope.UNKNOWN.value:
            parts.append(scope)
        title = cls._clip_title(" · ".join(parts))
        note = cls._retrieval_note(collected, source)
        return f"{title}     ({note})" if note else title

    @staticmethod
    def _unresolved_note(source) -> str:
        """Reject-all-candidates text; empty when this was a fetch failure."""
        outcome = getattr(source, "outcome", None) if source is not None else None
        value = getattr(outcome, "value", outcome)
        if value != CollectionOutcome.UNRESOLVED_SOURCE.value:
            return ""
        from react_review.tools.search.reconciler import unresolved_note_for
        return unresolved_note_for(getattr(source, "reference", None)) or (
            "no matching record: candidates did not match the citation")

    @classmethod
    def _retrieval_note(cls, collected: list, source) -> str:
        """How this paper was actually retrieved — never a planned path."""
        doi = ""
        if source is not None:
            doi = str((getattr(source, "provenance", None) or {}).get("source_doi") or "")
        if not doi and collected:
            doi = str(getattr(collected[0], "source_doi", "") or "")
        retriever = next((str(s.retriever_kind) for s in collected
                          if getattr(s, "retriever_kind", "")), "")
        if not retriever and source is not None:
            retriever = str((getattr(source, "provenance", None) or {}).get(
                "retriever_kind") or "")
        if retriever == "pmc":
            return "hit:PMC esearch by DOI"
        if retriever == "unpaywall":
            return "hit:Unpaywall"
        if retriever.startswith("openalex"):
            return "hit:OpenAlex"
        if retriever == "pubmed_abstract":
            return ("fallback: DOI miss → title search" if not doi
                    else "fallback: DOI miss → PubMed abstract")
        if retriever == "local_pdf":
            return "hit:local PDF"
        if retriever:
            return f"hit:{retriever}"
        return ""

    @classmethod
    def _clip_title(cls, title: str) -> str:
        if len(title) <= cls._TITLE_MAX:
            return title
        return title[: cls._TITLE_MAX - 3] + "..."

    @classmethod
    def _retrieval_failed(cls, collected: list, source) -> bool:
        if source is not None and not getattr(source, "retrieved", False):
            return True
        outcomes = [
            getattr(s.collection_outcome, "value", str(s.collection_outcome))
            for s in collected
        ]
        return bool(outcomes) and all(o in cls._ACCESS_FAILED for o in outcomes)

    @staticmethod
    def _source_ident(collected: list, source) -> str:
        path = next((str(s.source_file) for s in collected
                     if getattr(s, "source_file", "")), "")
        if not path and source is not None:
            path = str((getattr(source, "provenance", None) or {}).get("source_file") or "")
        if path:
            return Path(path).name
        kind = next((str(s.retriever_kind) for s in collected
                     if getattr(s, "retriever_kind", "")), "")
        if not kind and source is not None:
            kind = str((getattr(source, "provenance", None) or {}).get("retriever_kind") or "")
        return kind or "unknown source"

    @staticmethod
    def _source_chars(source) -> int:
        document = getattr(source, "document", None) if source is not None else None
        text = getattr(document, "full_text", "") or ""
        return len(text)

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
                claim_id = getattr(s, "review_data_id", "") or ""
                prefix = f"[{claim_id}] " if claim_id else ""
                out.append(f"{prefix}{s.group}/{s.field_type}: {outcome}")
        return out

    @staticmethod
    def _scope_for(source) -> str:
        """Return the retriever-declared scope, never one inferred from text."""
        if source is None:
            return DocumentScope.UNKNOWN.value
        document = getattr(source, "document", None)
        scope = getattr(document, "document_scope", None)
        if scope is not None:
            return getattr(scope, "value", str(scope))
        provenance = getattr(source, "provenance", None) or {}
        return str(provenance.get("document_scope") or DocumentScope.UNKNOWN.value)

    @staticmethod
    def _render_study(study_id: str, claims: list, collected: list) -> str:
        lines = [f"  {study_id}: {len(claims)} claim(s)"]
        for claim, src in zip(claims, collected):
            outcome = getattr(src.collection_outcome, "value", str(src.collection_outcome))
            got = src.source_value if src.source_value is not None else f"— ({outcome})"
            claim_id = claim.review_data_id or getattr(src, "review_data_id", "") or "-"
            lines.append(
                f"    [{claim_id}] {claim.group}/{claim.field_type or claim.raw_field_name}: "
                f"review {claim.value!r} vs source {got!r}")
            if src.source_quote:
                lines.append(f"        “{src.source_quote[:110]}”")
        return "\n".join(lines)

    @staticmethod
    def _render_collection(per_study: list[dict], n_evidence: int) -> str:
        """The one screen a reviewer reads before letting the audit proceed."""
        found = sum(p["n_found"] for p in per_study)
        scopes = Counter(
            p.get("document_scope") or DocumentScope.UNKNOWN.value
            for p in per_study
        )
        adequacy = Counter()
        for paper in per_study:
            adequacy.update(paper.get("adequacy") or {})
        lines = [f"  {len(per_study)} paper(s) · {n_evidence} claim(s) · "
                 f"{found} found · {n_evidence - found} missing", ""]
        lines.extend([
            "  document scope:",
            f"    full_text       {scopes[DocumentScope.FULL_TEXT.value]}",
            f"    abstract_only   {scopes[DocumentScope.ABSTRACT_ONLY.value]}",
            f"    metadata_only   {scopes[DocumentScope.METADATA_ONLY.value]}",
            f"    unknown         {scopes[DocumentScope.UNKNOWN.value]}",
            "",
            "  evidence adequacy:",
            f"    sufficient      {adequacy['sufficient']}",
            f"    insufficient    {adequacy['insufficient']}",
            f"    unknown         {adequacy['unknown']}",
            f"    not_assessed    {adequacy['not_assessed']}",
            "",
        ])
        width = max((len(p["study_id"]) for p in per_study), default=10)
        for p in per_study:
            missing = p["n_claims"] - p["n_found"]
            mark = "ok" if not missing else f"! {missing} missing"
            lines.append(f"    {p['study_id']:<{width}}  {p['n_found']}/{p['n_claims']}"
                         f"  {mark}  [{p.get('document_scope', 'unknown')}]")
            if p["subject"]:
                lines.append(f"      {p['subject']}")
        return "\n".join(lines)

    @staticmethod
    def _render_flags(final) -> str:
        if not final.human_review_flags:
            return "  (no items flagged)"
        return "\n".join(
            f"  [{f.audit_id or '-'}] [{f.label}] "
            f"{f.study_id}/{f.group}/{f.field_type} "
            f"[scope={f.document_scope.value} "
            f"adequacy={f.evidence_adequacy_status or 'not_assessed'}]: {f.reason}"
            for f in final.human_review_flags[:40])
