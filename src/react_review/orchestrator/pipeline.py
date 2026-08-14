"""The thin deterministic audit orchestrator.

Sequences: match review↔source (build_pairs) → compare each pair via the
``compare_values`` tool from the catalogue → aggregate into an AuditReport.
Control flow is deterministic; the comparison is delegated to the typed tool.
Per-pair failures are caught and surfaced as flags rather than aborting the run.
"""
from __future__ import annotations

import uuid
from collections import Counter

import structlog

from react_review.claim_ids import claim_id_of, declared_claim_id
from react_review.contracts import ContractError
from react_review.core.enums import AuditLabel, ReportVerdict
from react_review.orchestrator.matcher import build_pairs
from react_review.schemas.adequacy import (
    AdequacyEvaluatorIdentity,
    AdequacyStatus,
    AxisStatus,
    EvidenceAdequacy,
)
from react_review.schemas.audit import MatchResult
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.report import AuditReport, UnmatchedClaim
from react_review.tools.models import CompareInput
from react_review.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


def resolved_evidence_adequacy(
    source: SourceEvidenceItem,
    *,
    required: bool,
    evaluator_identity: AdequacyEvaluatorIdentity | None = None,
) -> EvidenceAdequacy | None:
    """Return the assessment in force, failing closed when one is required."""
    if source.evidence_adequacy is not None:
        return source.evidence_adequacy
    if not required:
        return None
    return EvidenceAdequacy(
        status=AdequacyStatus.UNKNOWN,
        document_scope=source.document_scope,
        reason_codes=["adequacy_not_assessed"],
        evaluator=(evaluator_identity or AdequacyEvaluatorIdentity()),
    )


def evidence_adequacy_reason(adequacy: EvidenceAdequacy) -> str:
    """A stable, human-readable explanation of a gate refusal."""
    details = []
    for axis in adequacy.required_axes:
        result = adequacy.axis_results.get(axis)
        if result is None or result.status in {AxisStatus.PASS, AxisStatus.NOT_REQUIRED}:
            continue
        details.append(f"{axis}: {result.reason or result.status.value}")
    if not details:
        details.extend(adequacy.reason_codes)
    if not details:
        details.append("the required claim-level evidence binding was not established")
    return f"evidence {adequacy.status.value}: " + "; ".join(details)


def adequacy_not_comparable(
    review: ReviewDataItem,
    source: SourceEvidenceItem,
    adequacy: EvidenceAdequacy,
    *,
    audit_id: str,
) -> MatchResult:
    """Create the refusal result directly; no comparison has happened."""
    return MatchResult(
        audit_id=audit_id,
        field_type=review.field_type,
        review_value=review.value,
        source_value=source.source_value,
        review_unit=review.unit,
        source_unit=source.source_unit,
        label=AuditLabel.NOT_COMPARABLE,
        reason=evidence_adequacy_reason(adequacy),
        match_mode="evidence_adequacy",
        review_required=True,
        evidence_adequacy=adequacy,
    )


class AuditOrchestrator:
    """Run the deterministic audit over a review table and a source table."""

    def __init__(
        self,
        catalogue: ToolRegistry,
        *,
        require_evidence_adequacy: bool = False,
        adequacy_identity: AdequacyEvaluatorIdentity | None = None,
    ) -> None:
        # The orchestrator only needs the compare tool in P1; search/verify/
        # extract are exercised by the Collector (P2) that fills the tables.
        self._compare = catalogue.get("compare_values")
        self._require_evidence_adequacy = require_evidence_adequacy
        self._adequacy_identity = adequacy_identity

    async def run(
        self,
        review_items: list[ReviewDataItem],
        source_items: list[SourceEvidenceItem],
        *,
        run_id: str = "",
        research_context: str = "",
    ) -> AuditReport:
        run_id = run_id or uuid.uuid4().hex[:12]
        pairs, unmatched_review, unmatched_source = build_pairs(review_items, source_items)
        logger.info(
            "audit_start",
            run_id=run_id,
            pairs=len(pairs),
            unmatched_review=len(unmatched_review),
            unmatched_source=len(unmatched_source),
        )

        results: list[MatchResult] = []
        flags: list[str] = []
        for review, source in pairs:
            review_id = declared_claim_id(review)
            source_id = declared_claim_id(source)
            if bool(review_id) != bool(source_id) or (
                    review_id and review_id != source_id):
                raise ContractError(
                    "matched review/source rows disagree on claim identity: "
                    f"review={review_id!r}, source={source_id!r}")
            verified_id = review_id or claim_id_of(review)
            adequacy = resolved_evidence_adequacy(
                source, required=self._require_evidence_adequacy,
                evaluator_identity=self._adequacy_identity)
            if (adequacy is not None
                    and adequacy.status is not AdequacyStatus.SUFFICIENT):
                res = adequacy_not_comparable(
                    review, source, adequacy, audit_id=verified_id)
                logger.info(
                    "compare_skipped_evidence_adequacy",
                    audit_id=verified_id,
                    status=adequacy.status.value,
                    document_scope=adequacy.document_scope.value,
                    reason_codes=adequacy.reason_codes,
                )
                res.study_id = review.study_id
                res.group = review.group
                res.timepoint = review.timepoint
                res.table_id = review.table_id
                res.cell_ref = review.cell_ref
                res.checklist_id = review.checklist_id
                results.append(res)
                continue
            try:
                res: MatchResult = await self._compare.run(
                    CompareInput(
                        audit_id=verified_id,
                        field_type=review.field_type,
                        review_value=review.value,
                        source_value=source.source_value,
                        review_unit=review.unit,
                        source_unit=source.source_unit,
                        column_header=review.column_header or review.raw_field_name,
                        source_quote=source.source_quote,
                        research_context=research_context,
                        source_components=(
                            source.source_components.model_dump()
                            if source.source_components else None),
                        review_scope=(review.population_scope.model_dump()
                                      if review.population_scope else None),
                        source_scope=(source.population_scope.model_dump()
                                      if source.population_scope else None),
                        evidence_adequacy=adequacy,
                    )
                )
            except Exception as exc:  # a bad pair must not abort the whole run
                logger.error("compare_failed", study=review.study_id, error=str(exc))
                flags.append(
                    f"compare failed for {review.study_id}/{review.group}/"
                    f"{review.field_type}: {exc}"
                )
                continue
            res.study_id = review.study_id
            res.group = review.group
            res.timepoint = review.timepoint
            res.table_id = review.table_id
            res.cell_ref = review.cell_ref
            res.checklist_id = review.checklist_id
            res.evidence_adequacy = adequacy
            results.append(res)

        for u in (*unmatched_review, *unmatched_source):
            flags.append(f"{u.reason_code}: {u.key_text} — {u.message}")

        report = self._aggregate(results, unmatched_review, unmatched_source, flags, run_id)
        logger.info(
            "audit_complete",
            run_id=run_id,
            verdict=report.verdict.value,
            match=report.n_match,
            mismatch=report.n_mismatch,
            unit_mismatch=report.n_unit_mismatch,
        )
        return report

    @staticmethod
    def _aggregate(
        results: list[MatchResult],
        unmatched_review: list[UnmatchedClaim],
        unmatched_source: list[UnmatchedClaim],
        flags: list[str],
        run_id: str,
    ) -> AuditReport:
        counts = Counter(r.label for r in results)
        n_match = counts[AuditLabel.MATCH]
        n_mismatch = counts[AuditLabel.MISMATCH]
        n_unit = counts[AuditLabel.UNIT_MISMATCH]
        n_nc = counts[AuditLabel.NOT_COMPARABLE]

        # A run holding a match that was only partially verified has not fully
        # checked its claims, and PASS would say otherwise.
        n_partial = sum(1 for r in results if r.review_required)

        if not results:
            verdict = ReportVerdict.INCOMPLETE
        elif n_mismatch > 0:
            verdict = ReportVerdict.FAIL
        elif n_unit > 0 or n_partial > 0:
            verdict = ReportVerdict.PARTIAL
        elif n_match == 0:
            # Every pair was not_comparable — nothing was actually verified.
            verdict = ReportVerdict.INCOMPLETE
        elif n_nc > 0:
            # Some pairs verified, others could not be compared → partial coverage.
            verdict = ReportVerdict.PARTIAL
        else:
            verdict = ReportVerdict.PASS

        summary = (
            f"[{verdict.value}] {len(results)} compared: {n_match} match, "
            f"{n_mismatch} mismatch, {n_unit} unit_mismatch, {n_nc} not_comparable. "
            f"Unmatched: {len(unmatched_review)} review / {len(unmatched_source)} source."
        )

        return AuditReport(
            run_id=run_id,
            results=results,
            n_match=n_match,
            n_mismatch=n_mismatch,
            n_unit_mismatch=n_unit,
            n_not_comparable=n_nc,
            unmatched_review=unmatched_review,
            unmatched_source=unmatched_source,
            verdict=verdict,
            flags=flags,
            summary=summary,
        )
