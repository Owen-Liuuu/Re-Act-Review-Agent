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

from react_review.core.enums import AuditLabel, ReportVerdict
from react_review.orchestrator.matcher import build_pairs
from react_review.schemas.audit import MatchResult
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.report import AuditReport, UnmatchedClaim
from react_review.tools.models import CompareInput
from react_review.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class AuditOrchestrator:
    """Run the deterministic audit over a review table and a source table."""

    def __init__(self, catalogue: ToolRegistry) -> None:
        # The orchestrator only needs the compare tool in P1; search/verify/
        # extract are exercised by the Collector (P2) that fills the tables.
        self._compare = catalogue.get("compare_values")

    async def run(
        self,
        review_items: list[ReviewDataItem],
        source_items: list[SourceEvidenceItem],
        *,
        run_id: str = "",
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
            try:
                res: MatchResult = await self._compare.run(
                    CompareInput(
                        field_type=review.field_type,
                        review_value=review.value,
                        source_value=source.source_value,
                        review_unit=review.unit,
                        source_unit=source.source_unit,
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

        if not results:
            verdict = ReportVerdict.INCOMPLETE
        elif n_mismatch > 0:
            verdict = ReportVerdict.FAIL
        elif n_unit > 0:
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
