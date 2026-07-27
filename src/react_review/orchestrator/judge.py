"""Agent 3 — Judge / Arbiter (deterministic MVP).

Adjudicates an audit report into a FinalVerification: keeps the verdict and
routes every item that is not a clean MATCH — plus review claims with no source
evidence — to a Human Review Flag (architecture: Judge → Human Review Flag).

Kept deterministic for MVP: the discrepancy signals already come from the
tolerance compare and the Collector's reflection. An LLM arbitration step (for
genuinely ambiguous cases) can slot in later.
"""
from __future__ import annotations

from react_review.core.enums import AuditLabel
from react_review.schemas.report import AuditReport, FinalVerification, HumanReviewFlag


class Judge:
    """Turn an AuditReport into a FinalVerification with human-review flags."""

    def adjudicate(self, report: AuditReport) -> FinalVerification:
        flags: list[HumanReviewFlag] = []

        for r in report.results:
            if r.label != AuditLabel.MATCH:
                flags.append(HumanReviewFlag(
                    study_id=r.study_id, group=r.group, field_type=r.field_type,
                    label=r.label.value, reason=r.reason,
                ))

        for key in report.unmatched_review:
            parts = key.split("/")
            flags.append(HumanReviewFlag(
                study_id=parts[0] if parts else "",
                group=parts[1] if len(parts) > 1 else "-",
                field_type=parts[3] if len(parts) > 3 else "",
                label="unmatched",
                reason="no source evidence for this review claim",
            ))

        summary = (
            f"[{report.verdict.value}] {report.n_match} match, {report.n_mismatch} "
            f"mismatch, {report.n_unit_mismatch} unit_mismatch; "
            f"{len(flags)} item(s) flagged for human review."
        )
        return FinalVerification(
            run_id=report.run_id, verdict=report.verdict,
            human_review_flags=flags, summary=summary,
        )
