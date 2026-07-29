"""Agent 3 — Judge / Arbiter (deterministic MVP).

Adjudicates an audit report into a FinalVerification: keeps the verdict and
routes every item that is not a clean MATCH — plus review claims with no source
evidence — to a Human Review Flag (architecture: Judge → Human Review Flag).

Kept deterministic for MVP: the discrepancy signals already come from the
tolerance compare and the Collector's reflection. An LLM arbitration step (for
genuinely ambiguous cases) can slot in later.
"""
from __future__ import annotations

from react_review.core.enums import AuditLabel, CollectionOutcome
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.report import AuditReport, FinalVerification, HumanReviewFlag

# A missing source value means two very different things; label them apart so a
# human sees "we couldn't get the paper" vs "the paper doesn't say this".
_OUTCOME_FLAG = {
    CollectionOutcome.SOURCE_ACCESS_FAILED: (
        "source_access_failed", "source full text could not be retrieved"),
    CollectionOutcome.MISSING_SOURCE: (
        "missing_source",
        "source paper retrieved but this value is not stated in it "
        "(possible fabrication)"),
}


class Judge:
    """Turn an AuditReport into a FinalVerification with human-review flags."""

    def adjudicate(
        self,
        report: AuditReport,
        source_items: list[SourceEvidenceItem] | None = None,
        review_items: list["ReviewDataItem"] | None = None,
    ) -> FinalVerification:
        outcome_by_key = {
            (s.study_id, s.group, s.field_type): s.collection_outcome
            for s in (source_items or [])
        }
        flags: list[HumanReviewFlag] = []

        for r in report.results:
            if r.label == AuditLabel.MATCH:
                continue
            label, reason = r.label.value, r.reason
            if r.label == AuditLabel.NOT_COMPARABLE:
                refined = _OUTCOME_FLAG.get(
                    outcome_by_key.get((r.study_id, r.group, r.field_type)))
                if refined:
                    label, reason = refined
            flags.append(HumanReviewFlag(
                study_id=r.study_id, group=r.group, field_type=r.field_type,
                label=label, reason=reason,
            ))

        for key in report.unmatched_review:
            parts = key.split("/")
            study = parts[0] if parts else ""
            group = parts[1] if len(parts) > 1 else "-"
            field_type = parts[3] if len(parts) > 3 else ""
            refined = _OUTCOME_FLAG.get(outcome_by_key.get((study, group, field_type)))
            label, reason = refined or (
                "unmatched", "no source evidence for this review claim")
            flags.append(HumanReviewFlag(
                study_id=study, group=group, field_type=field_type,
                label=label, reason=reason,
            ))

        # DKB guardrail: the concept mapping's own certainty, independent of the
        # value check. A CANDIDATE (LLM-proposed, not-yet-authoritative) must be
        # confirmed; an UNRESOLVED field (no concept found) is kept but flagged.
        for item in (review_items or []):
            status = getattr(item, "resolution_status", "resolved")
            if status == "candidate":
                flags.append(HumanReviewFlag(
                    study_id=item.study_id, group=item.group, field_type=item.field_type,
                    label="provisional_concept",
                    reason=f"field '{item.raw_field_name or item.field_type}' was "
                           f"auto-classified as {item.field_type} (provisional) — "
                           "confirm the concept mapping",
                ))
            elif status == "unresolved":
                flags.append(HumanReviewFlag(
                    study_id=item.study_id, group=item.group,
                    field_type=item.field_type or item.raw_field_name,
                    label="needs_review",
                    reason=f"field '{item.raw_field_name}' could not be mapped to a known "
                           "concept — not comparable, needs human review",
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
