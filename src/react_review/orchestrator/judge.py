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
    CollectionOutcome.UNRESOLVED_SOURCE: (
        "unresolved_source",
        "could not identify the source paper from its citation "
        "(no DOI and no confident online match)"),
    CollectionOutcome.UNKNOWN_COHORT: (
        "unknown_cohort",
        "the claim could not be tied to a cohort this review reports, so there "
        "was nothing specific to look for in the paper"),
}


def _locator(item) -> tuple:
    """The full identity of one audited cell — the key plus where it came from."""
    return (
        item.study_id, item.group, getattr(item, "timepoint", "single"),
        item.field_type, getattr(item, "table_id", "") or "",
        getattr(item, "cell_ref", None),
    )


def _flag(item, label: str, reason: str, *, field_type: str | None = None) -> HumanReviewFlag:
    """A flag that points at ONE cell, so a human can go and check it."""
    return HumanReviewFlag(
        study_id=item.study_id, group=item.group,
        timepoint=getattr(item, "timepoint", "single"),
        field_type=item.field_type if field_type is None else field_type,
        table_id=getattr(item, "table_id", "") or "",
        cell_ref=getattr(item, "cell_ref", None), label=label, reason=reason,
    )


class Judge:
    """Turn an AuditReport into a FinalVerification with human-review flags."""

    def adjudicate(
        self,
        report: AuditReport,
        source_items: list[SourceEvidenceItem] | None = None,
        review_items: list["ReviewDataItem"] | None = None,
    ) -> FinalVerification:
        # Keyed on the FULL locator, not (study, group, field): two rows of the
        # same study/cohort/field would otherwise overwrite each other here, and
        # the report would show one row's evidence beside the other's number.
        outcome_by_key = {
            _locator(s): s.collection_outcome for s in (source_items or [])
        }
        # Source evidence that refutes a candidate translation (unit/range).
        mismatch_by_key = {
            _locator(s): s.concept_mismatch_reason
            for s in (source_items or []) if s.concept_mismatch
        }
        flags: list[HumanReviewFlag] = []

        for r in report.results:
            if r.label == AuditLabel.MATCH:
                continue
            label, reason = r.label.value, r.reason
            if r.label == AuditLabel.NOT_COMPARABLE:
                refined = _OUTCOME_FLAG.get(outcome_by_key.get(_locator(r)))
                if refined:
                    label, reason = refined
            flags.append(_flag(r, label, reason))

        for claim in report.unmatched_review:
            refined = _OUTCOME_FLAG.get(outcome_by_key.get(_locator(claim)))
            if claim.reason_code == "ambiguous_match_key":
                # Refusing to pair is its own finding — never dress it up as a
                # missing source, which would read as a possible fabrication.
                label, reason = "ambiguous_match_key", claim.message
            else:
                label, reason = refined or ("unmatched", claim.message or
                                            "no source evidence for this review claim")
            flags.append(_flag(claim, label, reason))

        # Cohort guardrail: a claim whose arm could not be placed, or whose
        # evidence could not be confirmed to belong to that arm, is kept — and
        # said out loud. Neither may be presented as a checked result.
        cohort_flag = {
            "unknown": ("unknown_cohort",
                        "the review's cohort label could not be matched to any "
                        "cohort this review reports"),
            "ambiguous": ("cohort_ambiguous",
                          "which cohort this value belongs to could not be confirmed"),
        }
        for item in (review_items or []):
            entry = cohort_flag.get(getattr(item, "cohort_status", "resolved"))
            if entry is None:
                continue
            label, default = entry
            detail = next((str(r) for r in getattr(item, "reasons", [])
                           if r.code.startswith("cohort")), "")
            flags.append(_flag(
                item, label,
                f"cohort {item.cohort_label!r}: {detail or default}",
                field_type=item.field_type or item.raw_field_name))

        for s in (source_items or []):
            if s.cohort_check == "ambiguous":
                detail = next((str(r) for r in s.reasons
                               if r.code == "cohort_ambiguous"), "")
                flags.append(_flag(s, "cohort_ambiguous", detail or
                                   "the source cohort could not be confirmed"))

        # DKB guardrail: the concept mapping's own certainty, independent of the
        # value check. A CANDIDATE (LLM-proposed, not-yet-authoritative) must be
        # confirmed; an UNRESOLVED field (no concept found) is kept but flagged.
        for item in (review_items or []):
            status = getattr(item, "resolution_status", "resolved")
            if status == "candidate":
                contradiction = mismatch_by_key.get(_locator(item))
                if contradiction:
                    label, reason = "concept_contradicted", (
                        f"field '{item.raw_field_name or item.field_type}' was "
                        f"auto-classified as {item.field_type}, but the source "
                        f"evidence contradicts it: {contradiction}")
                else:
                    label, reason = "provisional_concept", (
                        f"field '{item.raw_field_name or item.field_type}' was "
                        f"auto-classified as {item.field_type} (provisional) — "
                        "confirm the concept mapping")
                flags.append(_flag(item, label, reason))
            elif status == "unresolved":
                # No concept to name it by — show the review's own column header
                # so the flag still says WHICH field a human should look at.
                flags.append(_flag(
                    item, "needs_review",
                    f"field '{item.raw_field_name}' could not be mapped to a known "
                    "concept — not comparable, needs human review",
                    field_type=item.field_type or item.raw_field_name))

        summary = (
            f"[{report.verdict.value}] {report.n_match} match, {report.n_mismatch} "
            f"mismatch, {report.n_unit_mismatch} unit_mismatch; "
            f"{len(flags)} item(s) flagged for human review."
        )
        return FinalVerification(
            run_id=report.run_id, verdict=report.verdict,
            human_review_flags=flags, summary=summary,
        )
