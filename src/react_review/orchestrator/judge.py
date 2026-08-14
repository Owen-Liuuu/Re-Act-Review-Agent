"""Agent 3 — Judge / Arbiter (deterministic MVP).

Adjudicates an audit report into a FinalVerification: keeps the verdict and
routes every item that is not a clean MATCH — plus review claims with no source
evidence — to a Human Review Flag (architecture: Judge → Human Review Flag).

Kept deterministic for MVP: the discrepancy signals already come from the
tolerance compare and the Collector's reflection. An LLM arbitration step (for
genuinely ambiguous cases) can slot in later.
"""
from __future__ import annotations

from react_review.claim_ids import declared_claim_id
from react_review.checklist.schema import ChecklistApplication
from react_review.core.enums import AuditLabel, CollectionOutcome, ReportVerdict
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
    # A broken reading is not a silent paper. Without these two the batch path's
    # failures would arrive as `missing_source`, which reads as "the review may
    # have made this up" — an accusation for a fault in the extractor.
    CollectionOutcome.EXTRACTION_FAILED: (
        "extraction_failed",
        "the paper was retrieved but the reading of it failed: no response, a "
        "malformed one, or one whose every reading failed a deterministic "
        "check. This says nothing about whether the paper states the value"),
    CollectionOutcome.EXTRACTION_UNRESOLVED: (
        "extraction_unresolved",
        "the paper was read and the answer is not unique: the arm, population "
        "or timepoint could not be pinned to one candidate, or the paper "
        "disagrees with itself. A human must choose, and nothing was released"),
}


def _locator(item) -> tuple:
    """The full identity of one audited cell — the key plus where it came from."""
    return (
        item.study_id, item.group, getattr(item, "timepoint", "single"),
        item.field_type, getattr(item, "table_id", "") or "",
        getattr(item, "cell_ref", None),
        getattr(item, "checklist_id", "") or "",
    )


def _identity(item) -> tuple:
    """Prefer the carried claim identity; locator is legacy fallback only."""

    claim_id = declared_claim_id(item)
    return ("claim", claim_id) if claim_id else ("legacy", *_locator(item))


def _identity_lookup(mapping: dict[tuple, object], item):
    """Use explicit identity first; locator fallback exists for legacy runs."""

    direct = mapping.get(_identity(item))
    if direct is not None:
        return direct
    return mapping.get(("legacy", *_locator(item)))


def _flag(
    item, label: str, reason: str, *, field_type: str | None = None,
    resolution_key: str = "", affected_cells: int = 1,
) -> HumanReviewFlag:
    """A flag that points at ONE cell, so a human can go and check it."""
    return HumanReviewFlag(
        audit_id=declared_claim_id(item),
        study_id=item.study_id, group=item.group,
        timepoint=getattr(item, "timepoint", "single"),
        field_type=item.field_type if field_type is None else field_type,
        table_id=getattr(item, "table_id", "") or "",
        cell_ref=getattr(item, "cell_ref", None),
        checklist_id=getattr(item, "checklist_id", "") or "",
        resolution_key=resolution_key or getattr(item, "resolution_key", ""),
        affected_cells=affected_cells, label=label, reason=reason,
    )


class Judge:
    """Turn an AuditReport into a FinalVerification with human-review flags."""

    def adjudicate(
        self,
        report: AuditReport,
        source_items: list[SourceEvidenceItem] | None = None,
        review_items: list["ReviewDataItem"] | None = None,
        checklist: ChecklistApplication | None = None,
    ) -> FinalVerification:
        # Keyed on the FULL locator, not (study, group, field): two rows of the
        # same study/cohort/field would otherwise overwrite each other here, and
        # the report would show one row's evidence beside the other's number.
        outcome_by_key = {
            _identity(s): s.collection_outcome for s in (source_items or [])
        }
        # Source evidence that refutes a candidate translation (unit/range).
        mismatch_by_key = {
            _identity(s): s.concept_mismatch_reason
            for s in (source_items or []) if s.concept_mismatch
        }
        flags: list[HumanReviewFlag] = []

        for r in report.results:
            if r.label == AuditLabel.MATCH:
                # A match that only checked PART of what the values reported is
                # not a finished check — an interval or a count the other side
                # never stated stays unverified, and must be seen.
                if r.review_required:
                    flags.append(_flag(r, "partially_verified", r.reason))
                continue
            label, reason = r.label.value, r.reason
            if r.label == AuditLabel.NOT_COMPARABLE:
                refined = _OUTCOME_FLAG.get(_identity_lookup(outcome_by_key, r))
                if refined:
                    label, reason = refined
            flags.append(_flag(r, label, reason))

        for claim in report.unmatched_review:
            refined = _OUTCOME_FLAG.get(_identity_lookup(outcome_by_key, claim))
            if claim.reason_code != "no_source_evidence":
                # Refusing an ambiguous or conflicting identity is its own
                # finding — never dress it up as a missing source.
                label, reason = claim.reason_code, claim.message
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

        # DKB guardrail is CONCEPT-level, not cell-level. One tentative mapping
        # can affect dozens of rows; emitting the same flag for every row creates
        # alert fatigue without creating any new decision for the reviewer.
        concept_groups: dict[str, list[ReviewDataItem]] = {}
        for index, item in enumerate(review_items or []):
            if getattr(item, "resolution_status", "resolved") not in {
                    "candidate", "unresolved"}:
                continue
            key = getattr(item, "resolution_key", "") or f"cell:{index}:{_locator(item)!r}"
            concept_groups.setdefault(key, []).append(item)

        for key, items in concept_groups.items():
            statuses = {getattr(item, "resolution_status", "resolved") for item in items}
            contradictions = []
            for item in items:
                mismatch = _identity_lookup(mismatch_by_key, item)
                if mismatch is not None:
                    contradictions.append((item, mismatch))
            exemplar = contradictions[0][0] if contradictions else items[0]
            raw_name = exemplar.raw_field_name or exemplar.field_type
            scope = (f"affects {len(items)} review cell(s) across "
                     f"{len({item.study_id for item in items})} study/studies")

            if contradictions:
                details = list(dict.fromkeys(reason for _, reason in contradictions))
                label = "concept_contradicted"
                reason = (
                    f"field '{raw_name}' was auto-classified as {exemplar.field_type}, "
                    f"but source evidence contradicts the mapping: {'; '.join(details)}; "
                    f"{scope}")
                field_type = exemplar.field_type
            elif "unresolved" in statuses:
                label = "needs_review"
                reason = (
                    f"field '{raw_name}' could not be mapped to a stable, "
                    f"self-consistent concept — not comparable; {scope}")
                field_type = exemplar.field_type or raw_name
            else:
                label = "provisional_concept"
                reason = (
                    f"field '{raw_name}' was auto-classified as "
                    f"{exemplar.field_type} (stable but still provisional); {scope}; "
                    "confirm the concept mapping")
                field_type = exemplar.field_type
            flags.append(_flag(
                exemplar, label, reason, field_type=field_type,
                resolution_key=(key if not key.startswith("cell:") else ""),
                affected_cells=len(items)))

        # A missing checklist requirement is a review-level finding, not a
        # fabricated claim with an empty value. Route it directly to the human.
        for gap in (checklist.gaps if checklist is not None else []):
            flags.append(HumanReviewFlag(
                study_id=gap.study_id, group=gap.group,
                field_type=gap.checklist_id, checklist_id=gap.checklist_id,
                label="checklist_gap", reason=gap.reason))

        verdict = report.verdict
        if checklist is not None and checklist.gaps and verdict == ReportVerdict.PASS:
            verdict = ReportVerdict.PARTIAL

        summary = (
            f"[{verdict.value}] {report.n_match} match, {report.n_mismatch} "
            f"mismatch, {report.n_unit_mismatch} unit_mismatch; "
            f"{len(flags)} review flag(s)."
        )
        return FinalVerification(
            run_id=report.run_id, verdict=verdict,
            human_review_flags=flags, summary=summary,
        )
