"""Turning a projection into the evidence object the rest of the audit reads.

This is the seam. Everything downstream — the comparison, the report, the
benchmark — consumes :class:`SourceValueResult`, so a batched reading has to
arrive in exactly that shape or nothing after it can be trusted to mean what it
used to mean.

Two rules govern what may be written here.

*A field the batch does not assess is left empty, not filled with "ok".* The
cohort guard belongs to the single-target path and has no counterpart in a
batch, where the arm is decided by the global assignment instead. Writing its
default through would report a check as passed that never ran, which is how a
published result came to claim more than it had.

*A derived total does not get a quote that prints it.* No passage of the paper
states a number the paper never computed. The partition sentence is carried
instead, because that is what the arithmetic actually rests on, and the
components keep their own quotes so a reader can re-add them.
"""
from __future__ import annotations

from react_review.schemas.evidence import AggregationProvenance, CohortCount
from react_review.tools.batch_project import (
    CONTRADICTORY,
    DERIVED,
    OK,
    Projection,
)
from react_review.tools.extract_source import SourceValueResult

#: Projection statuses that mean "the batch answered a different question than
#: the one asked", which the target guard already has words for.
_TARGET_CHECK = {
    "not_reported": "not_reported",
    "ambiguous": "ambiguous",
    "unsupported": "unsupported",
    "contradictory": "inconsistent",
    "scope_unresolved": "unsupported",
    "timepoint_unresolved": "unsupported",
    "batch_failed": "protocol_error",
}


def _provenance(projection: Projection) -> AggregationProvenance | None:
    """The account of a sum — or nothing at all, where no sum was ever in play.

    An arm or comparison projection never reaches the aggregation policy. Giving
    it an empty provenance record would say that a sum was considered and left
    no trace, which is a different claim from the true one: that the question
    never arose.
    """
    if projection.policy_id or projection.aggregation_errors:
        who = projection.evaluator
        return AggregationProvenance(
            policy_id=projection.policy_id,
            policy_sha256=projection.policy_sha256,
            evaluator_id=who.evaluator_id if who else "",
            evaluator_version=who.evaluator_version if who else "",
            evaluator_hash=who.evaluator_hash if who else "",
            git_commit=who.git_commit if who else "",
            git_commit_matches_evaluator=bool(
                who and who.git_commit_matches_evaluator),
            evaluator_status=who.status if who else "unavailable",
            # From the RUNTIME, which is the only thing that knows the policy
            # that ran is the policy readiness cleared. Reading it off the
            # identity alone let a result name one policy and be vouched for by
            # another.
            release_eligible=bool(projection.runtime
                                  and projection.runtime.release_eligible),
            required_axes=list(projection.required_axes),
            aggregation_set=projection.aggregation_set,
            population_quote=projection.population_quote,
            timepoint_quote=projection.timepoint_quote,
            partition_quote=projection.partition_quote,
            component_quotes=[c.quote for c in projection.cohort_counts if c.quote],
            errors=list(projection.aggregation_errors),
            unrelated_rejections=list(projection.unrelated_rejections))
    return None


def to_source_result(projection: Projection) -> SourceValueResult:
    """One claim's projection, as the evidence object the audit already speaks."""
    counts = [CohortCount(label=c.arm_label, count=c.count, quote=c.quote)
              for c in projection.cohort_counts]
    # Everything the aggregation did travels on EVERY path, including the ones
    # that never used it. A malformed set is a fact about the response, and a
    # released printed total is not a reason to stop reporting it. An error the
    # reason already states is not repeated: the aggregator names the errors it
    # refused on, so most of the time these are the same words twice.
    reason = projection.aggregation_reason
    extra = [e for e in projection.aggregation_errors if e not in reason]
    reason = "; ".join(x for x in [reason, *extra] if x)
    # The batch decides the arm globally and anchors every quote at parse time;
    # it never runs the single-target cohort guard, so that verdict stays blank
    # rather than inheriting a default nobody earned.
    common = dict(cohort_check="", cohort_counts=counts,
                  aggregation_status=projection.aggregation_status,
                  aggregation_reason=reason,
                  aggregation_provenance=_provenance(projection))

    if projection.status == DERIVED:
        return SourceValueResult(
            found=True, value=str(projection.derived_value),
            value_origin="derived_sum", derivation=projection.derivation,
            # Supports the arithmetic, not the number. The components carry the
            # evidence for each addend.
            quote=projection.partition_quote,
            source_scope=projection.verified_scope,
            target_check="ok", evidence_check="ok", **common)

    if projection.status == OK and projection.entry is not None:
        entry = projection.entry
        return SourceValueResult(
            found=True, value=entry.value, unit=entry.unit, quote=entry.quote,
            value_origin="verbatim",
            # The parts, as verified against this reading's own quote. The value
            # itself is NOT rewritten from them: the verbatim string is what the
            # paper printed, and a components path that could change it would be
            # a second, quieter extractor.
            source_components=entry.verified_components,
            group_label_in_paper=entry.identity.arm_label,
            assigned_arm_label=entry.identity.arm_label,
            source_scope=entry.identity.population,
            target_check="ok", evidence_check="ok", **common)

    reason = projection.reason or "the batch did not answer this claim"
    return SourceValueResult(
        found=False, value=None, value_origin="unresolved",
        not_found_reason=reason,
        target_check=_TARGET_CHECK.get(projection.status, "unsupported"),
        target_reason=reason,
        # A contradiction is a fact about the paper, so what was read is kept
        # visible even though none of it may be released as the answer.
        cohorts_seen=([c.identity.arm_label for c in projection.candidates
                       if c.identity.arm_label]
                      if projection.status == CONTRADICTORY else []),
        evidence_check="ok", **common)
