"""Scoring logic for the full-pipeline accuracy eval (C1).

Pure + deterministic so it unit-tests offline; the ``eval/run_full_accuracy.py``
script supplies the (slow, LLM-backed) Collector and does the IO. Measures the
Collector + audit against the benchmark answer key:

- end-to-end audit label accuracy (predicted vs the hand-labeled expected_label);
- strict discrepancy precision/recall (automatic mismatch/unit_mismatch only);
- safety visibility (a true discrepancy must never be silently labelled MATCH);
- source extraction: found-rate + value-match-rate vs the hand-labeled source;
- collection_outcome breakdown (found / source_access_failed / missing_source).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict as dataclass_asdict, dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from react_review.audit import ToleranceTable
from react_review.schemas.audit import MatchResult
from react_review.tools.compare import CompareValuesTool
from react_review.tools.models import CompareInput
from react_review.core.enums import AuditLabel
from react_review.schemas.evidence import ReviewDataItem
from react_review.normalize import parse_numeric
from react_review.normalize.cohorts import distinguishing_tokens

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# A "flag" (positive for discrepancy detection) is any non-clean, comparable verdict.
FLAG_LABELS = {AuditLabel.MISMATCH.value, AuditLabel.UNIT_MISMATCH.value}


@dataclass
class RowResult:
    """One answer-key row after the Collector + audit ran on it."""

    study_id: str
    group: str
    field_type: str
    expected_label: str
    predicted_label: str
    expected_source: str | None
    extracted_source: str | None
    found: bool
    outcome: str
    extraction_correct: bool
    review_value: str = ""
    review_unit: str = ""
    source_unit: str = ""
    source_quote: str = ""
    source_location: str = ""
    source_file: str = ""
    source_uri: str = ""
    value_origin: str = ""
    derivation: str = ""
    cohort_counts: list[dict[str, Any]] = field(default_factory=list)
    aggregation_status: str = ""
    aggregation_reason: str = ""
    evidence_check: str = ""
    evidence_reason: str = ""
    reasons: list[dict[str, Any]] = field(default_factory=list)
    match_mode: str = ""
    match_reason: str = ""
    components_compared: list[str] = field(default_factory=list)
    components_unconsumed: list[str] = field(default_factory=list)
    review_numeric: dict[str, Any] = field(default_factory=dict)
    source_numeric: dict[str, Any] = field(default_factory=dict)
    # Benchmark contract carried into the result artifact.  These fields let a
    # reader distinguish a declared capability gap from a new regression
    # without joining the JSON back to the answer-key CSV by row position.
    audit_id: str = ""
    column_header: str = ""
    expected_match_mode: str = ""
    expected_review_required: bool = False
    expected_semantic_relation: str = ""
    known_gap: str = ""
    review_required: bool = False
    target_check: str = ""
    target_reason: str = ""
    assigned_arm_label: str = ""
    source_components: dict[str, Any] = field(default_factory=dict)
    component_status: str = ""
    semantic_relation: str = ""
    semantic_controls: dict[str, bool] = field(default_factory=dict)
    semantic: dict[str, Any] = field(default_factory=dict)


class _CollectorLike(Protocol):
    def collect(self, review_item: ReviewDataItem, reference: Any,
                *, research_context: str = "") -> Awaitable[Any]: ...


async def _compare(comparator, field_type: str, rv: Any, sv: Any, ru: str, su: str,
                   *, column_header: str = "", quote: str = "",
                   research_context: str = "",
                   source_components: Any = None) -> MatchResult:
    """Score through the TOOL, so the eval exercises the same path a run does.

    Calling ``compare_values`` directly here meant the eval could never reach the
    semantic fallback, and a number it reported would not be the number the
    pipeline produces.
    """
    return await comparator.run(CompareInput(
        field_type=field_type, review_value=rv, source_value=sv,
        review_unit=ru, source_unit=su, column_header=column_header,
        source_quote=quote,
        research_context=research_context,
        source_components=(source_components.model_dump()
                           if hasattr(source_components, "model_dump")
                           else source_components)))


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _semantic_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {}


async def run_rows(
    rows: list[dict[str, str]],
    collector: _CollectorLike,
    tol: ToleranceTable,
    reference_for: Callable[[str], Any],
    research_context: str = "",
    comparator: Any = None,
    targets: dict[str, Any] | None = None,
) -> list[RowResult]:
    """Collect + audit each answer-key row into a scored RowResult.

    ``targets`` carries the review-side facts the answer key does not hold — the
    review's own column label and its own word for the cohort. Without them the
    extractor was asked for a field_type and a slug, which is a materially
    easier question than the pipeline actually poses and hid the wrong-arm
    failures this eval is meant to measure.
    """
    comparator = comparator or CompareValuesTool(tol)
    # Extraction accuracy is a separate question from audit equivalence.  It
    # must not create extra semantic judgements with no source quote, nor make
    # the audit's semantic-call count depend on how the scorer grades extraction.
    extraction_comparator = CompareValuesTool(tol)
    results: list[RowResult] = []
    for r in rows:
        # Only a target contract may add to the question the extractor is
        # asked. Deriving a raw field name from the answer key's own column
        # header would change the prompt of every benchmark that has one — and
        # so invalidate its recorded replay — for a run that asked for no
        # profile at all.
        target = (targets or {}).get(r.get("audit_id", ""))
        extra = {} if target is None else {
            "raw_field_name": target.raw_field_name,
            "cohort_label": target.cohort_label,
            "timepoint": target.timepoint or "single",
        }
        review = ReviewDataItem(
            study_id=r["study_id"], group=(r.get("group") or "-"),
            field_type=r["field_type"], value=(r.get("review_value") or None),
            unit=(r.get("unit") or ""), column_header=(r.get("column_header") or ""),
            **extra,
        )
        res = await collector.collect(
            review, reference_for(review.study_id), research_context=research_context)
        si = res.source_item

        match = await _compare(
            comparator, review.field_type, review.value, si.source_value,
            review.unit, si.source_unit,
            column_header=(r.get("column_header") or ""), quote=si.source_quote,
            research_context=research_context,
            source_components=getattr(si, "source_components", None))
        predicted = match.label
        # Extraction is "correct" when the extracted value matches the human's
        # hand-labeled source value within tolerance.  A partial structured
        # match still needs review and therefore is not a complete extraction:
        # extracting only ``0.42`` from ``0.42 (99.5% CI 0.31-0.57)`` must not
        # earn the same credit as extracting the point estimate and interval.
        expected_source = r.get("source_value") or None
        extraction_match = await _compare(
            extraction_comparator, review.field_type, expected_source, si.source_value,
            (r.get("source_unit") or ""), si.source_unit,
            research_context=research_context)
        extraction_correct = (
            si.source_value is not None
            and extraction_match.label == AuditLabel.MATCH
            and not extraction_match.review_required
        )

        results.append(RowResult(
            study_id=review.study_id, group=review.group, field_type=review.field_type,
            expected_label=(r.get("expected_label") or ""),
            predicted_label=predicted.value,
            expected_source=expected_source, extracted_source=si.source_value,
            found=si.source_value is not None,
            outcome=si.collection_outcome.value, extraction_correct=extraction_correct,
            review_value=str(review.value), review_unit=review.unit,
            source_unit=si.source_unit, source_quote=si.source_quote,
            source_location=si.source_location_in_paper,
            source_file=si.source_file, source_uri=si.source_uri,
            value_origin=si.value_origin, derivation=si.derivation,
            cohort_counts=[c.model_dump(mode="json") for c in si.cohort_counts],
            aggregation_status=si.aggregation_status,
            aggregation_reason=si.aggregation_reason,
            evidence_check=si.evidence_check, evidence_reason=si.evidence_reason,
            reasons=[reason.model_dump(mode="json") for reason in si.reasons],
            match_mode=match.match_mode, match_reason=match.reason,
            components_compared=list(match.components_compared),
            components_unconsumed=list(match.components_unconsumed),
            review_numeric=dataclass_asdict(parse_numeric(review.value)),
            source_numeric=dataclass_asdict(parse_numeric(si.source_value)),
            audit_id=(r.get("audit_id") or ""),
            column_header=(r.get("column_header") or ""),
            expected_match_mode=(r.get("expected_match_mode") or ""),
            expected_review_required=_as_bool(r.get("expected_review_required")),
            expected_semantic_relation=(r.get("expected_semantic_relation") or ""),
            known_gap=(r.get("known_gap") or ""),
            review_required=match.review_required,
            source_components=(si.source_components.model_dump(mode="json")
                               if getattr(si, "source_components", None) else {}),
            component_status=(si.source_components.status
                              if getattr(si, "source_components", None) else ""),
            target_check=getattr(si, "target_check", ""),
            target_reason=getattr(si, "target_reason", ""),
            assigned_arm_label=getattr(si, "assigned_arm_label", ""),
            semantic_relation=match.semantic_relation,
            semantic_controls=dict(match.semantic_controls),
            semantic=_semantic_dict(match.semantic),
        ))
    return results


def score_rows(results: list[RowResult]) -> dict[str, Any]:
    """Aggregate RowResults into accuracy / P-R / extraction metrics."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    label_correct = sum(r.predicted_label == r.expected_label for r in results)

    tp = fp = fn = tn = 0
    expected_discrepancies = silent_releases = visible_discrepancies = 0
    escalated_not_comparable = 0
    for r in results:
        pred_pos = r.predicted_label in FLAG_LABELS
        exp_pos = r.expected_label in FLAG_LABELS
        if exp_pos:
            expected_discrepancies += 1
            # MATCH + review_required is visible in the production Judge and
            # cannot be called a silent release.  It remains a strict-recall FN
            # above: safety visibility and automatic detection are deliberately
            # separate metrics.
            if (r.predicted_label == AuditLabel.MATCH.value
                    and not r.review_required):
                silent_releases += 1
            else:
                visible_discrepancies += 1
                if r.predicted_label == AuditLabel.NOT_COMPARABLE.value:
                    escalated_not_comparable += 1
        if pred_pos and exp_pos:
            tp += 1
        elif pred_pos and not exp_pos:
            fp += 1
        elif exp_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else None)
    visibility_rate = (visible_discrepancies / expected_discrepancies
                       if expected_discrepancies else None)

    n_found = sum(r.found for r in results)
    n_extract_ok = sum(r.extraction_correct for r in results)

    # Target selection is counted separately from label accuracy, because those
    # two answer different questions. A row that correctly REFUSES an
    # unresolvable arm and a row that never located anything both come out as
    # "not found"; only the first is the guard working. And a row that quietly
    # returns another arm's number is the failure this phase exists to remove,
    # so it gets a count of its own that has to stay at zero.
    correct_target = wrong_target = 0
    for r in results:
        if not r.found:
            continue
        if _same_target(r.expected_source, r.extracted_source):
            correct_target += 1
        elif r.target_check in ("", "ok"):
            wrong_target += 1

    return {
        "n": n,
        "label_accuracy": label_correct / n,
        "discrepancy": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        },
        "safety": {
            "expected_discrepancies": expected_discrepancies,
            "silent_release_count": silent_releases,
            "visible_discrepancies": visible_discrepancies,
            "review_visibility_rate": visibility_rate,
            "escalated_not_comparable": escalated_not_comparable,
        },
        "extraction": {
            "found_rate": n_found / n,
            "value_match_rate": n_extract_ok / n,
        },
        "target": {
            "correct_target_found_count": correct_target,
            # Accepted a value that is not the arm's own, with nothing flagged.
            "wrong_target_accepted_count": wrong_target,
            "ambiguous_target_rejected_count": sum(
                1 for r in results if not r.found and r.target_check
                in {"ambiguous", "direction_inverted", "inconsistent"}),
            "unreported_target_count": sum(
                1 for r in results if not r.found
                and r.target_check in {"not_reported", "unsupported"}),
            "reassigned_count": sum(1 for r in results
                                    if r.target_check == "reassigned"),
            "checks": dict(Counter(r.target_check for r in results if r.target_check)),
        },
        "outcomes": dict(Counter(r.outcome for r in results)),
        "confusion": {f"{e}->{p}": c
                      for (e, p), c in Counter(
                          (r.expected_label, r.predicted_label) for r in results).items()},
    }


def _same_target(expected: str | None, got: str | None) -> bool:
    """Whether an extracted value is the one the answer key records for this arm.

    Numbers decide when both sides print one, so ``0.42`` and
    ``0.42 (99.5% CI 0.31-0.57)`` count as the same arm's value reported at
    different completeness — that is an interval-completeness question, not a
    wrong-arm one. Text values compare on their distinguishing words, so
    ``ipilimumab`` matches ``ipilimumab group`` while ``nivolumab plus
    ipilimumab`` does not.
    """
    left, right = str(expected or ""), str(got or "")
    if not left or not right:
        return False
    left_numbers = _NUMBER_RE.findall(left)
    right_numbers = _NUMBER_RE.findall(right)
    if left_numbers and right_numbers:
        return left_numbers[0] == right_numbers[0]
    return distinguishing_tokens(left) == distinguishing_tokens(right)


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def format_report(metrics: dict[str, Any]) -> str:
    if metrics.get("n", 0) == 0:
        return "no rows scored."
    d, e, s = metrics["discrepancy"], metrics["extraction"], metrics["safety"]
    lines = [
        "",
        "================ C1 accuracy ================",
        f"rows scored            : {metrics['n']}",
        f"audit label accuracy   : {_pct(metrics['label_accuracy'])}",
        "-- discrepancy detection (flag = mismatch/unit_mismatch) --",
        f"  precision            : {_pct(d['precision'])}  (tp={d['tp']} fp={d['fp']})",
        f"  recall               : {_pct(d['recall'])}  (fn={d['fn']} tn={d['tn']})",
        f"  f1                   : {_pct(d['f1'])}",
        "-- safety visibility (true discrepancy must not silently MATCH) --",
        f"  silent releases      : {s['silent_release_count']}",
        f"  review visibility    : {_pct(s['review_visibility_rate'])}  "
        f"(visible={s['visible_discrepancies']} "
        f"escalated={s['escalated_not_comparable']})",
        "-- target selection (which arm/comparison the value came from) --",
        f"  wrong target accepted: {metrics['target']['wrong_target_accepted_count']}"
        "   (must be 0)",
        f"  correct target found : {metrics['target']['correct_target_found_count']}"
        f"  reassigned={metrics['target']['reassigned_count']}",
        f"  refused              : ambiguous="
        f"{metrics['target']['ambiguous_target_rejected_count']} "
        f"unreported={metrics['target']['unreported_target_count']}",
        "-- source extraction --",
        f"  found rate           : {_pct(e['found_rate'])}",
        f"  value match rate     : {_pct(e['value_match_rate'])}",
        f"collection outcomes    : {metrics['outcomes']}",
        "=============================================",
    ]
    return "\n".join(lines)
