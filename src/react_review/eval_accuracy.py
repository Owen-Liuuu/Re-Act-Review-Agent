"""Scoring logic for the full-pipeline accuracy eval (C1).

Pure + deterministic so it unit-tests offline; the ``eval/run_full_accuracy.py``
script supplies the (slow, LLM-backed) Collector and does the IO. Measures the
Collector + audit against the benchmark answer key:

- end-to-end audit label accuracy (predicted vs the hand-labeled expected_label);
- discrepancy-detection precision/recall (a "flag" = mismatch or unit_mismatch);
- source extraction: found-rate + value-match-rate vs the hand-labeled source;
- collection_outcome breakdown (found / source_access_failed / missing_source).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from react_review.audit import ToleranceTable
from react_review.schemas.audit import MatchResult
from react_review.tools.compare import CompareValuesTool
from react_review.tools.models import CompareInput
from react_review.core.enums import AuditLabel
from react_review.schemas.evidence import ReviewDataItem

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
    expected_source: str
    extracted_source: str
    found: bool
    outcome: str
    extraction_correct: bool


class _CollectorLike(Protocol):
    def collect(self, review_item: ReviewDataItem, reference: Any,
                *, research_context: str = "") -> Awaitable[Any]: ...


async def _compare(comparator, field_type: str, rv: Any, sv: Any, ru: str, su: str,
                   *, quote: str = "", research_context: str = "") -> MatchResult:
    """Score through the TOOL, so the eval exercises the same path a run does.

    Calling ``compare_values`` directly here meant the eval could never reach the
    semantic fallback, and a number it reported would not be the number the
    pipeline produces.
    """
    return await comparator.run(CompareInput(
        field_type=field_type, review_value=rv, source_value=sv,
        review_unit=ru, source_unit=su, source_quote=quote,
        research_context=research_context))


async def run_rows(
    rows: list[dict[str, str]],
    collector: _CollectorLike,
    tol: ToleranceTable,
    reference_for: Callable[[str], Any],
    research_context: str = "",
    comparator: Any = None,
) -> list[RowResult]:
    """Collect + audit each answer-key row into a scored RowResult."""
    comparator = comparator or CompareValuesTool(tol)
    results: list[RowResult] = []
    for r in rows:
        review = ReviewDataItem(
            study_id=r["study_id"], group=(r.get("group") or "-"),
            field_type=r["field_type"], value=(r.get("review_value") or None),
            unit=(r.get("unit") or ""),
        )
        res = await collector.collect(
            review, reference_for(review.study_id), research_context=research_context)
        si = res.source_item

        predicted = (await _compare(
            comparator, review.field_type, review.value, si.source_value,
            review.unit, si.source_unit, quote=si.source_quote,
            research_context=research_context)).label
        # Extraction is "correct" when the extracted value matches the human's
        # hand-labeled source value within tolerance.
        expected_source = r.get("source_value") or None
        extraction_correct = si.source_value is not None and (await _compare(
            comparator, review.field_type, expected_source, si.source_value,
            (r.get("source_unit") or ""), si.source_unit,
            research_context=research_context)).label == AuditLabel.MATCH

        results.append(RowResult(
            study_id=review.study_id, group=review.group, field_type=review.field_type,
            expected_label=(r.get("expected_label") or ""),
            predicted_label=predicted.value,
            expected_source=str(expected_source), extracted_source=str(si.source_value),
            found=si.source_value is not None,
            outcome=si.collection_outcome.value, extraction_correct=extraction_correct,
        ))
    return results


def score_rows(results: list[RowResult]) -> dict[str, Any]:
    """Aggregate RowResults into accuracy / P-R / extraction metrics."""
    n = len(results)
    if n == 0:
        return {"n": 0}

    label_correct = sum(r.predicted_label == r.expected_label for r in results)

    tp = fp = fn = tn = 0
    for r in results:
        pred_pos = r.predicted_label in FLAG_LABELS
        exp_pos = r.expected_label in FLAG_LABELS
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

    n_found = sum(r.found for r in results)
    n_extract_ok = sum(r.extraction_correct for r in results)

    return {
        "n": n,
        "label_accuracy": label_correct / n,
        "discrepancy": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        },
        "extraction": {
            "found_rate": n_found / n,
            "value_match_rate": n_extract_ok / n,
        },
        "outcomes": dict(Counter(r.outcome for r in results)),
        "confusion": {f"{e}->{p}": c
                      for (e, p), c in Counter(
                          (r.expected_label, r.predicted_label) for r in results).items()},
    }


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def format_report(metrics: dict[str, Any]) -> str:
    if metrics.get("n", 0) == 0:
        return "no rows scored."
    d, e = metrics["discrepancy"], metrics["extraction"]
    lines = [
        "",
        "================ C1 accuracy ================",
        f"rows scored            : {metrics['n']}",
        f"audit label accuracy   : {_pct(metrics['label_accuracy'])}",
        "-- discrepancy detection (flag = mismatch/unit_mismatch) --",
        f"  precision            : {_pct(d['precision'])}  (tp={d['tp']} fp={d['fp']})",
        f"  recall               : {_pct(d['recall'])}  (fn={d['fn']} tn={d['tn']})",
        f"  f1                   : {_pct(d['f1'])}",
        "-- source extraction --",
        f"  found rate           : {_pct(e['found_rate'])}",
        f"  value match rate     : {_pct(e['value_match_rate'])}",
        f"collection outcomes    : {metrics['outcomes']}",
        "=============================================",
    ]
    return "\n".join(lines)
