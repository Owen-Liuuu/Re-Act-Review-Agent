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
    # Which reading answered this row, and under what. A single-target row has
    # none of them, and `row_payload` drops all three rather than writing empty
    # strings: `asdict` would put them into every legacy row, which is a changed
    # report for a fact nothing recorded. See tests/test_legacy_bytes.py.
    batch_execution_id: str = ""
    batch_route: str = ""
    projection_status: str = ""
    # --- extraction quality, as orthogonal facts (metrics schema v2) ---------
    # One "extraction_correct" number answered several different questions at
    # once and got them wrong together: 313 against a gold 314 counted as a
    # correct extraction because it was within a relative band. Each question is
    # now asked separately, and the summary only calls an extraction ACCEPTED
    # when every necessary one is true.
    exact_value_match: bool = False
    value_within_tolerance: bool = False
    evidence_protocol_ok: bool = True
    component_complete: bool = True
    # true | false | not_assessable — graded against human GOLD, never against
    # the system's own guard. "unknown vs unknown" is not a correct answer.
    target_identity_correct: str = "not_assessable"
    target_scope_correct: str = "not_assessable"
    extraction_accepted: bool = False
    # What the run's own guards decided. Descriptive: available in production,
    # where there is no gold and therefore no notion of "correct".
    target_guard_status: str = ""
    scope_guard_status: str = ""
    review_scope: dict[str, Any] = field(default_factory=dict)
    source_scope: dict[str, Any] = field(default_factory=dict)
    scope_check: str = ""
    scope_reason: str = ""
    semantic_relation: str = ""
    semantic_controls: dict[str, bool] = field(default_factory=dict)
    semantic: dict[str, Any] = field(default_factory=dict)


class _CollectorLike(Protocol):
    def collect(self, review_item: ReviewDataItem, reference: Any,
                *, research_context: str = "") -> Awaitable[Any]: ...


async def _compare(comparator, field_type: str, rv: Any, sv: Any, ru: str, su: str,
                   *, column_header: str = "", quote: str = "",
                   research_context: str = "",
                   source_components: Any = None,
                   review_scope: Any = None,
                   source_scope: Any = None) -> MatchResult:
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
                           else source_components),
        review_scope=_as_dict(review_scope), source_scope=_as_dict(source_scope)))


def _graded(gold: Any, field: str, actual: str, same) -> str:
    """Three states, because "nobody said" is not "correct"."""
    expected = str(getattr(gold, field, "") or "").strip() if gold else ""
    if not expected:
        return "not_assessable"
    return "true" if same(expected, actual) else "false"


def _same_identity(expected: str, actual: str) -> bool:
    """Identity compared on distinguishing words, not on spelling.

    "Nivolumab plus Ipilimumab" and "nivolumab-plus-ipilimumab group" name the
    same arm; a byte comparison would grade the speller, not the extractor.
    """
    from react_review.normalize.cohorts import distinguishing_tokens

    return bool(actual) and distinguishing_tokens(expected) == distinguishing_tokens(actual)


def _same_scope(expected: str, actual: str) -> bool:
    from react_review.normalize.population import PopulationScope

    want, got = PopulationScope.parse(expected), PopulationScope.parse(actual or "")
    return want.basis == got.basis and want.analysis_set == got.analysis_set


def _exact_text(expected: Any, actual: Any) -> bool:
    return (expected is not None and actual is not None
            and str(expected).strip() == str(actual).strip())


def _target_scope(target: Any):
    """The population a target contract declares, if it declares one.

    Tolerant of a plain object standing in for a contract row: the review side
    simply has no declared scope then, which is the same answer a contract that
    omits the column gives.
    """
    from react_review.normalize.population import PopulationScope

    if hasattr(target, "scope"):
        return target.scope()
    declared = getattr(target, "population_scope", "")
    return PopulationScope.parse(declared, source="contract") if declared else None


def _as_dict(value: Any) -> dict | None:
    if value is None:
        return None
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _semantic_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value) if isinstance(value, dict) else {}


#: Fields a row has only when a batch produced it. Written when it did, absent
#: when it did not — never present-and-empty, which is a changed artifact.
BATCH_ROW_FIELDS = ("batch_execution_id", "batch_route", "projection_status")


def row_payload(result) -> dict:
    """One row as the report writes it.

    The single place that decides what a row looks like on disk, so a new field
    cannot reach a legacy artifact by way of a caller that used `asdict`
    directly and did not know there was a rule.
    """
    from dataclasses import asdict

    body = asdict(result)
    if not any(body.get(name) for name in BATCH_ROW_FIELDS):
        for name in BATCH_ROW_FIELDS:
            body.pop(name, None)
    return body


def _batch_field(source_item, name: str) -> str:
    """One field of a row's batch provenance, or "" where there was none.

    A single-target row has no batch provenance at all — not an empty one — so
    this reads through the absence rather than requiring every caller to.
    """
    provenance = getattr(source_item, "batch_provenance", None)
    return str(getattr(provenance, name, "") or "") if provenance else ""


class _Rows(list):
    """The rows, plus the readings that produced them.

    A list subclass rather than a new return type, because every caller already
    treats this as a list of rows and a batched run adds something ALONGSIDE
    them rather than changing what they are.
    """

    batch_readings: list = []


class _Collected:
    """Every row's source evidence, and the readings they came out of.

    Keyed by the position a row arrived at rather than by its identity, because
    an answer key may legitimately repeat a locator and the ORDER is what the
    caller asked about. The identity is checked against it, so a mismatch is an
    error rather than a silently misaligned column of results.
    """

    def __init__(self) -> None:
        self._by_position: dict[int, Any] = {}
        self.batch_readings: list[Any] = []

    def add(self, position: int, source_item: Any) -> None:
        self._by_position[position] = source_item

    def for_row(self, position: int) -> Any:
        return self._by_position[position]


async def _collect_all(collector, reviews, reference_for, opened, opener,
                       research_context) -> _Collected:
    """Collect every row, one study at a time.

    A batched read answers several rows from one response, so collection cannot
    be interleaved with comparison: a loop that compared as it went would either
    be unable to use a batch or would pay for one per row and discard it.
    """
    collected = _Collected()
    order: list[str] = []
    grouped: dict[str, list[tuple[int, Any]]] = {}
    for position, review in enumerate(reviews):
        if review.study_id not in grouped:
            grouped[review.study_id] = []
            order.append(review.study_id)
        grouped[review.study_id].append((position, review))

    collect_study = getattr(collector, "collect_study", None)
    seen: set[str] = set()
    for study_id in order:
        entries = grouped[study_id]
        reference = reference_for(study_id)
        if opener is not None and study_id not in opened:
            opened[study_id] = await opener(reference)
        source = opened.get(study_id)
        claims = [review for _, review in entries]

        if collect_study is not None:
            produced = await collect_study(
                claims, reference, research_context=research_context,
                **({"source": source} if source is not None else {}))
            results = produced.claim_results
            if len(results) != len(entries):
                raise RuntimeError(
                    f"{study_id}: {len(entries)} claims went in and "
                    f"{len(results)} came back; a result set that does not line "
                    "up cannot be assigned to rows")
            for (position, _), result in zip(entries, results):
                collected.add(position, result.source_item)
            for record in produced.batch_records:
                persistent = record.persistent()
                if persistent.execution_id not in seen:
                    seen.add(persistent.execution_id)
                    collected.batch_readings.append(persistent)
        else:
            for position, review in entries:
                result = await collector.collect(
                    review, reference, research_context=research_context,
                    **({"source": source} if source is not None else {}))
                collected.add(position, result.source_item)
    return collected


def review_items_for_rows(rows, targets) -> list[ReviewDataItem]:
    """The claims a benchmark run audits, built once.

    One function because a preflight that computes what a run WOULD send has to
    build the same claims the run builds. Two copies agreed on the day they were
    written and would drift on the first day somebody added a field to the target
    contract — and the drift would be invisible, because both sides would still
    look right on their own.
    """
    built: list[ReviewDataItem] = []
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
            "population_scope": _target_scope(target),
            "population_scope_source": getattr(
                target, "population_scope_source", ""),
        }
        built.append(ReviewDataItem(
            # The benchmark's own identity for this row. Production has
            # `review_data_id` already; here the answer key's `audit_id` is what
            # names a claim, and carrying it means nothing downstream has to
            # recover a row's identity from its position in a list.
            review_data_id=(r.get("audit_id") or ""),
            study_id=r["study_id"], group=(r.get("group") or "-"),
            field_type=r["field_type"], value=(r.get("review_value") or None),
            unit=(r.get("unit") or ""), column_header=(r.get("column_header") or ""),
            **extra,
        ))
    return built


async def run_rows(
    rows: list[dict[str, str]],
    collector: _CollectorLike,
    tol: ToleranceTable,
    reference_for: Callable[[str], Any],
    research_context: str = "",
    comparator: Any = None,
    targets: dict[str, Any] | None = None,
    gold: dict[str, Any] | None = None,
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
    results: _Rows = _Rows()
    # One retrieval per study, shared by that study's rows — the eval must pay
    # what a run pays, or its cost numbers describe a pipeline nobody runs.
    opened: dict[str, Any] = {}
    opener = getattr(collector, "open_study", None)

    # Collection happens per STUDY and comparison per row, in two passes rather
    # than one interleaved loop. A batched read answers several rows at once, so
    # a loop that collected and compared one row at a time could not use it —
    # and re-collecting per row would pay for the batch and then throw it away.
    reviews = review_items_for_rows(rows, targets)

    collected = await _collect_all(collector, reviews, reference_for, opened,
                                   opener, research_context)

    for position, (r, review) in enumerate(zip(rows, reviews)):
        si = collected.for_row(position)

        match = await _compare(
            comparator, review.field_type, review.value, si.source_value,
            review.unit, si.source_unit,
            column_header=(r.get("column_header") or ""), quote=si.source_quote,
            research_context=research_context,
            source_components=getattr(si, "source_components", None),
            review_scope=review.population_scope,
            source_scope=getattr(si, "population_scope", None))
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

        # The orthogonal facts. Each one answers exactly one question, and the
        # gold-graded ones stay "not_assessable" where no human said what the
        # right answer was — silence is not a pass.
        # Nothing was claimed about an identity or a population on a row that
        # produced no value, so there is nothing to grade — not a failure.
        gold_row = ((gold or {}).get(r.get("audit_id", ""))
                    if si.source_value is not None else None)
        identity_correct = _graded(
            gold_row, "expected_source_target_id",
            getattr(si, "assigned_arm_label", ""), _same_identity)
        scope_correct = _graded(
            gold_row, "expected_population_scope",
            (si.population_scope.describe()
             if getattr(si, "population_scope", None) else ""),
            _same_scope)
        components = getattr(si, "source_components", None)
        facts = {
            "exact_value_match": _exact_text(expected_source, si.source_value),
            "value_within_tolerance": extraction_correct,
            "evidence_protocol_ok": (si.evidence_check == "ok"
                                     and si.aggregation_status != "protocol_error"),
            "component_complete": (components is None
                                   or components.status != "incomplete"),
            "target_identity_correct": identity_correct,
            "target_scope_correct": scope_correct,
        }
        accepted = (
            si.source_value is not None
            and facts["value_within_tolerance"]
            and facts["evidence_protocol_ok"]
            and facts["component_complete"]
            and facts["target_identity_correct"] != "false"
            and facts["target_scope_correct"] != "false"
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
            batch_execution_id=_batch_field(si, "batch_execution_id"),
            batch_route=_batch_field(si, "route"),
            projection_status=_batch_field(si, "projection_status"),
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
            **facts, extraction_accepted=accepted,
            target_guard_status=getattr(si, "target_check", ""),
            scope_guard_status=match.scope_check,
            review_scope=(review.population_scope.model_dump(mode="json")
                          if review.population_scope else {}),
            source_scope=(si.population_scope.model_dump(mode="json")
                          if getattr(si, "population_scope", None) else {}),
            scope_check=match.scope_check, scope_reason=match.scope_reason,
            target_check=getattr(si, "target_check", ""),
            target_reason=getattr(si, "target_reason", ""),
            assigned_arm_label=getattr(si, "assigned_arm_label", ""),
            semantic_relation=match.semantic_relation,
            semantic_controls=dict(match.semantic_controls),
            semantic=_semantic_dict(match.semantic),
        ))
    # The readings, attached to the list the caller already holds. A row
    # names an execution id and this is where that reference resolves; a
    # reference pointing at nothing is worse than no reference at all.
    results.batch_readings = collected.batch_readings   # type: ignore[attr-defined]
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

    # Refusing is measured. A system that rejected every count would otherwise
    # show three green safety numbers and no capability at all, so how OFTEN the
    # scope question could be answered is reported next to how often it passed.
    scoped = [r for r in results if r.scope_check and r.scope_check != "not_required"]
    assessable = [r for r in scoped if r.scope_check in ("ok", "scope_mismatch")]
    unresolved_by_field: dict[str, int] = {}
    for r in scoped:
        if r.scope_check == "scope_unresolved":
            unresolved_by_field[r.field_type] = unresolved_by_field.get(r.field_type, 0) + 1
    n_found = sum(r.found for r in results)
    n_extract_ok = sum(r.extraction_correct for r in results)

    # Target selection is counted separately from label accuracy, because those
    # two answer different questions. A row that correctly REFUSES an
    # unresolvable arm and a row that never located anything both come out as
    # "not found"; only the first is the guard working. And a row that quietly
    # returns another arm's number is the failure this phase exists to remove,
    # so it gets a count of its own that has to stay at zero.
    # Graded against GOLD, never against the run's own guard: a system that
    # scored itself with its own verdict would be arguing, not measuring. Rows
    # with no gold are not counted either way.
    identity = Counter(r.target_identity_correct for r in results)
    scope_graded = Counter(r.target_scope_correct for r in results)
    # "Accepted" is meant literally: the audit let the value through as a clean
    # match. A row the audit refused, or flagged for review, got the identity or
    # the population wrong AND caught itself — that belongs in the distribution
    # above, not in a counter whose job is to stay at zero.
    def _released(r: RowResult) -> bool:
        return r.predicted_label == AuditLabel.MATCH.value and not r.review_required

    wrong_target = sum(1 for r in results
                       if r.target_identity_correct == "false" and _released(r))
    wrong_scope = sum(1 for r in results
                      if r.target_scope_correct == "false" and _released(r))
    correct_target = identity["true"]

    # The legacy counter, kept under its own name so Phase 7 artifacts remain
    # comparable. It compares VALUES, which is why it called 313-against-314 a
    # correct target in the first place.
    legacy_correct = legacy_wrong = 0
    for r in results:
        if not r.found:
            continue
        if _same_target(r.expected_source, r.extracted_source):
            legacy_correct += 1
        elif r.target_check in ("", "ok"):
            legacy_wrong += 1

    accepted = sum(r.extraction_accepted for r in results)
    return {
        "metrics_schema_version": 2,
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
        "extraction_quality": {
            "exact_value_match": sum(r.exact_value_match for r in results) / n,
            "value_within_tolerance": sum(r.value_within_tolerance for r in results) / n,
            "evidence_protocol_ok": sum(r.evidence_protocol_ok for r in results) / n,
            "component_complete": sum(r.component_complete for r in results) / n,
            "target_identity": dict(identity),
            "target_scope": dict(scope_graded),
            "extraction_accepted_rate": accepted / n,
        },
        "legacy_projection": {
            # Phase 6/7 definitions, kept so those artifacts stay comparable.
            # DEPRECATED: "correct" here means "within a relative band", which
            # is the confusion metrics schema v2 exists to end.
            "extraction_value_match_rate": n_extract_ok / n,
            "correct_target_by_value_count": legacy_correct,
            "wrong_target_by_value_count": legacy_wrong,
            "note": "value comparisons; see metrics.target.gold for gold-graded identity",
        },
        "target": {
            # DEPRECATED, kept with their Phase 6/7 meaning so those artifacts
            # stay readable: both compare VALUES, which is why 313 against a
            # gold 314 counted as the right target. The gold-graded answers are
            # under "gold" below; a name whose meaning changed silently would be
            # the very confusion this schema version exists to end.
            "correct_target_found_count": legacy_correct,
            "wrong_target_accepted_count": legacy_wrong,
            "gold": {
                "rows": identity["true"] + identity["false"],
                "identity_correct": identity["true"],
                "identity_wrong": identity["false"],
                # Released as a clean match while naming the wrong arm.
                "identity_wrong_released": wrong_target,
            },
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
        "scope": {
            "required": len(scoped),
            "scope_assessable_rate": (len(assessable) / len(scoped)) if scoped else None,
            "scope_resolved_rate": (sum(r.scope_check == "ok" for r in scoped) / len(scoped))
                                   if scoped else None,
            "unresolved_by_field": unresolved_by_field,
            "gold_rows": scope_graded["true"] + scope_graded["false"],
            "scope_wrong": scope_graded["false"],
            # Released as a clean match while counting the wrong population.
            "scope_wrong_released_count": wrong_scope,
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
    q = metrics["extraction_quality"]
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
        f"  wrong target released: {metrics['target']['gold']['identity_wrong_released']}"
        "   (must be 0, graded against gold)",
        f"  target identity gold : {metrics['target']['gold']['identity_correct']}"
        f" correct / {metrics['target']['gold']['rows']} graded"
        f"  reassigned={metrics['target']['reassigned_count']}",
        f"  refused              : ambiguous="
        f"{metrics['target']['ambiguous_target_rejected_count']} "
        f"unreported={metrics['target']['unreported_target_count']}",
        "-- population scope (only where a contract required it) --",
        f"  rows requiring scope : {metrics['scope']['required']}",
        f"  assessable           : {_pct(metrics['scope']['scope_assessable_rate'])}"
        f"  resolved: {_pct(metrics['scope']['scope_resolved_rate'])}",
        f"  wrong scope released : {metrics['scope']['scope_wrong_released_count']}"
        "   (must be 0, graded against gold)",
        f"  unresolved by field  : {metrics['scope']['unresolved_by_field'] or '{}'}",
        "-- source extraction (schema v2: orthogonal, gold-graded) --",
        f"  found rate           : {_pct(e['found_rate'])}",
        f"  accepted             : {_pct(q['extraction_accepted_rate'])}"
        "   (all necessary checks true)",
        f"  exact value          : {_pct(q['exact_value_match'])}"
        f"  within tolerance: {_pct(q['value_within_tolerance'])}",
        f"  evidence protocol ok : {_pct(q['evidence_protocol_ok'])}"
        f"  components complete: {_pct(q['component_complete'])}",
        f"  target identity      : {q['target_identity']}",
        f"  target scope         : {q['target_scope']}",
        f"collection outcomes    : {metrics['outcomes']}",
        "=============================================",
    ]
    return "\n".join(lines)
