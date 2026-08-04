"""Offline tests for the C1 accuracy scorer + row runner (no LLM)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from react_review.audit import ToleranceTable
from react_review.core.enums import CollectionOutcome
from react_review.eval_accuracy import RowResult, run_rows, score_rows
from react_review.schemas.evidence import SourceEvidenceItem
from react_review.steps.paper_verification.schemas import ReferenceEntry


def _row(expected, predicted, *, found=True, outcome="found", extraction=True):
    return RowResult(
        study_id="s", group="g", field_type="f",
        expected_label=expected, predicted_label=predicted,
        expected_source="x", extracted_source="x",
        found=found, outcome=outcome, extraction_correct=extraction,
    )


def test_score_rows_metrics():
    rows = [
        _row("match", "match"),                                    # TN
        _row("mismatch", "mismatch"),                              # TP
        _row("mismatch", "match"),                                 # FN (missed)
        _row("match", "mismatch", extraction=False),               # FP
        _row("not_comparable", "not_comparable",
             found=False, outcome="source_access_failed", extraction=False),
    ]
    m = score_rows(rows)
    assert m["n"] == 5
    assert m["label_accuracy"] == 3 / 5                            # rows 1,2,5 correct
    assert m["discrepancy"] == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 2,
        "precision": 0.5, "recall": 0.5, "f1": 0.5,
    }
    assert m["safety"] == {
        "expected_discrepancies": 2,
        "silent_release_count": 1,
        "visible_discrepancies": 1,
        "review_visibility_rate": 0.5,
        "escalated_not_comparable": 0,
    }
    assert m["extraction"]["found_rate"] == 4 / 5
    assert m["extraction"]["value_match_rate"] == 3 / 5
    assert m["outcomes"] == {"found": 4, "source_access_failed": 1}


def test_score_rows_empty():
    assert score_rows([])["n"] == 0


def test_review_required_match_is_visible_but_not_strict_detection():
    row = _row("mismatch", "match")
    row.review_required = True

    metrics = score_rows([row])

    assert metrics["discrepancy"]["fn"] == 1
    assert metrics["safety"]["silent_release_count"] == 0
    assert metrics["safety"]["visible_discrepancies"] == 1
    assert metrics["safety"]["review_visibility_rate"] == 1.0


class _FakeCollector:
    """Returns a preset (value, unit, outcome) per (study, group, field_type)."""

    def __init__(self, source_map):
        self._m = source_map

    async def collect(self, review_item, reference, *, research_context=""):
        v, u, oc = self._m.get(
            (review_item.study_id, review_item.group, review_item.field_type),
            (None, "", CollectionOutcome.SOURCE_ACCESS_FAILED))
        si = SourceEvidenceItem(
            study_id=review_item.study_id, group=review_item.group,
            field_type=review_item.field_type, source_value=v, source_unit=u,
            source_quote=f"quoted {v}", source_file="C:/papers/source.pdf",
            value_origin="verbatim",
            collection_outcome=oc)
        return SimpleNamespace(source_item=si)


@pytest.mark.asyncio
async def test_run_rows_predicts_labels_and_extraction():
    rows = [
        {"study_id": "ahmad_2022", "group": "t1dm", "field_type": "eat_thickness",
         "audit_id": "A1", "column_header": "EAT thickness",
         "expected_match_mode": "numeric", "expected_review_required": "false",
         "review_value": "6.60 ± 0.71", "unit": "mm",
         "source_value": "6.60 ± 0.71", "source_unit": "mm", "expected_label": "match"},
        {"study_id": "ahmad_2022", "group": "t1dm", "field_type": "bmi",
         "review_value": "20.57 ± 1.7", "unit": "kg/m2",
         "source_value": "20.57 ± 1.77", "source_unit": "kg/m2",
         "expected_label": "mismatch"},   # SD 1.7 vs 1.77 = 3.95% > 3%
    ]
    smap = {
        ("ahmad_2022", "t1dm", "eat_thickness"): ("6.60 ± 0.71", "mm", CollectionOutcome.FOUND),
        ("ahmad_2022", "t1dm", "bmi"): ("20.57 ± 1.77", "kg/m2", CollectionOutcome.FOUND),
    }
    res = await run_rows(rows, _FakeCollector(smap), ToleranceTable(),
                         lambda sid: ReferenceEntry(title=sid))
    assert res[0].predicted_label == "match" and res[0].extraction_correct is True
    assert res[1].predicted_label == "mismatch" and res[1].extraction_correct is True
    assert res[0].source_quote == "quoted 6.60 ± 0.71"
    assert res[0].source_file == "C:/papers/source.pdf"
    assert res[0].source_unit == "mm" and res[0].review_unit == "mm"
    assert res[0].value_origin == "verbatim"
    assert res[0].audit_id == "A1" and res[0].column_header == "EAT thickness"
    assert res[0].expected_match_mode == "numeric"
    assert res[0].expected_review_required is False
    assert res[0].match_mode == "numeric" and res[0].match_reason
    assert res[0].review_numeric["primary"] == 6.60
    assert res[0].source_numeric["spread"] == 0.71
    assert score_rows(res)["label_accuracy"] == 1.0


@pytest.mark.asyncio
async def test_missing_source_with_residual_unit_is_null_and_visible():
    rows = [{
        "study_id": "keles_2016", "group": "t1dm",
        "field_type": "eat_thickness", "review_value": "0.7", "unit": "mm",
        "source_value": "0.7", "source_unit": "cm",
        "expected_label": "unit_mismatch",
    }]
    smap = {
        ("keles_2016", "t1dm", "eat_thickness"):
            (None, "cm", CollectionOutcome.MISSING_SOURCE),
    }

    result = (await run_rows(
        rows, _FakeCollector(smap), ToleranceTable(),
        lambda sid: ReferenceEntry(title=sid),
    ))[0]

    assert result.extracted_source is None
    assert result.predicted_label == "not_comparable"
    assert result.match_reason == "the source value is missing; units were not compared"
    safety = score_rows([result])["safety"]
    assert safety["silent_release_count"] == 0
    assert safety["review_visibility_rate"] == 1.0
    assert safety["escalated_not_comparable"] == 1


@pytest.mark.asyncio
async def test_partial_structured_extraction_does_not_earn_value_match_credit():
    rows = [{
        "study_id": "trial", "group": "arm", "field_type": "hazard_ratio",
        "review_value": "0.42 (95% CI 0.31-0.57)", "unit": "ratio",
        "source_value": "0.42 (99.5% CI 0.31-0.57)", "source_unit": "ratio",
        "expected_label": "match",
    }]
    smap = {
        ("trial", "arm", "hazard_ratio"):
            ("0.42", "ratio", CollectionOutcome.FOUND),
    }

    result = (await run_rows(
        rows, _FakeCollector(smap), ToleranceTable(),
        lambda sid: ReferenceEntry(title=sid),
    ))[0]

    assert result.predicted_label == "match"
    assert result.review_required is True
    assert result.extraction_correct is False


class _RecordingCollector(_FakeCollector):
    """Keeps the review item it was asked about, so the question can be checked."""

    def __init__(self, source_map):
        super().__init__(source_map)
        self.asked: list = []

    async def collect(self, review_item, reference, *, research_context=""):
        self.asked.append(review_item)
        return await super().collect(review_item, reference,
                                     research_context=research_context)


_TARGET_ROW = {
    "study_id": "larkin_2015", "group": "nivolumab_plus_placebo",
    "field_type": "progression_free_survival", "audit_id": "MA011",
    "column_header": "PFS", "review_value": "6.9 (95% CI 4.3-9.5)",
    "unit": "months", "source_value": "6.9 (95% CI 4.3-9.5)",
    "source_unit": "months", "expected_label": "match",
}


@pytest.mark.asyncio
async def test_the_target_contract_reaches_the_extraction_question():
    """The review's own column label and cohort words must be asked for."""
    collector = _RecordingCollector({
        ("larkin_2015", "nivolumab_plus_placebo", "progression_free_survival"):
            ("6.9 (95% CI 4.3-9.5)", "months", CollectionOutcome.FOUND)})
    targets = {"MA011": SimpleNamespace(
        raw_field_name="PFS", cohort_label="Nivolumab (3 mg/kg) + placebo",
        timepoint="median_pfs")}
    await run_rows([_TARGET_ROW], collector, ToleranceTable(),
                   lambda sid: ReferenceEntry(title=sid), targets=targets)
    asked = collector.asked[0]
    assert asked.raw_field_name == "PFS"
    assert asked.cohort_label == "Nivolumab (3 mg/kg) + placebo"
    assert asked.timepoint == "median_pfs"


@pytest.mark.asyncio
async def test_without_a_contract_the_question_is_unchanged():
    """No profile, no additions — a recorded replay must stay reachable."""
    collector = _RecordingCollector({
        ("larkin_2015", "nivolumab_plus_placebo", "progression_free_survival"):
            ("6.9 (95% CI 4.3-9.5)", "months", CollectionOutcome.FOUND)})
    await run_rows([_TARGET_ROW], collector, ToleranceTable(),
                   lambda sid: ReferenceEntry(title=sid))
    asked = collector.asked[0]
    assert asked.raw_field_name == ""
    assert asked.cohort_label == ""
    assert asked.timepoint == "single"
    # …while the answer key's own column header still reaches the comparator.
    assert asked.column_header == "PFS"


def test_target_counters_separate_a_refusal_from_a_wrong_value():
    correct = _row("match", "match")
    correct.expected_source, correct.extracted_source = "6.9 months", "6.9 months"
    wrong = _row("match", "mismatch", extraction=False)
    wrong.expected_source, wrong.extracted_source = "6.9 months", "11.5 months"
    refused = _row("match", "not_comparable", found=False,
                   outcome="missing_source", extraction=False)
    refused.target_check = "ambiguous"

    target = score_rows([correct, wrong, refused])["target"]
    assert target["correct_target_found_count"] == 1
    assert target["wrong_target_accepted_count"] == 1
    assert target["ambiguous_target_rejected_count"] == 1


def test_a_partial_interval_is_not_counted_as_a_wrong_target():
    """0.42 out of "0.42 (99.5% CI …)" is incompleteness, not the wrong arm."""
    partial = _row("mismatch", "match")
    partial.expected_source = "0.42 (99.5% CI 0.31-0.57)"
    partial.extracted_source = "0.42"

    target = score_rows([partial])["target"]
    assert target["wrong_target_accepted_count"] == 0
    assert target["correct_target_found_count"] == 1
