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
    assert m["extraction"]["found_rate"] == 4 / 5
    assert m["extraction"]["value_match_rate"] == 3 / 5
    assert m["outcomes"] == {"found": 4, "source_access_failed": 1}


def test_score_rows_empty():
    assert score_rows([])["n"] == 0


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
            collection_outcome=oc)
        return SimpleNamespace(source_item=si)


@pytest.mark.asyncio
async def test_run_rows_predicts_labels_and_extraction():
    rows = [
        {"study_id": "ahmad_2022", "group": "t1dm", "field_type": "eat_thickness",
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
    assert score_rows(res)["label_accuracy"] == 1.0
