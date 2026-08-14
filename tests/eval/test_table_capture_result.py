"""The paired TableCapture decision is recomputed from its published metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.table_capture_result import ResultError, check_result


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "eval" / "table_capture_ab_v1_result.json"


def test_paired_result_records_exactly_the_preregistered_four_calls():
    report = check_result(RESULT)
    assert report["call_matrix"] == {
        ("eat_t1dm_review_2025", "table_capture_v1"),
        ("eat_t1dm_review_2025", "table_capture_v2"),
        ("melanoma_checkpoint_review_2017", "table_capture_v1"),
        ("melanoma_checkpoint_review_2017", "table_capture_v2"),
    }
    assert report["calls"] == 4


def test_v2_is_not_promoted_when_either_document_has_a_disqualifying_regression():
    report = check_result(RESULT)
    assert report["candidate_v2"] == "not_promoted"
    assert report["production_default"] == "table_capture_v1"
    assert "cell_accuracy_regressed" in report["documents"]["eat_t1dm_review_2025"]
    assert "hallucinated_cells_increased" in report["documents"]["eat_t1dm_review_2025"]
    assert "schema_failed" in report["documents"]["melanoma_checkpoint_review_2017"]


def test_result_proves_the_pair_used_the_same_inputs_and_controls():
    report = check_result(RESULT)
    assert report["paired_controls_verified"] is True
    assert report["response_hashes_unique"] is True
    assert report["candidate_domain_terms_introduced"] == 0


def test_a_claimed_promotion_that_contradicts_metrics_is_refused(tmp_path):
    body = json.loads(RESULT.read_text(encoding="utf-8"))
    body["decision"]["candidate_v2"] = "promoted"
    path = tmp_path / "result.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ResultError, match="candidate_v2"):
        check_result(path)
