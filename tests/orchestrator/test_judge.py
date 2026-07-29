"""Judge: a missing source value splits into source_access_failed vs missing_source."""
from __future__ import annotations

from react_review.core.enums import AuditLabel, CollectionOutcome, ReportVerdict
from react_review.orchestrator.judge import Judge
from react_review.schemas.audit import MatchResult
from react_review.schemas.evidence import SourceEvidenceItem
from react_review.schemas.report import AuditReport


def _report(label: AuditLabel = AuditLabel.NOT_COMPARABLE) -> AuditReport:
    return AuditReport(
        run_id="r",
        results=[MatchResult(study_id="ahmad_2022", group="t1dm",
                             field_type="eat_thickness", label=label,
                             reason="no source value")],
        n_not_comparable=1,
        verdict=ReportVerdict.PARTIAL,
    )


def _source(outcome: CollectionOutcome) -> SourceEvidenceItem:
    return SourceEvidenceItem(study_id="ahmad_2022", group="t1dm",
                              field_type="eat_thickness", collection_outcome=outcome)


def test_missing_source_labels_potential_fabrication():
    fv = Judge().adjudicate(_report(), [_source(CollectionOutcome.MISSING_SOURCE)])
    assert len(fv.human_review_flags) == 1
    flag = fv.human_review_flags[0]
    assert flag.label == "missing_source"
    assert "fabrication" in flag.reason


def test_source_access_failed_labels_access_gap():
    fv = Judge().adjudicate(_report(), [_source(CollectionOutcome.SOURCE_ACCESS_FAILED)])
    assert fv.human_review_flags[0].label == "source_access_failed"


def test_falls_back_to_not_comparable_without_source_outcome():
    fv = Judge().adjudicate(_report())                       # no source_items at all
    assert fv.human_review_flags[0].label == "not_comparable"
    # a FOUND source that still couldn't be compared is a genuine not_comparable
    fv2 = Judge().adjudicate(_report(), [_source(CollectionOutcome.FOUND)])
    assert fv2.human_review_flags[0].label == "not_comparable"


def test_clean_match_produces_no_flag():
    fv = Judge().adjudicate(_report(AuditLabel.MATCH), [_source(CollectionOutcome.FOUND)])
    assert fv.human_review_flags == []


def test_concept_status_flags_candidate_and_unresolved():
    from react_review.schemas.evidence import ReviewDataItem
    rep = AuditReport(run_id="r", verdict=ReportVerdict.PASS)
    review = [
        ReviewDataItem(study_id="ahmad_2022", group="t1dm", field_type="hba1c",
                       raw_field_name="HbA1c", value="7.0", resolution_status="candidate"),
        ReviewDataItem(study_id="ahmad_2022", group="t1dm", field_type="",
                       raw_field_name="Novel Score", value="9", resolution_status="unresolved"),
        ReviewDataItem(study_id="ahmad_2022", group="t1dm", field_type="bmi",
                       value="24"),                                      # resolved → no flag
    ]
    flags = Judge().adjudicate(rep, None, review).human_review_flags
    assert [f.label for f in flags] == ["provisional_concept", "needs_review"]
    assert flags[1].field_type == "Novel Score"        # unresolved shows the raw name
