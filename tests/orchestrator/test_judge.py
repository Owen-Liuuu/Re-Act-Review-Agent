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


def test_two_cells_of_the_same_field_do_not_overwrite_each_other():
    # Both rows are study/cohort/field-identical and differ only by cell. Keyed
    # on three fields, the second source item would replace the first and one
    # row's evidence would be attributed to the other row's number.
    from react_review.schemas.evidence import SourceEvidenceItem
    rep = AuditReport(
        run_id="r",
        results=[
            MatchResult(study_id="ahmad_2022", group="t1dm", field_type="eat_thickness",
                        table_id="t1", cell_ref=(0, 3), label=AuditLabel.NOT_COMPARABLE,
                        reason="no source value"),
            MatchResult(study_id="ahmad_2022", group="t1dm", field_type="eat_thickness",
                        table_id="t1", cell_ref=(1, 3), label=AuditLabel.NOT_COMPARABLE,
                        reason="no source value"),
        ],
        n_not_comparable=2, verdict=ReportVerdict.PARTIAL)
    source = [
        SourceEvidenceItem(study_id="ahmad_2022", group="t1dm", field_type="eat_thickness",
                           table_id="t1", cell_ref=(0, 3),
                           collection_outcome=CollectionOutcome.SOURCE_ACCESS_FAILED),
        SourceEvidenceItem(study_id="ahmad_2022", group="t1dm", field_type="eat_thickness",
                           table_id="t1", cell_ref=(1, 3),
                           collection_outcome=CollectionOutcome.MISSING_SOURCE),
    ]
    flags = Judge().adjudicate(rep, source).human_review_flags

    # each cell keeps ITS OWN outcome
    assert [f.label for f in flags] == ["source_access_failed", "missing_source"]
    assert [f.cell_ref for f in flags] == [(0, 3), (1, 3)]


def test_refusing_to_pair_is_not_reported_as_a_missing_source():
    # An ambiguous key must not read as "the paper doesn't say this", which the
    # report presents as a possible fabrication.
    from react_review.schemas.report import UnmatchedClaim
    rep = AuditReport(
        run_id="r", verdict=ReportVerdict.PARTIAL,
        unmatched_review=[UnmatchedClaim(
            study_id="ahmad_2022", group="t1dm", field_type="bmi",
            reason_code="ambiguous_match_key",
            message="2 claims share the key; refusing to guess")],
    )
    flag = Judge().adjudicate(rep).human_review_flags[0]
    assert flag.label == "ambiguous_match_key"
    assert "refusing to guess" in flag.reason


def test_candidate_contradicted_by_source_evidence_escalates_flag():
    from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
    rep = AuditReport(run_id="r", verdict=ReportVerdict.PASS)
    review = [ReviewDataItem(study_id="ahmad_2022", group="t1dm", field_type="eat_thickness",
                             raw_field_name="Weird fat", value="52",
                             resolution_status="candidate")]
    source = [SourceEvidenceItem(study_id="ahmad_2022", group="t1dm",
                                 field_type="eat_thickness", source_value="52",
                                 source_unit="cm3", concept_mismatch=True,
                                 concept_mismatch_reason="source unit 'cm3' is a different kind")]
    flags = Judge().adjudicate(rep, source, review).human_review_flags
    assert [f.label for f in flags] == ["concept_contradicted"]
    assert "contradicts" in flags[0].reason
