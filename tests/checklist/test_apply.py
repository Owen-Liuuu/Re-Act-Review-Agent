"""Phase 5C checklist coverage is deterministic and never invents claims."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.checklist import (
    Checklist,
    ChecklistItem,
    annotate_checklist_claims,
    apply_checklist,
    merge_checklist_applications,
)
from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.table import CapturedTable, CapturedTableSet

DEFAULT = Path(__file__).resolve().parents[2] / "configs" / "checklists" / "default.yaml"


def _table() -> CapturedTableSet:
    return CapturedTableSet(tables=[CapturedTable(
        table_id="table_1", caption="Study quality",
        header_rows=[["Study", "Overall quality"]],
        rows=[["Smith 2020", "Good"], ["Jones 2021", "Fair"]],
        row_axis_columns=["Study"],
    )])


def test_default_checklist_loads_with_a_reproducible_artifact_identity():
    first = Checklist.from_yaml(DEFAULT)
    second = Checklist.from_yaml(DEFAULT)
    assert first.sha256 == second.sha256 and len(first.sha256) == 64
    assert Path(first.source_file) == DEFAULT.resolve()
    assert len({item.id for item in first.items}) == len(first.items)
    assert first.items[0].id == "risk_of_bias_assessment"
    assert all("source_paper" not in item.where for item in first.items)


def test_presence_is_coverage_only_and_per_study_gap_is_explicit():
    checklist = Checklist(name="test", items=[
        ChecklistItem(
            id="risk_present", question="Was risk of bias assessed?",
            required=True, scope="review", value_kind="presence",
            where=["review_text"], aliases=["risk of bias"]),
        ChecklistItem(
            id="quality_each", question="Quality for each study?",
            required=True, scope="per_study", value_kind="categorical",
            field_types=["overall_quality"], aliases=["overall quality"]),
    ])
    claims = [ReviewDataItem(
        study_id="smith_2020", group="-", field_type="overall_quality",
        raw_field_name="Overall quality", value="Good",
        table_id="table_1", cell_ref=(0, 1))]

    result = apply_checklist(
        checklist, claims, _table(),
        review_text="Methods: risk of bias was assessed independently.",
        study_ids=["smith_2020", "jones_2021"])

    assert [a.status for a in result.assessments] == ["covered", "partial"]
    assert result.assessments[0].evidence[0].source == "review_text"
    assert result.assessments[1].found == 1
    assert [(gap.checklist_id, gap.study_id) for gap in result.gaps] == [
        ("quality_each", "jones_2021")]
    # apply_checklist does not append a fake value for either the presence item
    # or the missing Jones judgement.
    assert len(claims) == 1 and claims[0].value == "Good"
    routed = annotate_checklist_claims(checklist, claims)
    assert routed[0].origin == "checklist"
    assert routed[0].checklist_id == "quality_each"
    assert claims[0].origin == "review_table"             # no in-place mutation


def test_structured_component_must_really_be_present():
    checklist = Checklist(name="ci", items=[ChecklistItem(
        id="ci", question="CI per study", required=True, scope="per_study",
        value_kind="numeric", field_types=["hazard_ratio"],
        required_components=["ci"])])
    claims = [
        ReviewDataItem(study_id="s1", field_type="hazard_ratio",
                       value="0.62 (95% CI 0.48-0.81)"),
        ReviewDataItem(study_id="s2", field_type="hazard_ratio", value="0.71"),
    ]
    result = apply_checklist(checklist, claims, CapturedTableSet())
    assert result.assessments[0].status == "partial"
    assert result.assessments[0].found == 1
    assert [(g.checklist_id, g.study_id) for g in result.gaps] == [("ci", "s2")]


def test_approved_study_ids_are_authoritative_and_two_passes_merge():
    checklist = Checklist(name="two-pass", sha256="hash", items=[
        ChecklistItem(
            id="risk", question="Risk assessed?", required=True,
            scope="review", value_kind="presence", where=["review_text"],
            aliases=["risk of bias"]),
        ChecklistItem(
            id="quality", question="Quality per approved study?", required=True,
            scope="per_study", value_kind="categorical",
            field_types=["overall_quality"]),
    ])
    claims = [
        ReviewDataItem(study_id="approved", field_type="overall_quality", value="Good"),
        ReviewDataItem(study_id="table_only", field_type="overall_quality", value="Fair"),
    ]

    review_pass = apply_checklist(
        checklist, claims, CapturedTableSet(), review_text="risk of bias",
        scopes={"review"}, evaluation_pass="review")
    study_pass = apply_checklist(
        checklist, claims, CapturedTableSet(), study_ids=["approved"],
        scopes={"per_study", "per_cohort"}, evaluation_pass="study_coverage")
    merged = merge_checklist_applications(review_pass, study_pass)

    assert [a.checklist_id for a in merged.assessments] == ["risk", "quality"]
    assert [a.evaluation_pass for a in merged.assessments] == [
        "review", "study_coverage"]
    assert merged.completed_passes == ["review", "study_coverage"]
    quality = merged.assessments[1]
    assert (quality.expected, quality.found, quality.status) == (1, 1, "covered")
    assert {e.study_id for e in quality.evidence} == {"approved"}
    assert merged.gaps == []


def test_empty_approved_study_list_does_not_fall_back_to_table_claims():
    checklist = Checklist(name="empty", items=[ChecklistItem(
        id="quality", question="Quality?", required=True, scope="per_study",
        value_kind="categorical", field_types=["overall_quality"])])
    claims = [ReviewDataItem(
        study_id="table_only", field_type="overall_quality", value="Good")]

    result = apply_checklist(
        checklist, claims, CapturedTableSet(), study_ids=[],
        scopes={"per_study"}, evaluation_pass="study_coverage")

    assert result.assessments[0].status == "not_applicable"
    assert result.assessments[0].expected == 0
    assert result.gaps == []


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "items:\n"
        "  - {id: same, question: one, aliases: [one]}\n"
        "  - {id: same, question: two, aliases: [two]}\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate checklist"):
        Checklist.from_yaml(path)


def test_source_paper_is_readable_in_old_models_but_rejected_for_execution(tmp_path):
    # Keep the enum value so historical EvidencePackages remain readable.
    historical = Checklist(name="old", items=[ChecklistItem(
        id="source", question="Source value?", where=["source_paper"],
        aliases=["value"])])
    assert historical.items[0].where == ["source_paper"]
    with pytest.raises(ValueError, match="source_paper is not supported"):
        apply_checklist(historical, [], CapturedTableSet())

    path = tmp_path / "unsupported.yaml"
    path.write_text(
        "items:\n"
        "  - id: source\n"
        "    question: Source value?\n"
        "    where: [source_paper]\n"
        "    aliases: [value]\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="source_paper is not supported"):
        Checklist.from_yaml(path)
