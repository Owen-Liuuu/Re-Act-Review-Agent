"""Tests for step 4: table comparison.

Scope note (P0): these tests assert the behaviour of the CURRENT comparator,
which joins student and model fields by their exact (lower-cased) field name
and dispatches by value type. Several richer behaviours the prototype once had
were removed in an earlier refactor and are deliberately NOT tested here — they
are re-introduction targets for P1 (field_type normalization):

  * canonical-key merging (e.g. student "N" == model "sample_size")
  * author-string normalization inside the comparator
  * NEEDS_REVIEW when an ungrouped student value faces grouped model values

Group-aware matching is currently achieved only when both sides use the SAME
verbatim field name (e.g. "age_t1dm"); that behaviour IS covered below.
"""
from __future__ import annotations

import pytest

from react_review.core.enums import (
    ComparisonFlagCode,
    FieldStatus,
    ReportVerdict,
)
from react_review.steps.data_extraction.schemas import ExtractedField, ExtractedTable
from react_review.steps.table_comparison.mock_impl import (
    MockReportGenerator,
    MockTableComparator,
)
from react_review.steps.table_comparison.real_impl import (
    RealReportGenerator,
    RealTableComparator,
    _normalise_numeric,
    _normalise_tool,
)


@pytest.fixture
def student_table() -> ExtractedTable:
    return ExtractedTable(
        paper_id="paper-001",
        fields=[
            ExtractedField(field_name="sample_size", value=200),
            ExtractedField(field_name="study_design", value="RCT"),
        ],
        extractor_id="student",
    )


@pytest.fixture
def model_tables() -> list[ExtractedTable]:
    return [
        ExtractedTable(
            paper_id="paper-001",
            fields=[
                ExtractedField(field_name="sample_size", value=200),
                ExtractedField(field_name="study_design", value="RCT"),
            ],
            extractor_id="model-a",
        ),
    ]


@pytest.mark.asyncio
async def test_mock_table_comparator(
    student_table: ExtractedTable, model_tables: list[ExtractedTable]
):
    comparator = MockTableComparator()
    result = await comparator.compare(student_table, model_tables)
    assert result.paper_id == "paper-001"
    assert len(result.field_diffs) == 2
    assert result.agreement_rate > 0.0


@pytest.mark.asyncio
async def test_mock_report_generator():
    from react_review.steps.table_comparison.schemas import TableComparisonResult

    comparisons = [
        TableComparisonResult(paper_id="paper-001", agreement_rate=0.9,
                              compared_count=1, total_count=1),
    ]
    generator = MockReportGenerator()
    report = await generator.generate(comparisons, run_id="test-run")
    assert report.run_id == "test-run"
    assert "1 paper" in report.summary
    assert "90.0%" in report.summary


# ----------------------------------------------------------------------
# Extractor-failure + missing-value fundamentals (attribution safety)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_failure_does_not_blame_student():
    """A failed extractor yields MISSING_MODEL + EXTRACTOR_GAP, never FIELD_MISMATCH."""
    student = ExtractedTable(
        paper_id="p1",
        fields=[ExtractedField(field_name="sample_size", value=200)],
        extractor_id="student",
    )
    model = [
        ExtractedTable(
            paper_id="p1",
            fields=[
                ExtractedField(
                    field_name="sample_size", value=None,
                    extractor_failed=True,
                ),
            ],
            extractor_id="model-a",
        ),
    ]
    cr = await RealTableComparator().compare(student, model)
    assert len(cr.field_diffs) == 1
    assert cr.field_diffs[0].status == FieldStatus.MISSING_MODEL
    assert cr.compared_count == 0
    codes = {f.code for f in cr.flags}
    assert ComparisonFlagCode.EXTRACTOR_GAP.value in codes
    assert ComparisonFlagCode.FIELD_MISMATCH.value not in codes


@pytest.mark.asyncio
async def test_missing_student_not_flagged_as_mismatch():
    student = ExtractedTable(paper_id="p4", fields=[], extractor_id="student")
    model = [
        ExtractedTable(
            paper_id="p4",
            fields=[ExtractedField(field_name="sample_size", value=123)],
            extractor_id="model-a",
        )
    ]
    cr = await RealTableComparator().compare(student, model)
    assert len(cr.field_diffs) == 1
    assert cr.field_diffs[0].status == FieldStatus.MISSING_STUDENT
    codes = {f.code for f in cr.flags}
    assert ComparisonFlagCode.FIELD_MISMATCH.value not in codes
    assert ComparisonFlagCode.EXTRACTOR_GAP.value not in codes


# ----------------------------------------------------------------------
# Normalisation unit tests
# ----------------------------------------------------------------------


def test_tool_normalization_synonyms():
    # A tool label maps to its canonical synonym key (not the long form).
    assert _normalise_tool("Echo") == "echo"
    assert _normalise_tool("Transthoracic Echocardiography") == "echo"
    assert _normalise_tool("CCTA") == "ccta"


def test_tool_normalization_verbose_ccta():
    """Verbose CT-angiography phrasings collapse to 'ccta'."""
    assert _normalise_tool("Coronary computed tomography angiography") == "ccta"
    assert _normalise_tool("coronary ct angiography") == "ccta"
    assert _normalise_tool("cardiac computed tomography angiography") == "ccta"


def test_numeric_normalization_variants():
    assert _normalise_numeric("52.3±8.1") == "52.3"
    assert _normalise_numeric("52.3 ± 8.1") == "52.3"
    assert _normalise_numeric("52.3+/-8.1") == "52.3"
    assert _normalise_numeric("52.3 (36.1-65.5)") == "52.3"
    assert _normalise_numeric("52,3") == "52.3"
    assert _normalise_numeric("52") == "52"


# ----------------------------------------------------------------------
# Same-name (group-suffix) matching — works because both sides use the
# identical verbatim field name. Canonical merging of DIFFERENT names is a
# P1 target and is not exercised here.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_name_grouped_fields_do_not_cross_cohorts():
    student = ExtractedTable(
        paper_id="p2",
        fields=[
            ExtractedField(field_name="age_t1dm", value="12.9"),
            ExtractedField(field_name="age_control", value="11.5"),
        ],
        extractor_id="student",
    )
    model = [
        ExtractedTable(
            paper_id="p2",
            fields=[
                ExtractedField(field_name="age_t1dm", value="11.5"),
                ExtractedField(field_name="age_control", value="12.9"),
            ],
            extractor_id="model-a",
        ),
    ]
    cr = await RealTableComparator().compare(student, model)
    diffs_by_name = {d.field_name: d for d in cr.field_diffs}
    assert diffs_by_name["age_t1dm"].status == FieldStatus.DIFF
    assert diffs_by_name["age_control"].status == FieldStatus.DIFF
    assert cr.compared_count == 2
    assert cr.agreement_rate == 0.0


@pytest.mark.asyncio
async def test_same_name_grouped_fields_match():
    student = ExtractedTable(
        paper_id="p3",
        fields=[
            ExtractedField(field_name="age_t1dm", value="12.9"),
            ExtractedField(field_name="age_control", value="11.5"),
        ],
        extractor_id="student",
    )
    model = [
        ExtractedTable(
            paper_id="p3",
            fields=[
                ExtractedField(field_name="age_t1dm", value="12.90±1.3"),
                ExtractedField(field_name="age_control", value="11.5"),
            ],
            extractor_id="model-a",
        ),
    ]
    cr = await RealTableComparator().compare(student, model)
    diffs_by_name = {d.field_name: d for d in cr.field_diffs}
    assert diffs_by_name["age_t1dm"].status == FieldStatus.MATCH
    assert diffs_by_name["age_control"].status == FieldStatus.MATCH
    assert cr.agreement_rate == 1.0
    assert cr.coverage_rate == 1.0


# ----------------------------------------------------------------------
# Report verdict / summary
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_gap_is_not_blamed_across_papers():
    """Across a mixed set, an extractor gap stays an EXTRACTOR_GAP — never a
    student FIELD_MISMATCH — and the report is produced over both papers.

    (The overall verdict for extractor-gap papers is intentionally NOT asserted
    here: how a gap paper's 0% agreement should factor into the average is a
    known open question flagged for a later phase.)
    """
    ok_student = ExtractedTable(
        paper_id="p_ok",
        fields=[
            ExtractedField(field_name="sample_size", value=100),
            ExtractedField(field_name="study_design", value="RCT"),
        ],
        extractor_id="student",
    )
    ok_model = [ExtractedTable(
        paper_id="p_ok",
        fields=[
            ExtractedField(field_name="sample_size", value=100),
            ExtractedField(field_name="study_design", value="RCT"),
        ],
        extractor_id="model-a",
    )]
    gap_student = ExtractedTable(
        paper_id="p_gap",
        fields=[ExtractedField(field_name="sample_size", value=50)],
        extractor_id="student",
    )
    gap_model = [ExtractedTable(
        paper_id="p_gap",
        fields=[ExtractedField(
            field_name="sample_size", value=None, extractor_failed=True
        )],
        extractor_id="model-a",
    )]

    cmp_ = RealTableComparator()
    cr_ok = await cmp_.compare(ok_student, ok_model)
    cr_gap = await cmp_.compare(gap_student, gap_model)

    # Clean paper: everything matches.
    assert cr_ok.agreement_rate == 1.0
    # Gap paper: the failed field is an extractor gap, not a student mismatch.
    assert cr_gap.field_diffs[0].status == FieldStatus.MISSING_MODEL
    gap_codes = {f.code for f in cr_gap.flags}
    assert ComparisonFlagCode.EXTRACTOR_GAP.value in gap_codes
    assert ComparisonFlagCode.FIELD_MISMATCH.value not in gap_codes

    # A report can still be produced over the mixed set.
    report = await RealReportGenerator().generate([cr_ok, cr_gap], run_id="r_mixed")
    assert report.compared_papers == 2


@pytest.mark.asyncio
async def test_low_coverage_summary_is_honest():
    """High agreement with very low coverage must carry a WARNING in the summary."""
    from react_review.steps.table_comparison.schemas import TableComparisonResult
    cr = TableComparisonResult(
        paper_id="p_cov",
        field_diffs=[],
        agreement_rate=0.9,
        coverage_rate=0.06,   # tiny
        compared_count=1,
        total_count=17,
        skipped=False,
    )
    report = await RealReportGenerator().generate([cr], run_id="r_cov")
    assert report.verdict == ReportVerdict.PARTIAL
    assert "WARNING" in report.summary
    assert "very low" in report.summary.lower()
    assert "comparable fields" in report.summary.lower()
