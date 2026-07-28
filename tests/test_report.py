"""Tests for the deterministic HTML report renderer."""
from __future__ import annotations

from react_review.core.enums import AuditLabel, ReportVerdict
from react_review.report import render_html_report
from react_review.schemas.audit import MatchResult
from react_review.schemas.evidence import SourceEvidenceItem
from react_review.schemas.package import EvidencePackage
from react_review.schemas.report import AuditReport, FinalVerification, HumanReviewFlag


def _package() -> EvidencePackage:
    mr = MatchResult(
        study_id="ahmad_2022", group="t1dm", field_type="bmi",
        review_value="20.57 ± 1.7", review_unit="kg/m2",
        source_value="20.57 ± 1.77", source_unit="kg/m2",
        label=AuditLabel.MISMATCH, rel_error_pct=0.0, sd_rel_error_pct=3.95,
        reason="SD 1.7 vs 1.77 = 3.95% > 3%")
    src = SourceEvidenceItem(
        study_id="ahmad_2022", group="t1dm", field_type="bmi",
        source_value="20.57 ± 1.77", source_unit="kg/m2",
        source_quote="BMI (kg/m2) 20.57 ± 1.77", source_location_in_paper="Table 1")
    rep = AuditReport(run_id="demo", results=[mr], n_mismatch=1, verdict=ReportVerdict.FAIL)
    fv = FinalVerification(
        run_id="demo", verdict=ReportVerdict.FAIL,
        human_review_flags=[HumanReviewFlag(
            study_id="ahmad_2022", group="t1dm", field_type="bmi",
            label="mismatch", reason="SD 差异超容差")],
        summary="[FAIL] 0 match, 1 mismatch")
    return EvidencePackage(run_id="demo", source_items=[src], report=rep,
                           final_verification=fv)


def test_render_html_report():
    html = render_html_report(_package())
    assert html.startswith("<!doctype html>")
    assert "ahmad_2022" in html
    assert "20.57 ± 1.77" in html                     # source value
    assert "BMI (kg/m2) 20.57 ± 1.77" in html         # verbatim quote rendered
    assert "Table 1" in html                          # location
    assert "未通过" in html                            # FAIL verdict banner
    assert "不一致" in html                            # mismatch chip


def test_render_escapes_html():
    src = SourceEvidenceItem(study_id="s", group="t1dm", field_type="x",
                             source_quote="a <script>alert(1)</script> b")
    mr = MatchResult(study_id="s", group="t1dm", field_type="x",
                     label=AuditLabel.NOT_COMPARABLE)   # so the quote gets rendered
    rep = AuditReport(run_id="d", results=[mr], verdict=ReportVerdict.PASS)
    fv = FinalVerification(run_id="d", verdict=ReportVerdict.PASS, summary="ok")
    html = render_html_report(EvidencePackage(run_id="d", source_items=[src],
                                              report=rep, final_verification=fv))
    assert "<script>alert(1)</script>" not in html     # escaped, not injected
    assert "&lt;script&gt;" in html
