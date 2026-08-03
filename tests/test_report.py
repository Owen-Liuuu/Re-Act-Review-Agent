"""Tests for the deterministic HTML report renderer."""
from __future__ import annotations

from react_review.core.enums import AuditLabel, ReportVerdict
from react_review.eval_accuracy import RowResult, score_rows
from react_review.report import (
    render_eval_report,
    render_html_report,
    render_parser_report,
)
from react_review.schemas.audit import MatchResult
from react_review.schemas.evidence import SourceEvidenceItem
from react_review.schemas.package import EvidencePackage
from react_review.schemas.report import AuditReport, FinalVerification, HumanReviewFlag
from react_review.schemas.semantic import SemanticVerdict


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
    assert "FAIL" in html                             # verdict banner (English)
    assert "Mismatch" in html                         # mismatch chip (English)
    assert "Evidence by source paper" in html         # grouped-by-study section


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


def test_final_checklist_verdict_controls_the_banner():
    rep = AuditReport(run_id="d", n_match=1, verdict=ReportVerdict.PASS)
    fv = FinalVerification(
        run_id="d", verdict=ReportVerdict.PARTIAL,
        human_review_flags=[HumanReviewFlag(
            field_type="risk", checklist_id="risk", label="checklist_gap",
            reason="required checklist item was not found")],
        summary="[PARTIAL] required checklist gap")
    html = render_html_report(EvidencePackage(
        run_id="d", report=rep, final_verification=fv))
    assert '<div class="vb">PARTIAL</div>' in html
    assert "Required checklist gap" in html


def test_checklist_identity_prevents_source_quotes_from_overwriting_in_report():
    results = [
        MatchResult(study_id="s", field_type="effect_size", checklist_id="primary",
                    label=AuditLabel.MATCH, source_value="0.4"),
        MatchResult(study_id="s", field_type="effect_size", checklist_id="secondary",
                    label=AuditLabel.MATCH, source_value="0.2"),
    ]
    sources = [
        SourceEvidenceItem(study_id="s", field_type="effect_size",
                           checklist_id="primary", source_quote="primary quote"),
        SourceEvidenceItem(study_id="s", field_type="effect_size",
                           checklist_id="secondary", source_quote="secondary quote"),
    ]
    rep = AuditReport(run_id="d", results=results, n_match=2, verdict=ReportVerdict.PASS)
    html = render_html_report(EvidencePackage(
        run_id="d", report=rep, source_items=sources,
        final_verification=FinalVerification(
            run_id="d", verdict=ReportVerdict.PASS, summary="ok")))
    assert "primary quote" in html and "secondary quote" in html


def test_a_model_reached_verdict_shows_its_reasoning_and_its_controls():
    # A semantic MATCH that renders as a plain MATCH is the opaque output the
    # report exists to prevent: the reader must see the claim AND the checks.
    verdict = SemanticVerdict(
        relation="source_broader", equivalent=True, confidence=0.88,
        rationale="the source names the same country plus the unit",
        evidence_span="France, surgical ICU")
    mr = MatchResult(study_id="s", group="t1dm", field_type="setting",
                     review_value="France", source_value="France, surgical ICU",
                     label=AuditLabel.MATCH, match_mode="semantic",
                     review_required=True, semantic=verdict,
                     semantic_relation="source_broader",
                     semantic_controls={"numeric": True, "anchor": True,
                                        "polarity": True, "confidence": True})
    src = SourceEvidenceItem(study_id="s", group="t1dm", field_type="setting",
                             source_quote="Conducted in France, surgical ICU.")
    rep = AuditReport(run_id="d", results=[mr], verdict=ReportVerdict.PASS)
    fv = FinalVerification(run_id="d", verdict=ReportVerdict.PASS, summary="ok")
    html = render_html_report(EvidencePackage(run_id="d", source_items=[src],
                                              report=rep, final_verification=fv))
    assert "source_broader" in html and "0.88" in html
    assert "the source names the same country plus the unit" in html
    assert "France, surgical ICU" in html               # the span it cited
    assert "anchor" in html and "polarity" in html      # which controls ran


def test_a_deterministic_verdict_carries_no_semantic_block():
    assert "semantic ·" not in render_html_report(_package())


def test_render_eval_report():
    rows = [
        RowResult("ahmad_2022", "t1dm", "bmi", expected_label="mismatch",
                  predicted_label="mismatch", expected_source="20.57 ± 1.77",
                  extracted_source="20.57 ± 1.77", found=True, outcome="found",
                  extraction_correct=True),
        RowResult("keles_2016", "t1dm", "eat_thickness", expected_label="unit_mismatch",
                  predicted_label="mismatch", expected_source="0.7",
                  extracted_source="4.8", found=True, outcome="found",
                  extraction_correct=False),
    ]
    html = render_eval_report(score_rows(rows), rows)
    assert html.startswith("<!doctype html>")
    assert "Benchmark Accuracy Report" in html
    assert "Per-row results" in html
    assert "ahmad_2022" in html and "keles_2016" in html


def test_render_parser_report():
    stats = {
        "n_gt": 101, "n_parser": 112, "n_matched": 40, "recall": 0.396,
        "precision": 0.357, "value_match": 0.95, "value_matched": 38,
        "missed": {"subgroup_n": 15, "country": 9}, "spurious": {"sample_size": 16},
        "mismatched_values": [{"study": "keles_2016", "group": "t1dm", "field": "age",
                               "parser_value": "34", "gt_value": "34 (29-39)"}],
    }
    html = render_parser_report(stats)
    assert html.startswith("<!doctype html>")
    assert "Parser Accuracy Report" in html
    assert "subgroup_n" in html and "sample_size" in html
    assert "Value mismatches" in html
