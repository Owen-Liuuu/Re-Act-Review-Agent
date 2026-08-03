"""The model proposes; these controls decide whether to believe it.

The first four cases are the ones demonstrated in the acceptance review, where
exact-string comparison called equivalent things different. The rest are the
adversarial half: a model claiming equivalence that must NOT be taken at its word.
"""
from __future__ import annotations

import pytest

from react_review.audit.semantic_control import (
    anchored_in,
    apply_semantic_control,
    numbers_agree,
)
from react_review.core.enums import AuditLabel
from react_review.tools.semantic_compare import SemanticVerdict


def _verdict(**kw) -> SemanticVerdict:
    base = dict(relation="same", equivalent=True, confidence=0.9,
                rationale="the same thing worded differently",
                review_normalized="", source_normalized="", evidence_span="")
    base.update(kw)
    return SemanticVerdict(**base)


def _control(review, source, verdict=None, quote="", **kw):
    return apply_semantic_control(verdict or _verdict(), review_value=review,
                                  source_value=source, source_quote=quote, **kw)


# --- the four failures the review demonstrated ---

def test_an_abbreviation_and_its_expansion_are_the_same():
    out = _control("ICU", "intensive care unit",
                   _verdict(relation="same", confidence=0.95, evidence_span="intensive care unit"),
                   quote="Patients were recruited in the intensive care unit.")
    assert out.label is AuditLabel.MATCH and not out.review_required


def test_a_broader_review_value_matches_but_must_be_seen():
    # "France" for a source that says "France, surgical ICU" is defensible and
    # also a loss of specificity — recording it as a plain match hides that.
    out = _control("France", "France, surgical ICU",
                   _verdict(relation="source_broader", confidence=0.9,
                            evidence_span="France, surgical ICU"),
                   quote="The trial ran in France, surgical ICU, 2019-2021.")
    assert out.label is AuditLabel.MATCH
    assert out.review_required is True


def test_a_differently_spelled_measure_is_the_same():
    out = _control("HR", "hazard ratio",
                   _verdict(relation="same", confidence=0.92, evidence_span="hazard ratio"),
                   quote="The hazard ratio for death was reported.")
    assert out.label is AuditLabel.MATCH


def test_wording_variants_of_a_quality_rating_are_the_same():
    # "Good" vs "Good Quality" — the exact-string metric counted these as wrong.
    out = _control("Good", "Good Quality",
                   _verdict(relation="same", confidence=0.9, evidence_span="Good Quality"),
                   quote="Overall study quality: Good Quality.")
    assert out.label is AuditLabel.MATCH


# --- adversarial: claims the controls must refuse ---

def test_different_numbers_are_never_the_same_thing():
    # The load-bearing control: a confident model cannot launder a numeric error.
    out = _control("6.60 mm", "3.83 mm", _verdict(relation="same", confidence=0.99))
    assert out.label is AuditLabel.MISMATCH
    assert out.failed_control == "numeric"


def test_a_confidence_interval_difference_survives_the_semantic_path():
    out = _control("0.62 (95% CI 0.48-0.81)", "0.62 (95% CI 0.30-1.10)",
                   _verdict(relation="same", confidence=0.99))
    assert out.label is AuditLabel.MISMATCH and out.failed_control == "numeric"


def test_an_invented_supporting_span_is_refused():
    out = _control("ICU", "intensive care unit",
                   _verdict(relation="same", confidence=0.95,
                            evidence_span="the intensive care unit of the study"),
                   quote="Patients were recruited from a ward.")
    assert out.label is AuditLabel.NOT_COMPARABLE
    assert out.failed_control == "anchor" and out.review_required


def test_a_negation_on_one_side_only_is_a_difference():
    out = _control("hypertension", "no hypertension",
                   _verdict(relation="same", confidence=0.95))
    assert out.label is AuditLabel.MISMATCH and out.failed_control == "polarity"


def test_low_confidence_is_not_a_verdict():
    out = _control("ICU", "ward", _verdict(relation="same", confidence=0.50))
    assert out.label is AuditLabel.NOT_COMPARABLE
    assert out.failed_control == "confidence" and out.review_required


def test_unknown_relation_is_not_a_verdict():
    out = _control("x", "y", _verdict(relation="unknown", confidence=0.99))
    assert out.label is AuditLabel.NOT_COMPARABLE and out.review_required


def test_the_model_calling_them_different_is_a_mismatch():
    out = _control("surgery", "chemotherapy",
                   _verdict(relation="different", equivalent=False, confidence=0.9))
    assert out.label is AuditLabel.MISMATCH


# --- the pieces ---

@pytest.mark.parametrize("span, quote, ok", [
    ("intensive care unit", "in the Intensive Care Unit (ICU)", True),   # case
    ("mean ± SD", "reported as mean  ±  SD", True),                       # spacing
    ("non-survivors", "among non survivors", True),                       # punctuation
    ("a ward", "in the intensive care unit", False),
    ("", "any quote", False),
])
def test_anchoring_is_normalised_containment_not_byte_equality(span, quote, ok):
    # Byte-exact matching would reject correct evidence over PDF spacing.
    assert anchored_in(span, quote) is ok


@pytest.mark.parametrize("review, source, agree", [
    ("6.60", "6.60", True),
    ("6.60", "3.83", False),
    ("45/120 (37.5%)", "37.5%", True),        # compared as a proportion
    ("45/120 (37.5%)", "44/120 (36.7%)", False),
    ("ICU", "intensive care unit", True),      # no numbers to disagree about
    ("Grade 3", "Grade 4", False),
])
def test_numeric_non_drift_compares_like_with_like(review, source, agree):
    assert numbers_agree(review, source, 0.01) is agree


def test_thresholds_are_configurable_and_recorded():
    strict = _control("ICU", "intensive care unit",
                      _verdict(confidence=0.8, evidence_span="intensive care unit"),
                      quote="the intensive care unit", min_confidence=0.9)
    assert strict.label is AuditLabel.NOT_COMPARABLE
    assert strict.checks["confidence"] is False


# --- how much rests on the uncalibrated threshold ---

def test_the_sensitivity_curve_shows_what_the_threshold_is_deciding():
    from react_review.audit.semantic_control import (
        format_threshold_sensitivity,
        threshold_sensitivity,
    )
    verdicts = [
        SemanticVerdict(relation="same", confidence=0.95),
        SemanticVerdict(relation="source_broader", confidence=0.72),
        SemanticVerdict(relation="same", confidence=0.55),
        SemanticVerdict(relation="different", confidence=0.99),   # not gated by it
    ]
    counts = threshold_sensitivity(verdicts)
    assert counts[0.50] == 3 and counts[0.70] == 2 and counts[0.90] == 1
    assert "0.70:2" in format_threshold_sensitivity(counts, len(verdicts))


def test_a_run_with_no_semantic_claims_reports_nothing_rather_than_zeroes():
    # The EAT benchmark escalates nothing (every value parses as a number), and a
    # row of "0.70:0" would read as a finding rather than as an empty path.
    from react_review.audit.semantic_control import (
        format_threshold_sensitivity,
        threshold_sensitivity,
    )
    assert format_threshold_sensitivity(threshold_sensitivity([]), 0) == ""
    # But claims that were all judged "different" ARE worth reporting as zeroes:
    # there the threshold genuinely decided nothing.
    counts = threshold_sensitivity([SemanticVerdict(relation="different", confidence=0.9)])
    assert format_threshold_sensitivity(counts, 1) != ""
