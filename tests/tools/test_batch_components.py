"""The parts of a batched reading, checked against its own quote and its rivals.

Phase 6B's failure arriving through the batch route: the extraction returned
`0.57` from a sentence printing `0.57; 99.5% CI, 0.43 to 0.76`, the interval was
parsed and then dropped on the way out, and the comparator saw a bare point
estimate — indistinguishable from a paper that reports no interval. The review's
own 95% CI went unchecked and the pair was called a MATCH.

The components were there the whole time. `batch_parse` read them; nothing
carried them to the result. So the gap was not "the model did not answer" but
"the answer was not used", which is the harder kind to see.
"""
from __future__ import annotations

import json

from react_review.audit.compare import compare_values
from react_review.schemas.batch import COMPARISON
from react_review.schemas.evidence import SourceNumericComponents
from react_review.tools.batch_parse import parse_batch

#: One sentence, two estimates, two intervals. The case attribution exists for.
TWO_ESTIMATES = (
    "Significantly longer progression-free survival was observed in the "
    "nivolumab-plus-ipilimumab group than in the ipilimumab group (hazard ratio "
    "for death or disease progression, 0.42; 99.5% CI, 0.31 to 0.57; P<0.001) "
    "and in the nivolumab group than in the ipilimumab group (hazard ratio, "
    "0.57; 99.5% CI, 0.43 to 0.76; P<0.001)")

DOCUMENT = "In this trial. " + TWO_ESTIMATES + " Further text follows."


def _reading(**over):
    body = {"left_label": "nivolumab", "right_label": "ipilimumab",
            "value": "0.57", "unit": "ratio", "quote": TWO_ESTIMATES,
            "value_components": {"point_estimate": "0.57", "ci_level": 99.5,
                                 "ci_lower": "0.43", "ci_upper": "0.76"}}
    body.update(over)
    return body


def _parse(readings, aggregable=False):
    return parse_batch({"readings": readings}, DOCUMENT,
                       target_shape=COMPARISON, aggregable=aggregable)


# --- attribution, which is the whole difficulty -----------------------------

def test_an_interval_is_only_this_estimates_when_no_rival_sits_nearer():
    """Both intervals are in the same sentence. 0.57 owns the second one."""
    reading = _parse([
        _reading(),
        _reading(left_label="nivolumab plus ipilimumab", value="0.42",
                 value_components={"point_estimate": "0.42", "ci_level": 99.5,
                                   "ci_lower": "0.31", "ci_upper": "0.57"}),
    ])
    verified = {entry.value: entry.verified_components
                for entry in reading.usable}
    assert verified["0.57"].ci_lower == 0.43 and verified["0.57"].ci_upper == 0.76
    assert verified["0.42"].ci_lower == 0.31 and verified["0.42"].ci_upper == 0.57
    assert all(c.status == "ok" for c in verified.values())


def test_a_reading_that_claims_its_rivals_interval_is_refused():
    """0.57 with 0.31–0.57 is the second sentence wearing the first's bounds."""
    reading = _parse([_reading(value_components={
        "point_estimate": "0.57", "ci_level": 99.5,
        "ci_lower": "0.31", "ci_upper": "0.57"})])
    assert reading.usable == []
    assert any("not" in r["reason"] for r in reading.rejected)


def test_a_protocol_error_rejects_only_its_own_entry():
    """The other readings were checked separately and are not implicated."""
    reading = _parse([
        _reading(),
        _reading(left_label="nivolumab plus ipilimumab", value="0.42",
                 value_components={"point_estimate": "0.42", "ci_level": 99.5,
                                   "ci_lower": "9.99", "ci_upper": "9.98"}),
    ])
    assert [e.value for e in reading.usable] == ["0.57"]
    assert len(reading.rejected) == 1


def test_a_component_not_printed_in_the_quote_is_refused():
    reading = _parse([_reading(value_components={
        "point_estimate": "0.57", "ci_level": 90, "ci_lower": "0.43",
        "ci_upper": "0.76"})])
    assert reading.usable == []
    assert "not printed" in reading.rejected[0]["reason"] or \
        "does not state" in reading.rejected[0]["reason"]


def test_a_reading_whose_quote_states_an_interval_it_omits_is_incomplete():
    """Never filled in from the text: a value the model did not report is not a
    value it read. What changes is that the omission stops being silent."""
    reading = _parse([_reading(value_components=None)])
    entry = reading.usable[0]
    assert entry.verified_components.status == "incomplete"
    assert set(entry.verified_components.missing) == {
        "ci_level", "ci_lower", "ci_upper"}
    assert entry.verified_components.ci_level is None


# --- and the result carries them --------------------------------------------

def test_the_verified_parts_reach_the_result_without_rewriting_the_value():
    from react_review.tools.batch_project import OK, Projection
    from react_review.tools.batch_result import to_source_result

    entry = _parse([_reading()]).usable[0]
    result = to_source_result(Projection(status=OK, entry=entry))
    assert result.value == "0.57", "the verbatim string is what the paper printed"
    assert result.source_components.ci_level == 99.5
    assert result.source_components.ci_upper == 0.76


# --- what the comparator now does with them ---------------------------------

def test_a_confidence_level_disagreement_is_now_visible():
    """MA015: the review reports 95% and the paper 99.5%. Their bounds are not
    the same quantity, and before the parts were carried it read as a MATCH."""
    verdict = compare_values(
        field_type="hazard_ratio", review_value="0.57 (95% CI 0.43-0.76)",
        source_value="0.57",
        source_components=SourceNumericComponents(
            point_estimate=0.57, ci_level=99.5, ci_lower=0.43, ci_upper=0.76,
            status="ok"))
    assert verdict.label.value == "mismatch"
    assert "99.5" in verdict.reason and "95" in verdict.reason


def test_an_incomplete_source_may_not_produce_a_bare_match():
    """Different in kind from a paper that reports no interval: here the
    interval is demonstrably in the evidence and was not carried out."""
    incomplete = SourceNumericComponents(
        point_estimate=0.57, status="incomplete",
        missing=["ci_level", "ci_lower", "ci_upper"])
    verdict = compare_values(
        field_type="hazard_ratio", review_value="0.57 (95% CI 0.43-0.76)",
        source_value="0.57", source_components=incomplete)
    assert verdict.label.value == "not_comparable"
    assert verdict.review_required


def test_a_paper_that_reports_no_interval_still_compares():
    """The refusal must not spread to the ordinary case: a source with nothing
    missing is judged exactly as before."""
    complete = SourceNumericComponents(point_estimate=0.57, status="ok")
    verdict = compare_values(
        field_type="hazard_ratio", review_value="0.57 (95% CI 0.43-0.76)",
        source_value="0.57", source_components=complete)
    assert verdict.label.value == "match"
    assert verdict.review_required, "unverified parts still force review"
