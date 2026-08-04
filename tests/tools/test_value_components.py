"""Confidence intervals as components, and what happens when they go missing.

D6B-02: the extractor returned ``0.42`` from a sentence printing ``0.42; 99.5%
CI, 0.31 to 0.57``. The interval was read and dropped, so the comparator could
not tell that source from a paper reporting no interval at all, and the review's
own interval went unchecked.

These tests fix the contract in both directions: components that are not in the
evidence are refused, and evidence the components omit keeps the answer partial.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.audit import ToleranceTable, compare_values
from react_review.core.enums import AuditLabel
from react_review.llm.base import LLMBackend
from react_review.normalize.cohorts import parse_comparison
from react_review.schemas.evidence import SourceNumericComponents
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.compare import CompareValuesTool
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
)
from react_review.tools.models import CompareInput
from react_review.tools.value_components import (
    INCOMPLETE,
    OK,
    PROTOCOL_ERROR,
    canonical_value,
    parse_component_block,
    quote_states_interval,
    verify_components,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase7"
PAPER = (FIXTURES / "larkin_excerpt.txt").read_text(encoding="utf-8")
HR_SENTENCE = (
    "Significantly longer progression-free survival was observed in the "
    "nivolumab-plus-ipilimumab group than in the ipilimumab group (hazard ratio "
    "for death or disease progression, 0.42; 99.5% CI, 0.31 to 0.57; P<0.001) "
    "and in the nivolumab group than in the ipilimumab group (hazard ratio, "
    "0.57; 99.5% CI, 0.43 to 0.76; P<0.001).")
COMBO_VS_NIVO = (
    "The hazard ratio for the comparison between the nivolumab-plus-ipilimumab "
    "group and the nivolumab group was 0.74 (95% CI, 0.60 to 0.92).")


# --- reading the block ---------------------------------------------------

def test_components_are_read_as_numbers():
    parsed, error = parse_component_block(
        {"point_estimate": "0.42", "ci_level": 99.5, "ci_lower": 0.31,
         "ci_upper": "0.57"})
    assert error == ""
    assert parsed == {"point_estimate": 0.42, "ci_level": 99.5,
                      "ci_lower": 0.31, "ci_upper": 0.57}


def test_unreadable_components_are_refused():
    _, error = parse_component_block({"ci_level": "high"})
    assert "not a number" in error
    _, error = parse_component_block(["95", "0.31"])
    assert "not a structured object" in error


# --- verification against the quote --------------------------------------

def test_a_complete_interval_verifies():
    components, status, _ = verify_components(
        {"point_estimate": 0.74, "ci_level": 95.0, "ci_lower": 0.60,
         "ci_upper": 0.92},
        value="0.74 (95% CI, 0.60 to 0.92)", quote=COMBO_VS_NIVO)
    assert status == OK
    assert components.complete_interval
    assert components.ci_level == 95.0


def test_a_component_the_quote_does_not_print_is_a_protocol_error():
    _, status, reason = verify_components(
        {"point_estimate": 0.74, "ci_level": 95.0, "ci_lower": 0.60,
         "ci_upper": 0.99},
        value="0.74 (95% CI, 0.60 to 0.92)", quote=COMBO_VS_NIVO)
    assert status == PROTOCOL_ERROR and "not printed" in reason


def test_a_point_estimate_that_contradicts_the_value_is_a_protocol_error():
    _, status, reason = verify_components(
        {"point_estimate": 0.74}, value="0.42 (99.5% CI, 0.31 to 0.57)",
        quote=HR_SENTENCE)
    assert status == PROTOCOL_ERROR and "returns" in reason


def test_an_interval_the_quote_states_for_another_value_is_refused():
    """0.43-0.76 belongs to 0.57 in this sentence, not to 0.42."""
    _, status, reason = verify_components(
        {"point_estimate": 0.42, "ci_level": 99.5, "ci_lower": 0.43,
         "ci_upper": 0.76},
        value="0.42", quote=HR_SENTENCE,
        rival_values=["0.57 (99.5% CI, 0.43 to 0.76)"])
    assert status == PROTOCOL_ERROR
    assert "for this value" in reason


def test_the_recorded_partial_response_is_incomplete_not_complete():
    """The Phase 6B response itself: point estimate only, interval in the quote."""
    recorded = json.loads(
        (FIXTURES / "partial_ci_hr.json").read_text(encoding="utf-8"))["response"]
    components, status, reason = verify_components(
        {}, value=recorded["value"], quote=recorded["quote"],
        rival_values=["0.57"])
    assert status == INCOMPLETE
    assert components.missing == ["ci_level", "ci_lower", "ci_upper"]
    assert "did not return" in reason


def test_a_value_with_no_interval_in_its_quote_is_complete():
    components, status, _ = verify_components(
        {"point_estimate": 945.0}, value="945",
        quote="A total of 945 patients underwent randomization")
    assert status == OK and components.missing == []


def test_the_interval_scanner_reads_the_papers_own_wordings():
    assert quote_states_interval(COMBO_VS_NIVO, "0.74")
    assert quote_states_interval(
        "6.9 months (95% confidence interval [CI], 4.3 to 9.5) in the nivolumab "
        "group", "6.9")
    assert not quote_states_interval(
        "A total of 945 patients underwent randomization", "945")


def test_canonical_value_rebuilds_one_comparable_string():
    components = SourceNumericComponents(
        point_estimate=0.42, ci_level=99.5, ci_lower=0.31, ci_upper=0.57)
    assert canonical_value(components) == "0.42 (99.5% CI 0.31-0.57)"


# --- the components reach the comparator ---------------------------------

def test_components_supply_an_interval_the_verbatim_value_lacks():
    """MA013's shape: the source string is bare, the components are not."""
    without = compare_values(
        field_type="hazard_ratio", review_value="0.42 (95% CI 0.37-0.48)",
        source_value="0.42", review_unit="ratio", source_unit="ratio")
    assert without.label is AuditLabel.MATCH        # unverified, review-required
    assert without.components_unconsumed == ["ci"]

    with_components = compare_values(
        field_type="hazard_ratio", review_value="0.42 (95% CI 0.37-0.48)",
        source_value="0.42", review_unit="ratio", source_unit="ratio",
        source_components={"point_estimate": 0.42, "ci_level": 99.5,
                           "ci_lower": 0.31, "ci_upper": 0.57, "status": "ok"})
    assert with_components.label is AuditLabel.MISMATCH
    assert "confidence interval" in with_components.reason
    assert with_components.components_compared == ["ci"]
    # NB the LEVEL (95 vs 99.5) is still not a component here — that is 7C.


def test_components_never_overwrite_what_the_paper_printed():
    result = compare_values(
        field_type="hazard_ratio", review_value="0.42 (95% CI 0.31-0.57)",
        source_value="0.42 (99.5% CI 0.31-0.57)",
        source_components={"point_estimate": 0.42, "ci_lower": 9.9,
                           "ci_upper": 9.9, "status": "ok"})
    assert result.label is AuditLabel.MATCH


def test_a_refused_component_block_is_not_used():
    result = compare_values(
        field_type="hazard_ratio", review_value="0.42 (95% CI 0.37-0.48)",
        source_value="0.42",
        source_components={"ci_lower": 0.31, "ci_upper": 0.57,
                           "status": "protocol_error"})
    assert result.components_unconsumed == ["ci"]


@pytest.mark.asyncio
async def test_the_tool_passes_components_through():
    tool = CompareValuesTool(ToleranceTable())
    result = await tool.run(CompareInput(
        field_type="hazard_ratio", review_value="0.42 (95% CI 0.37-0.48)",
        source_value="0.42", review_unit="ratio", source_unit="ratio",
        source_components={"point_estimate": 0.42, "ci_level": 99.5,
                           "ci_lower": 0.31, "ci_upper": 0.57, "status": "ok"}))
    assert result.label is AuditLabel.MISMATCH


# --- end to end through the extraction contract --------------------------

class _ReplayBackend(LLMBackend):
    def __init__(self, response: dict) -> None:
        super().__init__()
        self._response = response

    @property
    def model_id(self) -> str:
        return "glm-4.5-flash"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        return json.dumps(self._response)


def _hr_input(**kw) -> ExtractSourceValueInput:
    body = dict(
        document=PaperDocument(paper_id="larkin_2015",
                               reference=ReferenceEntry(title="Larkin 2015"),
                               full_text=PAPER),
        field_type="hazard_ratio",
        group="nivolumab_plus_ipilimumab_vs_ipilimumab",
        raw_field_name="HR", unit_hint="ratio",
        cohorts={"nivolumab_plus_ipilimumab": ["Nivolumab (1 mg/kg) + ipilimumab (3 mg/kg)"],
                 "ipilimumab_plus_placebo": ["Ipilimumab (3 mg/kg) + placebo"],
                 "nivolumab_plus_placebo": ["Nivolumab (3 mg/kg) + placebo"]},
        extraction_profile="targeted_v4",
        comparison=parse_comparison("nivolumab_plus_ipilimumab_vs_ipilimumab"))
    body.update(kw)
    return ExtractSourceValueInput(**body)


def _comparison_entry(components: dict | None) -> dict:
    entry = {"left_label": "nivolumab-plus-ipilimumab group",
             "right_label": "ipilimumab group", "value": "0.42",
             "unit": "ratio", "quote": HR_SENTENCE}
    if components is not None:
        entry["value_components"] = components
    return entry


@pytest.mark.asyncio
async def test_extraction_returns_verified_components():
    response = {"found": True, "value": "0.42", "unit": "ratio",
                "quote": HR_SENTENCE, "location": "Results",
                "comparisons_reported": [_comparison_entry(
                    {"point_estimate": 0.42, "ci_level": 99.5,
                     "ci_lower": 0.31, "ci_upper": 0.57})]}
    result = await ExtractSourceValueTool(_ReplayBackend(response)).run(_hr_input())
    assert result.found is True
    assert result.source_components.ci_level == 99.5
    assert result.source_components.status == OK
    # The verbatim value is still what the paper printed.
    assert result.value == "0.42"


@pytest.mark.asyncio
async def test_extraction_marks_a_dropped_interval_incomplete():
    response = {"found": True, "value": "0.42", "unit": "ratio",
                "quote": HR_SENTENCE, "location": "Results",
                "comparisons_reported": [_comparison_entry(None)]}
    result = await ExtractSourceValueTool(_ReplayBackend(response)).run(_hr_input())
    assert result.found is True
    assert result.source_components.status == INCOMPLETE
    assert result.source_components.missing == ["ci_level", "ci_lower", "ci_upper"]


@pytest.mark.asyncio
async def test_extraction_refuses_components_the_quote_does_not_support():
    response = {"found": True, "value": "0.42", "unit": "ratio",
                "quote": HR_SENTENCE, "location": "Results",
                "comparisons_reported": [_comparison_entry(
                    {"point_estimate": 0.42, "ci_level": 95.0,
                     "ci_lower": 0.31, "ci_upper": 0.57})]}
    result = await ExtractSourceValueTool(_ReplayBackend(response)).run(_hr_input())
    assert result.found is False
    assert result.evidence_check == "protocol_error"
    assert "ci level 95 is not printed" in result.not_found_reason
