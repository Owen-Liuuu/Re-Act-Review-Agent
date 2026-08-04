"""Wrong-arm and wrong-comparison selection, offline.

Every case here is a real Phase 6B failure or its immediate neighbour, replayed
against the deterministic assignment with no model in the loop: the recorded
responses live in ``tests/fixtures/phase7``.

The two outcomes that matter are kept apart on purpose. Returning the requested
arm's own value is capability; refusing an arm that cannot be identified is
safety. A test that accepted either would not be able to tell them apart, and
neither would the metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.llm.base import LLMBackend
from react_review.normalize.cohorts import (
    label_affinity,
    parse_comparison,
)
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
)
from react_review.tools.target_assignment import (
    ArmEvidence,
    ComparisonEvidence,
    assign_arms,
    parse_arms,
    parse_comparisons,
    resolve_arm_target,
    resolve_comparison_target,
    resolve_sides,
    values_consistent,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase7"
PAPER = (FIXTURES / "larkin_excerpt.txt").read_text(encoding="utf-8")

# The review's own words for its three arms, as the Phase 7 target contract
# records them.
REVIEW = {
    "nivolumab_plus_ipilimumab": "Nivolumab (1 mg/kg) + ipilimumab (3 mg/kg)",
    "ipilimumab_plus_placebo": "Ipilimumab (3 mg/kg) + placebo",
    "nivolumab_plus_placebo": "Nivolumab (3 mg/kg) + placebo",
}
PFS_SENTENCE = (
    "The median progression-free survival was 6.9 months (95% confidence "
    "interval [CI], 4.3 to 9.5) in the nivolumab group, 11.5 months (95% CI, "
    "8.9 to 16.7) in the nivolumab-plus-ipilimumab group, and 2.9 months (95% "
    "CI, 2.8 to 3.4) in the ipilimumab group.")
HR_SENTENCE = (
    "Significantly longer progression-free survival was observed in the "
    "nivolumab-plus-ipilimumab group than in the ipilimumab group (hazard ratio "
    "for death or disease progression, 0.42; 99.5% CI, 0.31 to 0.57; P<0.001) "
    "and in the nivolumab group than in the ipilimumab group (hazard ratio, "
    "0.57; 99.5% CI, 0.43 to 0.76; P<0.001).")
HR_COMBO_VS_NIVO = (
    "The hazard ratio for the comparison between the nivolumab-plus-ipilimumab "
    "group and the nivolumab group was 0.74 (95% CI, 0.60 to 0.92).")


def _pfs_arms(values: dict[str, str] | None = None) -> list[ArmEvidence]:
    values = values or {
        "nivolumab group": "6.9 months (95% confidence interval [CI], 4.3 to 9.5)",
        "nivolumab-plus-ipilimumab group": "11.5 months (95% CI, 8.9 to 16.7)",
        "ipilimumab group": "2.9 months (95% CI, 2.8 to 3.4)",
    }
    return [ArmEvidence(label=label, value=value, unit="months",
                        quote=PFS_SENTENCE) for label, value in values.items()]


def _comparisons() -> list[ComparisonEvidence]:
    return [
        ComparisonEvidence(left_label="nivolumab-plus-ipilimumab group",
                           right_label="ipilimumab group",
                           value="0.42 (99.5% CI, 0.31 to 0.57)", unit="ratio",
                           quote=HR_SENTENCE),
        ComparisonEvidence(left_label="nivolumab group",
                           right_label="ipilimumab group",
                           value="0.57 (99.5% CI, 0.43 to 0.76)", unit="ratio",
                           quote=HR_SENTENCE),
        ComparisonEvidence(left_label="nivolumab-plus-ipilimumab group",
                           right_label="nivolumab group",
                           value="0.74 (95% CI, 0.60 to 0.92)", unit="ratio",
                           quote=HR_COMBO_VS_NIVO),
    ]


# --- the lexical primitives ---------------------------------------------

@pytest.mark.parametrize("raw,left,right", [
    ("nivolumab_plus_ipilimumab_vs_ipilimumab", "nivolumab plus ipilimumab",
     "ipilimumab"),
    ("nivolumab vs ipilimumab", "nivolumab", "ipilimumab"),
    ("A vs. B", "a", "b"),
    ("drug_a_versus_drug_b", "drug a", "drug b"),
])
def test_parse_comparison_reads_both_sides(raw, left, right):
    parsed = parse_comparison(raw)
    assert (parsed.left, parsed.right) == (left, right)


@pytest.mark.parametrize("raw", [
    "", "t1dm", "control", "a_vs_b_vs_c", "vs_b", "a_vs_", "advsersity",
])
def test_parse_comparison_refuses_what_is_not_a_pair(raw):
    assert parse_comparison(raw) is None


def test_comparison_direction_is_not_normalised_away():
    forward = parse_comparison("a_vs_b")
    assert (forward.left, forward.right) != (forward.right, forward.left)
    assert forward.inverted().left == forward.right


def test_affinity_separates_a_combination_arm_from_its_monotherapy():
    combo = REVIEW["nivolumab_plus_ipilimumab"]
    mono = REVIEW["nivolumab_plus_placebo"]
    assert (label_affinity(combo, "nivolumab-plus-ipilimumab group")
            > label_affinity(mono, "nivolumab-plus-ipilimumab group"))
    assert (label_affinity(mono, "nivolumab group")
            > label_affinity(mono, "nivolumab-plus-ipilimumab group"))


# --- assignment ----------------------------------------------------------

def test_all_three_arms_are_assigned_at_once():
    mapping, reason, margin = assign_arms(
        REVIEW, ["nivolumab group", "nivolumab-plus-ipilimumab group",
                 "ipilimumab group"])
    assert reason == ""
    assert mapping == {
        "nivolumab_plus_ipilimumab": "nivolumab-plus-ipilimumab group",
        "ipilimumab_plus_placebo": "ipilimumab group",
        "nivolumab_plus_placebo": "nivolumab group",
    }
    assert margin > 0


def test_one_at_a_time_would_tie_where_all_at_once_does_not():
    """The tie the global assignment exists to break, stated explicitly."""
    ipi_only = "ipilimumab group"
    assert (label_affinity(REVIEW["ipilimumab_plus_placebo"], ipi_only)
            == label_affinity(REVIEW["nivolumab_plus_ipilimumab"], ipi_only))
    mapping, _, _ = assign_arms(
        REVIEW, ["nivolumab-plus-ipilimumab group", ipi_only])
    assert mapping["ipilimumab_plus_placebo"] == ipi_only


def test_indistinguishable_arms_are_refused_not_split():
    mapping, reason, _ = assign_arms(
        {"a": "treatment", "b": "treatment"}, ["treatment", "treatment arm"])
    assert mapping == {} and "equally well" in reason


def test_more_arms_than_can_be_checked_is_refused():
    many = {f"k{i}": f"arm {i}" for i in range(9)}
    mapping, reason, _ = assign_arms(many, [f"arm {i}" for i in range(9)])
    assert mapping == {} and "not assigned deterministically" in reason


# --- D6B-01: the requested arm's own value -------------------------------

@pytest.mark.parametrize("key,expected", [
    ("nivolumab_plus_placebo",
     "6.9 months (95% confidence interval [CI], 4.3 to 9.5)"),
    ("nivolumab_plus_ipilimumab", "11.5 months (95% CI, 8.9 to 16.7)"),
    ("ipilimumab_plus_placebo", "2.9 months (95% CI, 2.8 to 3.4)"),
])
def test_each_arm_gets_its_own_value(key, expected):
    outcome = resolve_arm_target(target_key=key, review_labels=REVIEW,
                                 arms=_pfs_arms())
    assert outcome.ok and outcome.value == expected


def test_a_swapped_enumeration_is_caught_by_locality():
    """Right labels, right values, wrong pairing — the quote disagrees."""
    swapped = _pfs_arms({
        "nivolumab group": "11.5 months (95% CI, 8.9 to 16.7)",
        "nivolumab-plus-ipilimumab group":
            "6.9 months (95% confidence interval [CI], 4.3 to 9.5)",
        "ipilimumab group": "2.9 months (95% CI, 2.8 to 3.4)",
    })
    outcome = resolve_arm_target(target_key="nivolumab_plus_placebo",
                                 review_labels=REVIEW, arms=swapped)
    assert outcome.status == "inconsistent"
    assert "nearest" in outcome.reason


def test_locality_accepts_the_paper_as_written():
    ok, reason = values_consistent(PFS_SENTENCE, [
        ("nivolumab group", "6.9"),
        ("nivolumab-plus-ipilimumab group", "11.5"),
        ("ipilimumab group", "2.9")])
    assert ok is True and reason == ""


def test_an_arm_the_paper_does_not_report_is_not_substituted():
    outcome = resolve_arm_target(
        target_key="nivolumab_plus_placebo", review_labels=REVIEW,
        arms=[a for a in _pfs_arms() if a.label != "nivolumab group"])
    assert outcome.status == "not_reported"
    assert outcome.value is None


def test_no_enumeration_at_all_is_not_a_licence_to_guess():
    outcome = resolve_arm_target(target_key="nivolumab_plus_placebo",
                                 review_labels=REVIEW, arms=[])
    assert outcome.status == "not_reported" and outcome.value is None


# --- D6B-01: comparisons -------------------------------------------------

@pytest.mark.parametrize("group,expected", [
    ("nivolumab_plus_ipilimumab_vs_ipilimumab", "0.42 (99.5% CI, 0.31 to 0.57)"),
    ("nivolumab_plus_ipilimumab_vs_nivolumab", "0.74 (95% CI, 0.60 to 0.92)"),
    ("nivolumab_vs_ipilimumab", "0.57 (99.5% CI, 0.43 to 0.76)"),
])
def test_each_comparison_gets_its_own_hazard_ratio(group, expected):
    outcome = resolve_comparison_target(
        comparison=parse_comparison(group), review_labels=REVIEW,
        comparisons=_comparisons())
    assert outcome.ok and outcome.value == expected


def test_short_side_names_resolve_to_the_monotherapy_arms():
    """"nivolumab versus ipilimumab" is not about the combination arm."""
    sides, reason = resolve_sides(parse_comparison("nivolumab_vs_ipilimumab"),
                                  REVIEW)
    assert reason == ""
    assert sides == (REVIEW["nivolumab_plus_placebo"],
                     REVIEW["ipilimumab_plus_placebo"])


def test_a_comparison_the_paper_omits_is_not_answered_with_another_one():
    outcome = resolve_comparison_target(
        comparison=parse_comparison("nivolumab_plus_ipilimumab_vs_nivolumab"),
        review_labels=REVIEW, comparisons=_comparisons()[:1])
    assert outcome.status == "not_reported" and outcome.value is None


def test_the_mirror_image_comparison_is_refused_not_returned():
    inverted = [ComparisonEvidence(
        left_label="nivolumab group", right_label="nivolumab-plus-ipilimumab group",
        value="1.35", quote=HR_COMBO_VS_NIVO)]
    outcome = resolve_comparison_target(
        comparison=parse_comparison("nivolumab_plus_ipilimumab_vs_nivolumab"),
        review_labels=REVIEW, comparisons=inverted)
    assert outcome.status == "direction_inverted"
    assert outcome.value is None


# --- the enumeration must carry its own evidence -------------------------

def test_an_arm_quote_that_is_not_in_the_paper_is_rejected():
    arms, error = parse_arms(
        [{"label": "nivolumab group", "value": "6.9 months",
          "quote": "6.9 months in the nivolumab arm of this trial"}], PAPER)
    assert arms == [] and "contiguous passage" in error


def test_an_arm_quote_that_names_another_arm_only_is_rejected():
    arms, error = parse_arms(
        [{"label": "ipilimumab group", "value": "11.5 months",
          "quote": "11.5 months (95% CI, 8.9 to 16.7) in the "
                   "nivolumab-plus-ipilimumab group"}], PAPER)
    assert arms == [] and "does not name that arm" in error


def test_an_arm_quote_without_its_value_is_rejected():
    arms, error = parse_arms(
        [{"label": "nivolumab group", "value": "7.2 months",
          "quote": "6.9 months (95% confidence interval [CI], 4.3 to 9.5) in "
                   "the nivolumab group"}], PAPER)
    assert arms == [] and "does not contain the value" in error


def test_a_comparison_quote_must_name_both_sides():
    comparisons, error = parse_comparisons(
        [{"left_label": "nivolumab-plus-ipilimumab group",
          "right_label": "ipilimumab group", "value": "0.74",
          "quote": "The hazard ratio for the comparison between the "
                   "nivolumab-plus-ipilimumab group and the nivolumab group "
                   "was 0.74 (95% CI, 0.60 to 0.92)."}], PAPER)
    assert comparisons == []
    assert "direction cannot be confirmed" in error


def test_duplicate_arm_labels_are_rejected():
    """One arm may not be enumerated twice under two spellings."""
    arms, error = parse_arms(
        [{"label": "nivolumab group",
          "value": "6.9 months (95% confidence interval [CI], 4.3 to 9.5)",
          "quote": PFS_SENTENCE},
         {"label": "the nivolumab group", "value": "11.5 months (95% CI, 8.9 to 16.7)",
          "quote": PFS_SENTENCE}], PAPER)
    assert arms == [] and "same label" in error


# --- through the tool, on the recorded Phase 6B responses -----------------

class _ReplayBackend(LLMBackend):
    def __init__(self, response: dict) -> None:
        super().__init__()
        self._response = response
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "glm-4.5-flash"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return json.dumps(self._response)


def _recorded(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _input(**kw) -> ExtractSourceValueInput:
    body = dict(
        document=PaperDocument(paper_id="larkin_2015",
                               reference=ReferenceEntry(title="Larkin 2015"),
                               full_text=PAPER),
        field_type="progression_free_survival", group="nivolumab_plus_placebo",
        raw_field_name="PFS", unit_hint="months",
        cohort_display=REVIEW["nivolumab_plus_placebo"],
        cohorts={key: [label] for key, label in REVIEW.items()},
        extraction_profile="targeted_v4")
    body.update(kw)
    return ExtractSourceValueInput(**body)


@pytest.mark.asyncio
async def test_recorded_wrong_arm_response_no_longer_yields_the_wrong_value():
    """The Phase 6B response that answered 11.5 for the nivolumab arm."""
    recorded = _recorded("wrong_arm_pfs.json")["response"]
    result = await ExtractSourceValueTool(_ReplayBackend(recorded)).run(_input())
    # The response enumerates nothing, so there is no arm evidence to assign.
    assert result.found is False
    assert result.value is None
    assert result.target_check == "not_reported"
    assert result.not_found_reason


@pytest.mark.asyncio
async def test_an_enumerating_response_returns_the_requested_arm():
    """Same paper, same wrong top-level pick — but now with the enumeration."""
    response = {
        "found": True, "value": "11.5 months (95% CI, 8.9 to 16.7)",
        "unit": "months", "quote": PFS_SENTENCE,
        "group_label_in_paper": "nivolumab plus ipilimumab",
        "source_field_name": "median progression-free survival",
        "location": "Results",
        "arms_reported": [
            {"label": "nivolumab group",
             "value": "6.9 months (95% confidence interval [CI], 4.3 to 9.5)",
             "unit": "months", "quote": PFS_SENTENCE},
            {"label": "nivolumab-plus-ipilimumab group",
             "value": "11.5 months (95% CI, 8.9 to 16.7)", "unit": "months",
             "quote": PFS_SENTENCE},
            {"label": "ipilimumab group", "value": "2.9 months (95% CI, 2.8 to 3.4)",
             "unit": "months", "quote": PFS_SENTENCE},
        ],
    }
    result = await ExtractSourceValueTool(_ReplayBackend(response)).run(_input())
    assert result.found is True
    assert result.value == "6.9 months (95% confidence interval [CI], 4.3 to 9.5)"
    assert result.assigned_arm_label == "nivolumab group"
    # The model's own pick was the combination arm's value. That disagreement is
    # recorded rather than smoothed over.
    assert result.target_check == "reassigned"
    assert "11.5" in result.target_reason


@pytest.mark.asyncio
async def test_a_comparison_request_gets_the_requested_pair():
    response = {
        "found": True, "value": "0.42", "unit": "ratio", "quote": HR_SENTENCE,
        "source_field_name": "hazard ratio", "location": "Results",
        "comparisons_reported": [
            {"left_label": "nivolumab-plus-ipilimumab group",
             "right_label": "ipilimumab group", "value": "0.42 (99.5% CI, 0.31 to 0.57)",
             "unit": "ratio", "quote": HR_SENTENCE},
            {"left_label": "nivolumab-plus-ipilimumab group",
             "right_label": "nivolumab group", "value": "0.74 (95% CI, 0.60 to 0.92)",
             "unit": "ratio", "quote": HR_COMBO_VS_NIVO},
        ],
    }
    result = await ExtractSourceValueTool(_ReplayBackend(response)).run(_input(
        field_type="hazard_ratio", group="nivolumab_plus_ipilimumab_vs_nivolumab",
        raw_field_name="HR", unit_hint="ratio", cohort_display="",
        comparison=parse_comparison("nivolumab_plus_ipilimumab_vs_nivolumab")))
    assert result.found is True
    assert result.value == "0.74 (95% CI, 0.60 to 0.92)"
    assert result.target_check == "reassigned"


@pytest.mark.asyncio
async def test_a_study_level_row_is_not_forced_through_arm_assignment():
    """"-" is the whole study: there is no arm to assign, and none is demanded."""
    response = {"found": True, "value": "945", "unit": "count",
                "quote": "A total of 945 patients underwent randomization",
                "source_field_name": "randomized patients", "location": "Results"}
    result = await ExtractSourceValueTool(_ReplayBackend(response)).run(_input(
        field_type="sample_size", group="-", raw_field_name="Randomized patients, n",
        unit_hint="count", cohort_display=""))
    assert result.found is True and result.value == "945"
    assert result.target_check == "ok"


@pytest.mark.asyncio
async def test_the_targeted_output_skeleton_is_valid_json():
    """A malformed example teaches the model to answer malformed JSON."""
    backend = _ReplayBackend({"found": False, "not_found_reason": "probe"})
    await ExtractSourceValueTool(backend).run(_input(
        field_type="hazard_ratio", group="nivolumab_vs_ipilimumab",
        comparison=parse_comparison("nivolumab_vs_ipilimumab")))
    prompt = backend.prompts[-1]
    skeleton = prompt[prompt.index("## OUTPUT"):]
    body = skeleton[skeleton.index("{"):]
    parsed = json.loads(body.replace("true or false", "true"))
    assert "arms_reported" in parsed and "comparisons_reported" in parsed


@pytest.mark.asyncio
async def test_a_comparison_is_not_described_to_the_model_as_a_cohort():
    backend = _ReplayBackend({"found": False, "not_found_reason": "probe"})
    await ExtractSourceValueTool(backend).run(_input(
        field_type="hazard_ratio", group="nivolumab_vs_ipilimumab",
        comparison=parse_comparison("nivolumab_vs_ipilimumab")))
    target = backend.prompts[-1].split("## RULES")[0]
    assert 'comparison of "nivolumab" versus "ipilimumab"' in target
    assert "nivolumab_vs_ipilimumab" not in target
