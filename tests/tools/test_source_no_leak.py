"""Source-side extraction must never see the review's reported cell value."""
from __future__ import annotations

import json
from typing import get_args, get_origin

import pytest
from pydantic import ValidationError

from react_review.schemas.evidence import ReviewDataItem
from react_review.tools.extract_source import (
    SourceQuery,
    render_source_extract_prompt,
    source_query_from_claim,
)

SENTINEL = "REVIEW_LEAK_6.60±0.71_SENTINEL"


def _is_text_only(annotation: object) -> bool:
    if annotation is str:
        return True
    if get_origin(annotation) is list:
        return get_args(annotation) == (str,)
    return False


def test_source_query_has_no_slot_for_a_numeric_review_value():
    names = set(SourceQuery.model_fields)
    assert "value" not in names
    assert "review_value" not in names
    assert not any(name.startswith("review_") for name in names)
    for name, info in SourceQuery.model_fields.items():
        assert _is_text_only(info.annotation), (
            f"SourceQuery.{name} is {info.annotation!r}; the envelope must not "
            "be able to hold a review cell value")


def test_source_query_rejects_a_value_field():
    with pytest.raises(ValidationError):
        SourceQuery(concept="age", value=SENTINEL)  # type: ignore[call-arg]


def test_rendered_source_prompt_does_not_contain_the_review_value():
    claim = ReviewDataItem(
        study_id="aslan_2015",
        group="t1dm",
        field_type="bmi",
        raw_field_name="BMI Kg/m2",
        value=SENTINEL,
        unit="kg/m2",
        cohort_label="T1DM",
    )
    query = source_query_from_claim(
        claim,
        concept="body mass index",
        concept_variants=["BMI", "body mass index"],
        research_context="an audit of a systematic review",
    )
    dumped = json.dumps(query.model_dump(), ensure_ascii=False)
    assert SENTINEL not in dumped

    prompt = render_source_extract_prompt(
        query,
        paper_text="Table 1. Age (years) 31 ± 8 for the T1DM column.",
        raw_label=claim.raw_field_name,
        group=claim.group,
        cohort_display=claim.cohort_label,
        cohorts={"t1dm": ["T1DM"], "control": ["Controls"]},
    )
    assert SENTINEL not in prompt
