"""The locate and transcribe prompts cannot change without a new version.

``legacy_v3`` / ``targeted_v4`` / ``targeted_v6`` / ``targeted_v5_batch`` stay
frozen. These two files pin the new pair. Comments and helper names are outside
the boundary; a different question is not.
"""
from __future__ import annotations

import inspect

import pytest

from react_review.prompt_contract import (
    BATCH_LOCATE_V1,
    BATCH_TRANSCRIBE_V1,
    PromptContract,
)
from react_review.tools.batch_prompt import _BODIES
from react_review.tools.batch_split import (
    BATCH_LOCATE_VERSION,
    BATCH_TRANSCRIBE_VERSION,
    build_batch_locate_prompt,
    build_batch_transcribe_prompt,
)
from react_review.tools.extraction_profile import BATCH_SPLIT_PROFILE, PROMPT_VERSIONS

LOCATE = PromptContract.load(BATCH_LOCATE_V1)
TRANSCRIBE = PromptContract.load(BATCH_TRANSCRIBE_V1)
LOCATE_CASES = {case.case_id: case for case in LOCATE.cases}
TRANSCRIBE_CASES = {case.case_id: case for case in TRANSCRIBE.cases}


@pytest.mark.parametrize("case_id", sorted(LOCATE_CASES))
def test_locate_still_asks_what_it_was_published_asking(case_id):
    case = LOCATE_CASES[case_id]
    from react_review.prompt_contract import sha256_text

    assert sha256_text(case.render(LOCATE.renderer)) == case.rendered_sha256.upper()


@pytest.mark.parametrize("case_id", sorted(TRANSCRIBE_CASES))
def test_transcribe_still_asks_what_it_was_published_asking(case_id):
    case = TRANSCRIBE_CASES[case_id]
    from react_review.prompt_contract import sha256_text

    assert sha256_text(case.render(TRANSCRIBE.renderer)) == case.rendered_sha256.upper()


def test_nothing_in_either_split_contract_has_drifted():
    assert LOCATE.drifts() == []
    assert TRANSCRIBE.drifts() == []


def test_the_split_contracts_pin_the_version_strings_the_code_uses():
    assert LOCATE.prompt_version == BATCH_LOCATE_VERSION
    assert TRANSCRIBE.prompt_version == BATCH_TRANSCRIBE_VERSION
    assert LOCATE.extraction_profile == BATCH_SPLIT_PROFILE
    assert TRANSCRIBE.extraction_profile == BATCH_SPLIT_PROFILE
    assert PROMPT_VERSIONS[BATCH_SPLIT_PROFILE]


def test_every_target_shape_is_pinned_on_locate():
    pinned = {case.inputs["target_shape"] for case in LOCATE.cases}
    assert pinned == set(_BODIES)


def test_locate_does_not_ask_for_values_even_on_an_aggregable_study():
    """Numbers belong to transcribe. Locate growing aggregation fields would
    put the value back in the first call."""
    rendered = LOCATE_CASES["study"].render(LOCATE.renderer)
    assert "Do NOT return a value" in rendered
    assert "aggregation_sets" not in rendered
    assert '"value": "verbatim value"' not in rendered


def test_transcribe_warns_that_a_neighbour_is_a_different_reading():
    rendered = TRANSCRIBE_CASES["two_passages"].render(TRANSCRIBE.renderer)
    assert "another block is a different reading" in rendered


def test_a_pinned_locate_case_supplies_every_input_the_renderer_takes():
    parameters = set(inspect.signature(build_batch_locate_prompt).parameters)
    for case in LOCATE.cases:
        assert set(case.inputs) == parameters, case.case_id


def test_a_pinned_transcribe_case_supplies_every_input_the_renderer_takes():
    parameters = set(inspect.signature(build_batch_transcribe_prompt).parameters)
    for case in TRANSCRIBE.cases:
        assert set(case.inputs) == parameters, case.case_id
