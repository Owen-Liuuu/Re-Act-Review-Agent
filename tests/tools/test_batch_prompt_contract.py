"""The batch prompt cannot change without somebody deciding that it should.

`legacy_v3` has had a golden SHA since Phase 7. The batch contract was written,
routed to by a frozen benchmark profile, and left unpinned — and the benchmark
profile does not cover it: that pins the run profile, which names route STRINGS,
not the words those routes send. A character changed in the template silently
invalidates every recording made under it, and the symptom is a replay miss,
which reads as a missing recording rather than an edited prompt.

What is pinned is the RENDERED prompt. Comments, renames and extracted helpers
do not change what the model is asked and must not fail these tests. A different
question must.
"""
from __future__ import annotations

import inspect

import pytest

from react_review.contracts import ContractError
from react_review.prompt_contract import BATCH_V5, PromptContract
from react_review.tools.batch_prompt import _BODIES, build_batch_prompt
from react_review.tools.extraction_profile import BATCH_PROFILE_NAME, PROMPT_VERSIONS

CONTRACT = PromptContract.load(BATCH_V5)
CASES = {case.case_id: case for case in CONTRACT.cases}


# --- the pins ---------------------------------------------------------------

@pytest.mark.parametrize("case_id", sorted(CASES))
def test_the_branch_still_asks_what_it_was_published_asking(case_id):
    """Named per branch, so a failure says WHICH question moved.

    "A hash changed" sends the reader to diff a whole template. "The STUDY
    branch with aggregation changed" is the thing they have to decide about.
    """
    case = CASES[case_id]
    rendered = case.render(CONTRACT.renderer)
    from react_review.prompt_contract import sha256_text

    assert sha256_text(rendered) == case.rendered_sha256.upper(), (
        f"{case_id} renders differently than published. {case.why}. If the "
        "change is intended, this is a NEW PROMPT VERSION, not a new hash "
        "typed into the contract file")


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_the_recording_key_the_branch_writes_under_is_unchanged(case_id):
    """The pin that decides whether recordings survive.

    The prompt's SHA enters the cache key AND the BatchQuestionId, so this is
    the same fact seen from the other end: a key that moved means every
    recording made under the old one is now a miss.
    """
    case = CASES[case_id]
    key = CONTRACT.key_for(case.render(CONTRACT.renderer))
    assert key.upper() == case.cache_key.upper()


def test_nothing_in_the_contract_has_drifted():
    """The whole file at once — the check the D1-7 preflight will run."""
    assert CONTRACT.drifts() == []


def test_the_contract_pins_the_version_string_the_code_actually_uses():
    """Otherwise the recordings are filed under a version nothing names."""
    assert CONTRACT.prompt_version == PROMPT_VERSIONS[BATCH_PROFILE_NAME]
    assert CONTRACT.extraction_profile == BATCH_PROFILE_NAME


# --- the pins cover the branches that exist ---------------------------------

def test_every_target_shape_is_pinned():
    """A shape added without a case would ship unpinned and look covered."""
    pinned = {case.inputs["target_shape"] for case in CONTRACT.cases}
    assert pinned == set(_BODIES), (
        f"shapes with no pinned case: {sorted(set(_BODIES) - pinned)}")


def test_both_conditional_blocks_are_pinned_on_and_off():
    """Each is a paragraph the prompt grows, and the assembly is what is sent."""
    timepoints = {bool(case.inputs["timepoint_label"]) for case in CONTRACT.cases}
    assert timepoints == {True, False}

    from react_review.tools.batch_prompt import aggregation_applies

    aggregation = {aggregation_applies(case.inputs["target_shape"],
                                       case.inputs["field_type"])
                   for case in CONTRACT.cases}
    assert aggregation == {True, False}


def test_a_pinned_case_supplies_every_input_the_renderer_takes():
    """A new parameter with a default would ship unpinned.

    The pinned cases would keep rendering identically — they do not pass it —
    so every hash here would stay green while production sent something these
    cases never exercise.
    """
    parameters = set(inspect.signature(build_batch_prompt).parameters)
    for case in CONTRACT.cases:
        assert set(case.inputs) == parameters, (
            f"{case.case_id} pins {sorted(set(case.inputs) ^ parameters)} "
            "differently from what build_batch_prompt accepts")


# --- the loader refuses what it cannot verify -------------------------------

def test_a_contract_naming_no_renderer_is_refused(tmp_path):
    """A loader that always called the batch builder would happily "verify" a
    contract for some other prompt."""
    path = tmp_path / "unknown.json"
    path.write_text('{"contract_id": "some_other_prompt", "cases": [{}]}',
                    encoding="utf-8")
    with pytest.raises(ContractError, match="names no renderer"):
        PromptContract.load(path)


def test_a_contract_that_pins_nothing_is_refused(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text('{"contract_id": "batch_v5", "cases": []}', encoding="utf-8")
    with pytest.raises(ContractError, match="pins no cases"):
        PromptContract.load(path)


def test_a_changed_prompt_is_caught_in_every_branch_it_reaches(monkeypatch):
    """One space in the shared rules block, and all six branches say so."""
    from react_review.tools import batch_prompt

    monkeypatch.setattr(batch_prompt, "_RULES", batch_prompt._RULES + " ")
    drifted = CONTRACT.drifts()
    named = {line.split(":")[0] for line in drifted}
    assert named == set(CASES)
    assert any("replay miss" in line for line in drifted)
