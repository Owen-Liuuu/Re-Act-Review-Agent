"""The preflight had no tests, and it produced two false blockers before it did.

Both were the same fault: the probe was not wired the way the run is wired, so
it computed cache keys for a question nobody was going to ask and then reported
the resulting misses as recordings that did not exist. First an empty research
context (0/9 HIT), then a missing cohort registry (6/9). The correct answer is
6/6, and nothing in the suite would have said so.

So what is tested here is not "does it run" but "does it ask the same question
the recording will ask".
"""
from __future__ import annotations

import json
import sys

import pytest

from tests.conftest import requires_frozen_evaluator

from react_review.contracts import repo_root

BENCH = repo_root() / "eval/benchmarks/melanoma_checkpoint_2017"
PLAN = BENCH / "d1_7_expected_plan.json"
PROFILE = "phase8_batch_v6_profile.json"
SOURCE_PDF = BENCH / "raw/sources/larkin_2015.pdf"

sys.path.insert(0, str(repo_root() / "eval"))

needs_paper = pytest.mark.skipif(
    not SOURCE_PDF.is_file(),
    reason="the source paper is copyrighted and not in the repo")


def _observe(cache=None, model_id="glm-4.5-flash", attempts=3):
    import d1_7_preflight as preflight

    return preflight.observe(BENCH, PROFILE, cache, attempts, model_id)


# --- the probe asks the run's question --------------------------------------

@needs_paper
def test_the_preflight_builds_the_collector_the_run_builds():
    """The cohort display name reaches the single-target prompt.

    Built without the registry, the probe asked a different question, missed on
    keys the recording holds, and reported them as live calls the recording did
    not cover. That was published as a blocker.
    """
    import inspect

    import d1_7_preflight as preflight

    source = inspect.getsource(preflight.observe)
    assert "benchmark_cohorts" in source
    assert "benchmark_reviews" in source
    # And the context, which reaches the prompt too.
    assert 'manifest.get("domain")' in source


@needs_paper
def test_the_arm_identity_route_is_fully_covered_by_the_phase7_recording():
    """The measured fact the pre-registration turns on.

    If this ever fails, a recording run would go live for the arm identities as
    well as the batch route, and could not be described as isolating batching.
    """
    requires_frozen_evaluator()
    cache = (repo_root()
             / "output/baselines/melanoma_checkpoint_2017/phase7_extraction_cache.json")
    if not cache.is_file():
        pytest.skip("the phase7 recording is not in this checkout")

    observed = _observe(cache)
    uncovered = [t for t in observed["v4_trace"] if not t["hit"]]
    assert observed["v4_trace"], "no arm-identity lookup happened at all"
    assert uncovered == [], (
        f"{len(uncovered)} arm-identity lookup(s) have no recording: "
        f"{[(t['group'], t['attempt']) for t in uncovered]}")


@needs_paper
def test_every_batch_slot_is_checked_not_only_the_first_attempt():
    """A failed first attempt reaches for attempt 1, and a stale recording
    there would be replayed into a run that believes it asked."""
    requires_frozen_evaluator()
    observed = _observe()
    assert observed["batches"]
    for batch in observed["batches"]:
        assert len(batch["cache_key_slots"]) == 3
        assert len(set(batch["cache_key_slots"])) == 3, (
            "the attempt must be inside the key, or all three slots are one")


@needs_paper
def test_the_model_id_comes_from_the_pin_not_from_whichever_cache_is_open():
    """The tools take the model id from the backend first and the open cache
    second, so a preflight against an EMPTY cache with no backend computes keys
    under the literal string "replay" — addresses no recording will occupy."""
    requires_frozen_evaluator()
    one = _observe(model_id="glm-4.5-flash")
    other = _observe(model_id="some-other-model")
    assert (one["batches"][0]["cache_key_slots"]
            != other["batches"][0]["cache_key_slots"])


# --- the plan is compared, not merely written -------------------------------

@needs_paper
def test_the_checked_in_plan_still_asks_exactly_what_it_pre_registered():
    """The first pre-registration published identities computed under a bug and
    never regenerated. Nothing compared them to anything.

    The plan describes a recording that HAPPENED, under contract
    phase8_batch_v2, so it is not regenerated when the evaluator moves. What
    must not move is what the run ASKS: the evaluator decides how an answer is
    judged, not what question reaches the model, and if that ever stops being
    true the recording stops being replayable.
    """
    requires_frozen_evaluator()
    import d1_7_preflight as preflight

    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    drifts = preflight.compare(plan, _observe(), settings={})
    identity_drifts = [d for d in drifts if "pre-registered contract" not in d]
    assert identity_drifts == [], identity_drifts
    # The one expected difference, named rather than filtered silently.
    assert drifts == ["the plan pre-registered contract 'phase8_batch_v2' and "
                      "this is 'phase8_batch_v5'"]


@needs_paper
def test_a_changed_batch_identity_is_reported_as_drift():
    requires_frozen_evaluator()
    import d1_7_preflight as preflight

    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    plan["expected_batches"][0]["question_id"] = "0" * 32
    drifts = preflight.compare(plan, _observe(), settings={})
    assert any("question_id differs" in d for d in drifts)


@needs_paper
def test_a_batch_the_plan_does_not_expect_is_reported():
    requires_frozen_evaluator()
    import d1_7_preflight as preflight

    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    plan["expected_batches"] = plan["expected_batches"][:-1]
    drifts = preflight.compare(plan, _observe(), settings={})
    assert any("does not expect" in d for d in drifts)


def test_changed_model_settings_are_reported_as_drift():
    import d1_7_preflight as preflight

    plan = {"model_settings_sha256": preflight.settings_digest({"temperature": 0.1})}
    observed = {"contract": type("C", (), {"profile_id": ""})(),
                "research_context": "", "model_id": "", "batches": []}
    drifts = preflight.compare(plan, observed, {"temperature": 0.9})
    assert any("model settings changed" in d for d in drifts)


# --- what must never be read, and what must never be written ----------------

def test_the_preflight_reads_no_secret_from_the_local_config(tmp_path):
    """`config.local.yaml` holds the api key. What has to be frozen is how the
    model was configured, not who was allowed to call it."""
    import d1_7_preflight as preflight

    config = tmp_path / "config.yaml"
    config.write_text(
        "llm:\n  provider: glm\n  model: m\n  temperature: 0.1\n"
        "  max_tokens: 8192\n  api_key: SECRET-DO-NOT-READ\n", encoding="utf-8")

    settings = preflight.model_settings(config)
    assert "SECRET-DO-NOT-READ" not in json.dumps(settings)
    assert not any("key" in name for name in settings)
    assert settings["temperature"] == 0.1 and settings["max_tokens"] == 8192


def test_a_probe_cache_refuses_to_record_anything():
    import d1_7_preflight as preflight

    probe = preflight.ProbeCache(None, {"readings": []})
    with pytest.raises(AssertionError, match="records nothing"):
        probe.put("k", {})
    with pytest.raises(AssertionError, match="writes nothing"):
        probe.save()


def test_a_pinned_model_cannot_be_asked_anything():
    import asyncio

    import d1_7_preflight as preflight

    model = preflight.PinnedModel("glm-4.5-flash")
    assert model.model_id == "glm-4.5-flash"
    with pytest.raises(AssertionError, match="never asks the model"):
        asyncio.run(model.complete("anything"))


# --- the gate is frozen before the result exists ----------------------------

def test_the_feature_gate_names_its_baseline_row_by_row():
    """"Graded against the D1 feature gate" named a gate that did not exist,
    which is defining the standard after seeing the result with extra steps."""
    gate = json.loads((repo_root() / "configs/gates/d1_batch_v1.json"
                       ).read_text(encoding="utf-8"))
    assert len(gate["baseline_rows"]) == 15
    assert {r["baseline_state"] for r in gate["baseline_rows"]} <= {
        "correct", "refused", "wrong_released"}
    assert gate["hard_conditions"]["silent_releases"] == 0
    assert "label_accuracy" in " ".join(gate["reported_never_gated"])
    assert "correct -> wrong_released" in gate["transitions"]["forbidden"]
