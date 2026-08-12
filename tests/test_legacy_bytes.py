"""A legacy artifact gains nothing when the batch path grows a field.

Every phase of this work is verified by replaying a frozen benchmark and
comparing the result to what it was. That comparison is worth what it costs only
if it compares BYTES — and a Pydantic model with a new optional field writes
`"batch_provenance": null` into every row that never had one, which is a changed
artifact for a fact nothing recorded.

The temptation is `exclude_none=True`. It is wrong here: `provider_input_tokens`
is null on purpose and has always been written, because "the provider did not
report" is a different statement from zero and must not be silently dropped. So
the omission is targeted — these fields, and only when unset.

Four places serialise, and each is pinned separately, because fixing one and
assuming the others follow is how three of them stayed broken last time.
"""
from __future__ import annotations

import json

from react_review.schemas.batch import BatchQuestionId
from react_review.schemas.evidence import (
    AggregationProvenance,
    BatchProjectionProvenance,
    SourceEvidenceItem,
)
from react_review.schemas.telemetry import RunTelemetry

BATCH_KEYS = {"batch_provenance", "aggregation_provenance"}


def _legacy_item() -> SourceEvidenceItem:
    return SourceEvidenceItem(study_id="larkin", group="a", field_type="cohort_n",
                              source_value="316", source_quote="a quote")


# --- 1. the evidence row itself --------------------------------------------

def test_a_legacy_row_carries_no_batch_keys_at_all():
    body = _legacy_item().model_dump(mode="json")
    assert not (BATCH_KEYS & set(body)), sorted(BATCH_KEYS & set(body))


def test_a_batch_row_carries_exactly_the_provenance_it_has():
    item = _legacy_item().model_copy(update={
        "batch_provenance": BatchProjectionProvenance(
            claim_id="c1", batch_execution_id="E1", route="targeted_v5_batch")})
    body = item.model_dump(mode="json")
    assert body["batch_provenance"]["claim_id"] == "c1"
    # The one it does NOT have stays absent rather than arriving as null.
    assert "aggregation_provenance" not in body


def test_both_appear_when_both_are_set():
    item = _legacy_item().model_copy(update={
        "batch_provenance": BatchProjectionProvenance(claim_id="c1"),
        "aggregation_provenance": AggregationProvenance(policy_id="safe_sum_v5")})
    body = item.model_dump(mode="json")
    assert BATCH_KEYS <= set(body)


def test_a_meaningful_null_is_not_dropped_with_them():
    """`exclude_none` would take these too, and they are already recorded."""
    body = _legacy_item().model_dump(mode="json")
    assert "source_components" in body and body["source_components"] is None
    assert "population_scope" in body and body["population_scope"] is None


# --- 2. AgentRun.final, which is a blind dump of the row --------------------

def test_the_agent_run_final_of_a_legacy_row_gains_no_key():
    """`collector._result` writes `source_item.model_dump(mode="json")`."""
    final = _legacy_item().model_dump(mode="json")
    assert not (BATCH_KEYS & set(final))
    assert json.dumps(final)          # and it is serialisable as it stands


# --- 3. the evidence package, which dumps the whole tree -------------------

def test_a_legacy_package_gains_no_key_anywhere():
    from react_review.schemas.package import EvidencePackage

    package = EvidencePackage(run_id="r1", source_items=[_legacy_item()])
    text = json.dumps(package.model_dump(mode="json"))
    for key in BATCH_KEYS:
        assert f'"{key}"' not in text, key


# --- 4. telemetry, which is dumped whole into every accuracy report --------

def test_legacy_telemetry_has_no_batch_section():
    """`run_full_accuracy` writes `telemetry.model_dump(mode="json")`."""
    body = RunTelemetry().model_dump(mode="json")
    assert "batch" not in body
    # The provider-token nulls are still there, still meaning "not reported".
    assert body["provider_input_tokens"] is None
    assert body["provider_output_tokens"] is None


def test_the_legacy_telemetry_field_set_is_exactly_what_it_was():
    """A new top-level counter would change every recorded run's bytes."""
    assert set(RunTelemetry().model_dump(mode="json")) == {
        "tool_attempts", "backend_requests", "repeated_attempts",
        "backend_failures", "prompt_chars", "output_chars",
        "provider_input_tokens", "provider_output_tokens", "call_seconds",
        "wall_seconds", "cache_hits", "cache_misses",
    }


# --- the identities are serialisable, since provenance carries them --------

def test_a_question_id_round_trips_as_json():
    question = BatchQuestionId(study_id="larkin", field_type="cohort_n")
    assert json.loads(json.dumps(question.model_dump(mode="json")))["study_id"] \
        == "larkin"
