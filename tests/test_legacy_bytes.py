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


# --- 5. the accuracy report's rows -----------------------------------------

def _row(**kw):
    from react_review.eval_accuracy import RowResult

    body = dict(study_id="larkin", group="a", field_type="cohort_n",
                expected_label="match", predicted_label="match",
                expected_source="316", extracted_source="316", found=True,
                outcome="found", extraction_correct=True)
    body.update(kw)
    return RowResult(**body)



def test_a_single_target_row_carries_no_batch_fields():
    """`asdict` would put all three into every legacy row.

    This was written as a comment claiming the fields were harmless, and the
    replay comparison disagreed within the minute — which is the argument for
    comparing bytes rather than reasoning about them.
    """
    from react_review.eval_accuracy import BATCH_ROW_FIELDS, row_payload

    body = row_payload(_row())
    assert not (set(BATCH_ROW_FIELDS) & set(body))


def test_a_batched_row_carries_them():
    from react_review.eval_accuracy import row_payload

    body = row_payload(_row(batch_execution_id="E1",
                            batch_route="targeted_v5_batch",
                            projection_status="ok"))
    assert body["batch_execution_id"] == "E1"
    assert body["batch_route"] == "targeted_v5_batch"


def test_one_populated_field_brings_the_others_with_it():
    """Partial omission would make a row's shape depend on its content."""
    from react_review.eval_accuracy import BATCH_ROW_FIELDS, row_payload

    body = row_payload(_row(projection_status="scope_unresolved"))
    assert set(BATCH_ROW_FIELDS) <= set(body)


# --- 6. the evidence package's batch readings ------------------------------

def test_a_package_with_no_readings_has_no_batch_records_key():
    from react_review.schemas.package import EvidencePackage

    body = EvidencePackage(run_id="r1").model_dump(mode="json")
    assert "batch_records" not in body


def test_a_package_with_readings_carries_them():
    from react_review.schemas.batch import BatchReadingRecord
    from react_review.schemas.package import EvidencePackage

    package = EvidencePackage(run_id="r1", batch_records=[
        BatchReadingRecord(execution_id="E1", claim_ids=["c1", "c2"])])
    body = package.model_dump(mode="json")
    assert body["batch_records"][0]["execution_id"] == "E1"


def test_every_claim_reference_resolves_to_a_recorded_reading():
    """A reference pointing at nothing is worse than no reference."""
    from react_review.schemas.batch import BatchReadingRecord
    from react_review.schemas.package import EvidencePackage

    item = _legacy_item().model_copy(update={
        "batch_provenance": BatchProjectionProvenance(
            claim_id="c1", batch_execution_id="E1")})
    package = EvidencePackage(
        run_id="r1", source_items=[item],
        batch_records=[BatchReadingRecord(execution_id="E1", claim_ids=["c1"])])
    known = {r.execution_id for r in package.batch_records}
    for row in package.source_items:
        provenance = row.batch_provenance
        if provenance and provenance.batch_execution_id:
            assert provenance.batch_execution_id in known


# --- 7. per-stage telemetry ------------------------------------------------

def test_a_run_that_used_no_stage_has_no_stages_key():
    assert "stages" not in RunTelemetry().model_dump(mode="json")


def test_a_stage_bucket_appears_only_for_the_stage_that_was_used():
    from react_review.schemas.telemetry import BATCH_EXTRACTION

    telemetry = RunTelemetry()
    telemetry.record_call(prompt="p", output="o", seconds=0.5,
                          stage=BATCH_EXTRACTION)
    body = telemetry.model_dump(mode="json")
    assert set(body["stages"]) == {BATCH_EXTRACTION}


def test_a_stage_call_still_counts_once_globally():
    """The globals were always updated exactly once per call, and still are."""
    from react_review.schemas.telemetry import SEMANTIC

    telemetry = RunTelemetry()
    telemetry.record_call(prompt="pp", output="ooo", seconds=1.0, stage=SEMANTIC)
    assert telemetry.backend_requests == 1
    assert telemetry.prompt_chars == 2 and telemetry.output_chars == 3
    assert telemetry.stages[SEMANTIC].requests == 1


def test_stage_cache_counting_never_touches_the_global_totals():
    """The harness folds each cache's own totals in when a run ends.

    Adding to them here would double every cache number it reports.
    """
    from react_review.schemas.telemetry import BATCH_EXTRACTION

    telemetry = RunTelemetry()
    telemetry.record_stage_cache(BATCH_EXTRACTION, hits=3, misses=1)
    assert telemetry.cache_hits == 0 and telemetry.cache_misses == 0
    assert telemetry.stages[BATCH_EXTRACTION].cache_hits == 3


def test_an_unknown_stage_is_refused():
    """A typo would become a bucket nobody reads, not a visible cost."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="unknown telemetry stage"):
        RunTelemetry().record_stage_cache("extractoin", hits=1)


# --- 8. batch statistics ---------------------------------------------------

def test_a_run_that_never_batched_has_no_batch_section():
    assert "batch" not in RunTelemetry().model_dump(mode="json")


def test_batch_statistics_count_what_batching_actually_did():
    telemetry = RunTelemetry()
    telemetry.record_batch(claims=3, failed=False)
    telemetry.record_batch(claims=1, failed=False)
    telemetry.record_batch(claims=2, failed=True)
    stats = telemetry.batch
    assert stats.batches == 3 and stats.claims == 6
    assert stats.singleton_batches == 1 and stats.failed_batches == 1
    assert round(stats.claims_per_batch, 2) == 2.0


def test_projection_outcomes_are_counted_by_name():
    """"Unresolved" as one number cannot say which way a run is failing."""
    telemetry = RunTelemetry()
    telemetry.record_projection("scope_unresolved", "rejected")
    telemetry.record_projection("scope_unresolved", "rejected")
    telemetry.record_projection("not_reported", "not_applicable")
    stats = telemetry.batch
    assert stats.projections == {"scope_unresolved": 2, "not_reported": 1}
    assert stats.aggregation_rejected == 2
    # `not_applicable` is not an attempt: nothing was offered to add up.
    assert stats.aggregation_attempts == 2


def test_a_printed_total_beside_a_valid_sum_is_corroboration():
    telemetry = RunTelemetry()
    telemetry.record_projection("ok", "derived")
    assert telemetry.batch.explicit_vs_derived_agreements == 1
    assert telemetry.batch.explicit_vs_derived_conflicts == 0


def test_a_contradiction_beside_a_valid_sum_is_a_conflict():
    """Agreement and conflict must never be summed into one 'checked' figure."""
    telemetry = RunTelemetry()
    telemetry.record_projection("contradictory", "derived")
    assert telemetry.batch.explicit_vs_derived_conflicts == 1
    assert telemetry.batch.explicit_vs_derived_agreements == 0


# --- 9. an EMPTY telemetry object, which is not the same as none ----------

def test_a_package_given_a_telemetry_that_measured_nothing_omits_it():
    """The case the previous test missed by passing None.

    A zero RunTelemetry serialises to a full dictionary of zeroes and is
    therefore truthy, so `if not body.get("telemetry")` decided nothing and
    every package gained the key. The claim that an unmeasured run carried no
    telemetry was simply false, and the test that was supposed to prove it never
    constructed the object it was about.
    """
    from react_review.schemas.package import EvidencePackage

    package = EvidencePackage(run_id="r1", telemetry=RunTelemetry())
    assert "telemetry" not in package.model_dump(mode="json")


def test_one_measurement_of_any_kind_brings_it_back():
    from react_review.schemas.package import EvidencePackage

    for mutate in (lambda t: t.attempt("x"),
                   lambda t: t.record_cache(hits=1, misses=0),
                   lambda t: t.record_batch(claims=1, failed=False),
                   lambda t: setattr(t, "wall_seconds", 0.1)):
        telemetry = RunTelemetry()
        mutate(telemetry)
        assert telemetry.has_measurements()
        body = EvidencePackage(run_id="r1", telemetry=telemetry).model_dump(mode="json")
        assert "telemetry" in body


def test_claims_per_batch_cannot_be_supplied():
    """A settable field would let 999 sit beside batches=2 and claims=4."""
    import pytest as _pytest

    from react_review.schemas.telemetry import BatchStats

    with _pytest.raises(Exception):
        BatchStats(batches=2, claims=4, claims_per_batch=999)
    assert BatchStats(batches=2, claims=4).model_dump()["claims_per_batch"] == 2.0
