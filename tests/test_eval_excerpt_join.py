"""Joining a run's readings to the key — the layer that had no tests at all.

`assess` is where a run's batches meet the answer key, and it was written,
wired into the harness and reported on without a single test touching it. The
first review of it found three ways to be silently wrong: a join key that
collides, a missing document that produces all-zero counts, and a declared
document hash nobody compares. All three read as "no window problem".
"""
from __future__ import annotations

import pytest

from react_review.eval_excerpt import GoldError, assess, index_gold
from react_review.schemas.batch import BatchReadingRecord, ExcerptProvenance

PAPER = ("A total of 945 patients underwent randomization: 316 to the nivolumab "
         "group and 314 to the combination group. Later, the discussion repeats "
         "that 316 to the nivolumab group were treated.")
SHA = "DOC-1"


def _gold(**over):
    body = {
        "gold_id": "excerpt_gold_test", "document_sha256": SHA,
        "batches": [{
            "batch_id": "larkin/cohort_n/arm", "study_id": "larkin_2015",
            "field_type": "cohort_n", "target_shape": "arm",
            "audit_ids": ["MA004", "MA008"],
            "witnesses": [
                {"witness_id": "w1", "witness_type": "explicit",
                 "source_quote": "316 to the nivolumab group", "modality": "text"},
                {"witness_id": "w2", "witness_type": "explicit",
                 "source_quote": "314 to the combination group", "modality": "text"},
            ]}],
    }
    body.update(over)
    return body


def _reading(*, claim_ids=("MA008", "MA004"), spans=((0, len(PAPER)),),
             windowed=True, study="larkin_2015", field="cohort_n", shape="arm",
             provenance=True):
    return BatchReadingRecord(
        execution_id="E1", study_id=study, field_type=field, target_shape=shape,
        claim_ids=list(claim_ids),
        excerpt_provenance=(ExcerptProvenance(
            windowed=windowed, source_chars=len(PAPER),
            excerpt_chars=sum(b - a for a, b in spans), spans=list(spans),
            selection_method_id="m", selection_version="v2")
            if provenance else None))


def _assess(readings, gold=None, *, text=PAPER, sha=SHA):
    return assess(readings, gold or _gold(), lambda _: text,
                  sha_for=lambda _: sha)


# --- the join ----------------------------------------------------------------

def test_a_reading_is_matched_to_the_key_by_the_claims_it_answered():
    """Order does not matter; the claim SET is the identity."""
    report = _assess([_reading()])
    assert report.assessable
    assert report.tally.gold_covered_batches == 1
    assert report.batches[0]["claim_ids"] == ["MA004", "MA008"]


def test_two_gold_batches_claiming_the_same_reading_are_refused():
    """One set of claims has one reading. Two keys for it means one silently
    judges the other's witnesses, and the numbers still look plausible."""
    gold = _gold()
    duplicate = dict(gold["batches"][0])
    duplicate["batch_id"] = "a second key for the same claims"
    gold["batches"] = [gold["batches"][0], duplicate]
    with pytest.raises(GoldError, match="both claim"):
        index_gold(gold)


def test_a_gold_batch_that_names_no_claims_is_refused():
    gold = _gold()
    gold["batches"][0]["audit_ids"] = []
    with pytest.raises(GoldError, match="names no audit_ids"):
        index_gold(gold)


def test_the_same_field_at_two_timepoints_is_two_keys_not_one():
    """The join used to be (study, field, shape), which is not an identity.

    One study reports one field at several timepoints, and each is a separate
    reading with a separate window. Under the old key they collided and the
    second was judged against the first's witnesses.
    """
    gold = _gold()
    second = {**gold["batches"][0], "batch_id": "larkin/cohort_n/arm@12m",
              "audit_ids": ["MA020"],
              "witnesses": [{"witness_id": "w3", "witness_type": "explicit",
                             "source_quote": "not printed in this paper",
                             "modality": "text"}]}
    gold["batches"].append(second)

    report = assess([_reading(), _reading(claim_ids=("MA020",))],
                    gold, lambda _: PAPER, sha_for=lambda _: SHA)
    assert report.assessable
    outcomes = {b["batch_id"]: {w["outcome"] for w in b["witnesses"]}
                for b in report.batches}
    assert outcomes["larkin/cohort_n/arm"] == {"covered"}
    # Judged against its OWN witness, which this paper does not contain.
    assert outcomes["larkin/cohort_n/arm@12m"] == {"fulltext_unlocatable"}


def test_a_key_that_describes_a_different_reading_is_refused():
    """Matching claims but a different field means the key is not about this."""
    report = _assess([_reading(field="event_count")])
    assert not report.assessable
    assert "different reading" in report.reason


def test_readings_the_key_says_nothing_about_are_reported_not_dropped():
    """A key silently covering half a run gives a figure over a sample nobody
    chose."""
    extra = _reading(claim_ids=("MA099",))
    extra = extra.model_copy(update={"execution_id": "E-unjudged"})
    report = _assess([_reading(), extra])
    assert report.assessable
    assert report.unjudged_run_batches == ("E-unjudged",)


# --- refusals, rather than zeros ---------------------------------------------

def test_a_run_with_no_extracted_text_is_not_assessable():
    """All-zero counts read as "no window problem" — the most dangerous thing
    this could say when nothing was judged at all."""
    report = assess([_reading()], _gold(), lambda _: None, sha_for=lambda _: SHA)
    assert not report.assessable
    assert "no extracted text" in report.reason
    assert report.tally is None
    assert report.as_dict()["assessable"] is False


def test_a_document_that_is_not_the_one_the_key_measured_is_refused():
    """Offsets mean nothing against a different extraction of the same PDF."""
    report = _assess([_reading()], sha="A-DIFFERENT-EXTRACTION")
    assert not report.assessable
    assert "would judge one document" in report.reason


def test_a_run_whose_batches_are_all_unknown_to_the_key_is_not_assessable():
    report = _assess([_reading(claim_ids=("MA099",))])
    assert not report.assessable
    assert "no batch this run made" in report.reason


def test_a_reading_that_recorded_nothing_about_its_window_is_refused():
    report = _assess([_reading(provenance=False)])
    assert not report.assessable
    assert "no excerpt provenance" in report.reason


# --- what the window actually held -------------------------------------------

def test_a_witness_outside_every_window_is_a_window_miss():
    report = _assess([_reading(spans=((0, 40),))])
    assert report.assessable
    assert report.tally.gold_missing_batches == 1
    assert report.tally.gold_covered_batches == 0


def test_a_quote_that_occurs_twice_counts_as_covered_if_either_place_was_sent():
    """Papers repeat themselves. A window that kept the discussion but not the
    results still showed the evidence, and blaming the selector for the copy it
    did not send would be blaming it for a passage it did include.
    """
    second = PAPER.rindex("316 to the nivolumab group")
    gold = _gold()
    gold["batches"][0]["witnesses"] = [gold["batches"][0]["witnesses"][0]]

    report = assess([_reading(spans=((second, len(PAPER)),))], gold,
                    lambda _: PAPER, sha_for=lambda _: SHA)
    assert report.assessable
    assert report.tally.gold_covered_batches == 1


def test_the_report_names_the_selector_that_chose_the_window():
    report = _assess([_reading()])
    entry = report.batches[0]
    assert entry["selection_version"] == "v2"
    assert entry["locator_version"]


# --- the selector reports what it SENT, not what its markers declare ---------

def test_a_truncated_block_is_not_reported_as_fully_sent():
    """The marker keeps naming the block it came from — those bytes are in the
    prompt — but the text beneath it is cut to the remaining room.

    Reading the spans back off the markers therefore over-reported: a
    20,000-character excerpt declared 21,000 characters of source, and a witness
    in the part that was cut off would have been called covered.
    """
    from react_review.tools.extract_source import _MAX_TEXT, select_excerpt

    text = ("A" * 500) + " ".join(f"sample size total {i}" for i in range(6000))
    excerpt, spans = select_excerpt(text, target="sample size",
                                    raw_label="Total, n", field_type="sample_size")
    assert len(excerpt) == _MAX_TEXT

    markers = sum(len(f"\n\n[SOURCE EXCERPT {a}:{b}]\n") for a, b in spans)
    assert sum(b - a for a, b in spans) + markers == len(excerpt), (
        "the spans claim more source than the excerpt can hold")
    for start, end in spans:
        assert text[start:end] in excerpt


def test_the_prompt_text_is_unchanged_by_reporting_what_was_sent():
    """v1 and v2 differ in what is REPORTED. A prompt that changed would
    invalidate every recording ever made under targeted_v5_batch.
    """
    from react_review.tools.extract_source import _paper_excerpt, select_excerpt

    text = ("A" * 500) + " ".join(f"sample size total {i}" for i in range(6000))
    kwargs = dict(target="sample size", raw_label="Total, n",
                  field_type="sample_size")
    assert select_excerpt(text, **kwargs)[0] == _paper_excerpt(text, **kwargs)


# --- the key describes the batches the run actually makes -------------------

BENCH = __import__("react_review.contracts", fromlist=["repo_root"]).repo_root() / \
    "eval/benchmarks/melanoma_checkpoint_2017"


def _gold_v2():
    import json

    return json.loads((BENCH / "excerpt_gold_v2.json").read_text(encoding="utf-8"))


def test_the_key_names_every_batch_the_real_grouping_produces():
    """v1 asserted one reading for three arm counts. The run makes three.

    They come from three different review columns, and the group key includes
    the raw field name — so `larkin_2015/cohort_n/arm` was a reading nobody
    makes. It stayed invisible while coverage joined on (study, field, shape),
    which collapsed all three into it.
    """
    import csv

    from react_review.dkb import load_runtime_knowledge
    from react_review.eval_excerpt import (
        benchmark_reviews,
        dry_run_collector,
        planned_batches,
    )
    from react_review.eval_profile import load_profile

    root = BENCH.parents[2]
    rows = list(csv.DictReader(
        (BENCH / "audit_template.csv").open(encoding="utf-8-sig")))
    profile = load_profile(BENCH, "phase8_batch_v3_profile.json",
                           answer_key_ids=[r["audit_id"] for r in rows])
    collector = dry_run_collector(profile.run_contract, load_runtime_knowledge(
        root / "configs/knowledge.seed.json", root / "configs/ontology"))

    groups = planned_batches(collector,
                             benchmark_reviews(rows, profile.targets),
                             profile.run_contract.extraction_routes["value"])
    produced = {tuple(sorted(c.review_data_id for c in g.claims)) for g in groups}
    judged = {tuple(sorted(b["audit_ids"])) for b in _gold_v2()["batches"]}
    assert produced == judged, (
        f"the key judges batches the run does not make: {judged - produced}; "
        f"and misses ones it does: {produced - judged}")


def test_the_key_is_bound_to_the_document_its_offsets_came_from():
    gold = _gold_v2()
    assert len(gold["document_sha256"]) == 64
    assert gold["text_extraction_library"], (
        "a PDF extractor's version changes its output; record which one wrote "
        "these offsets")


# --- the harness glue -------------------------------------------------------

def test_the_harness_reads_the_same_document_the_key_measured(tmp_path):
    """`_excerpt_coverage` had no test either, and it is where the run's
    readings, the benchmark's PDFs and the key are brought together.

    The failure it must not have is reading the paper with a different
    extraction than the one the offsets came from — which would leave every
    number plausible and every witness in the wrong place.
    """
    import types

    from react_review.eval_excerpt import coverage_for_run

    results = types.SimpleNamespace(batch_readings=[
        BatchReadingRecord(execution_id="E1", study_id="larkin_2015",
                           field_type="hazard_ratio", target_shape="comparison",
                           claim_ids=["MA012", "MA013", "MA014", "MA015"],
                           excerpt_provenance=ExcerptProvenance(
                               windowed=False, source_chars=1, excerpt_chars=1,
                               spans=[(0, 10**9)], selection_method_id="m",
                               selection_version="v2"))])
    studies = [types.SimpleNamespace(study_id="larkin_2015",
                                     source_pdf="raw/sources/larkin_2015.pdf")]

    report = coverage_for_run(results, studies, BENCH,
                              BENCH / "excerpt_gold_v2.json")
    if not (BENCH / "raw/sources/larkin_2015.pdf").is_file():
        # The source paper is copyrighted and not in the repo. The glue must
        # then REFUSE rather than report zeros.
        assert report is not None and not report.assessable
        assert "no extracted text" in report.reason
        return
    assert report.assessable
    assert report.tally.gold_covered_batches == 1
    assert report.unjudged_run_batches == ()


def test_the_harness_is_silent_when_a_benchmark_publishes_no_key():
    """Silent, not zero: a benchmark without a key has not judged its windows."""
    import types

    from react_review.eval_excerpt import coverage_for_run

    results = types.SimpleNamespace(batch_readings=[_reading()])
    assert coverage_for_run(results, [], BENCH, None) is None
