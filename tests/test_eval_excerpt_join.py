"""Joining a run's readings to the key — the layer that had no tests at all.

`assess` is where a run's batches meet the answer key, and it was written,
wired into the harness and reported on without a single test touching it. The
first review of it found three ways to be silently wrong: a join key that
collides, a missing document that produces all-zero counts, and a declared
document hash nobody compares. All three read as "no window problem".
"""
from __future__ import annotations

import pytest

from tests.conftest import requires_frozen_evaluator

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


def test_a_reading_the_key_says_nothing_about_makes_the_result_unassessable():
    """A key covering half a run gives a figure over a sample nobody chose.

    Reporting it as a note beside a green number was not enough: the number was
    still printed, still true of the batches it counted, and still silent about
    the rest.
    """
    extra = _reading(claim_ids=("MA099",))
    extra = extra.model_copy(update={"execution_id": "E-unjudged"})
    report = _assess([_reading(), extra])
    assert not report.assessable
    assert report.unjudged_run_batches == ("E-unjudged",)
    assert "different sets of readings" in report.reason


def test_a_batch_the_key_expects_and_the_run_never_made_is_not_assessable():
    """"6 of 6 covered" is true of what a run did and silent about what it did
    not. Both directions are checked, or the denominator is whatever happened."""
    gold = _gold()
    gold["batches"].append({
        "batch_id": "a batch this run never made", "study_id": "larkin_2015",
        "field_type": "event_count", "target_shape": "arm",
        "audit_ids": ["MA050"],
        "witnesses": [{"witness_id": "w9", "witness_type": "explicit",
                       "source_quote": "316 to the nivolumab group",
                       "modality": "text"}]})
    report = _assess([_reading()], gold)
    assert not report.assessable
    assert report.missing_from_run == ("a batch this run never made",)


def test_a_run_that_produced_no_reading_at_all_is_not_assessable(tmp_path):
    """Silence removed the coverage section from exactly the reports that most
    need one — the runs where everything failed."""
    import json
    import types

    from react_review.eval_excerpt import coverage_for_run

    path = tmp_path / "gold.json"
    path.write_text(json.dumps(_gold()), encoding="utf-8")
    report = coverage_for_run(types.SimpleNamespace(batch_readings=[]), [],
                              tmp_path, path)
    assert report is not None and not report.assessable
    assert "no batched reading" in report.reason
    assert report.missing_from_run == ("larkin/cohort_n/arm",)


def test_a_batch_with_an_unlocatable_witness_is_never_reported_as_covered():
    """Dropping the unlocatable witness and judging the rest let a batch whose
    partition sentence the PDF extractor had mangled be reported as fully
    covered on the strength of the witnesses that survived."""
    gold = _gold()
    gold["batches"][0]["witnesses"].append({
        "witness_id": "w-mangled", "witness_type": "partition",
        "source_quote": "a sentence this extraction does not contain",
        "modality": "text"})
    report = assess([_reading()], gold, lambda _: PAPER, sha_for=lambda _: SHA)
    assert report.assessable
    counts = report.tally.as_dict()
    assert counts["gold_covered_batches"] == 0
    assert counts["gold_text_assessable_batches"] == 0
    assert counts["gold_unlocatable_batches"] == 1


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
    requires_frozen_evaluator()
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
    profile = load_profile(BENCH, "phase8_batch_v6_profile.json",
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

def test_the_harness_judges_a_whole_planned_run_against_the_key():
    """`coverage_for_run` had no test either, and it is where a run's readings,
    the benchmark's PDFs and the key are brought together.

    Built from the SAME planner the dry run uses, so this checks the key against
    every batch the run would make rather than against one hand-made record —
    which, now that both directions are checked, would be refused anyway.
    """
    requires_frozen_evaluator()
    import csv
    import types

    from react_review.dkb import load_runtime_knowledge
    from react_review.eval_excerpt import (
        benchmark_reviews,
        coverage_for_run,
        dry_run_collector,
        planned_batches,
    )
    from react_review.eval_profile import load_profile
    from react_review.retrieval.local_pdf import _pdf_text
    from react_review.tools.extract_source import (
        SELECTION_METHOD_ID,
        SELECTION_VERSION,
        select_excerpt,
    )

    paper = BENCH / "raw/sources/larkin_2015.pdf"
    if not paper.is_file():
        pytest.skip("the source paper is copyrighted and not in the repo")

    root = BENCH.parents[2]
    rows = list(csv.DictReader(
        (BENCH / "audit_template.csv").open(encoding="utf-8-sig")))
    profile = load_profile(BENCH, "phase8_batch_v6_profile.json",
                           answer_key_ids=[r["audit_id"] for r in rows])
    collector = dry_run_collector(profile.run_contract, load_runtime_knowledge(
        root / "configs/knowledge.seed.json", root / "configs/ontology"))

    text = _pdf_text(paper)
    readings = []
    for group in planned_batches(collector,
                                 benchmark_reviews(rows, profile.targets),
                                 profile.run_contract.extraction_routes["value"]):
        excerpt, spans = select_excerpt(
            text, target=(collector._concept_for(group.key.field_type)
                          or group.key.raw_field_name or group.key.field_type),
            raw_label=group.key.raw_field_name, field_type=group.key.field_type,
            variants=collector._concept_variants_for(group.key.field_type))
        readings.append(BatchReadingRecord(
            execution_id=f"{group.key.field_type}/{group.key.raw_field_name}",
            study_id=group.claims[0].study_id, field_type=group.key.field_type,
            target_shape=group.shape,
            claim_ids=[c.review_data_id for c in group.claims],
            excerpt_provenance=ExcerptProvenance(
                windowed=len(excerpt) != len(text), source_chars=len(text),
                excerpt_chars=len(excerpt), spans=spans,
                selection_method_id=SELECTION_METHOD_ID,
                selection_version=SELECTION_VERSION)))

    studies = [types.SimpleNamespace(study_id="larkin_2015",
                                     source_pdf="raw/sources/larkin_2015.pdf")]
    report = coverage_for_run(types.SimpleNamespace(batch_readings=readings),
                              studies, BENCH, profile.excerpt_gold_path)
    assert report.assessable, report.reason
    counts = report.tally
    assert counts.gold_text_assessable_batches == len(readings) == 7
    assert counts.gold_covered_batches == 7
    assert counts.gold_missing_batches == 0
    assert counts.gold_unlocatable_batches == 0
    assert report.unjudged_run_batches == () and report.missing_from_run == ()


def test_the_harness_is_silent_when_a_benchmark_publishes_no_key():
    """Silent, not zero: a benchmark without a key has not judged its windows."""
    import types

    from react_review.eval_excerpt import coverage_for_run

    results = types.SimpleNamespace(batch_readings=[_reading()])
    assert coverage_for_run(results, [], BENCH, None) is None


# --- the selector's OUTPUT is pinned, not compared to itself ----------------

#: A frozen synthetic document. Long enough to force the ceiling and to make the
#: last block truncate, which is the case that was wrong.
_PIN_TEXT = "".join(
    f"block {i} sample size total randomised patients {i * 7 % 97} "
    for i in range(4000))
_PIN_QUERY = dict(target="sample size", raw_label="Randomized patients, n",
                  field_type="sample_size",
                  variants=["number randomised", "total n"])
_PIN_EXCERPT_SHA = "CDE732A036692E03C4F6F3020514DBE1"
_PIN_SPANS = [(0, 3000), (2500, 5500), (7500, 10500), (17500, 20500),
              (20000, 23000), (27500, 30500), (37500, 39291)]


def test_the_selector_still_sends_the_bytes_it_was_published_sending():
    """A pin on the OUTPUT, not a comparison of two current implementations.

    `select_excerpt() == _paper_excerpt()` only says the two agree today; both
    could move together and every recording made under targeted_v5_batch would
    stop replaying, reported as a missing recording. The excerpt is part of the
    prompt, so its bytes are part of the prompt contract.
    """
    import hashlib

    from react_review.tools.extract_source import select_excerpt

    excerpt, spans = select_excerpt(_PIN_TEXT, **_PIN_QUERY)
    assert hashlib.sha256(excerpt.encode("utf-8")).hexdigest().upper()[:32] == \
        _PIN_EXCERPT_SHA, (
        "the excerpt sent for this query changed. If that is intended it is a "
        "new prompt version, not a new hash typed into this test")


def test_the_spans_the_selector_reports_are_pinned_too():
    """The report is not the prompt, so it has its own pin — and its own
    version. v1 read them off the markers and over-reported the truncated
    block; the numbers here are what was actually sent.
    """
    from react_review.tools.extract_source import SELECTION_VERSION, select_excerpt

    excerpt, spans = select_excerpt(_PIN_TEXT, **_PIN_QUERY)
    assert spans == _PIN_SPANS
    assert SELECTION_VERSION == "v2"
    markers = sum(len(f"\n\n[SOURCE EXCERPT {a}:{b}]\n") for a, b in spans)
    assert sum(b - a for a, b in spans) + markers == len(excerpt)
