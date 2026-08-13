"""Telling a window that dropped the evidence from a paper that never had it.

Both produce the same sentence from the extractor — "the paper does not report
it" — so the only thing that separates them is a record of what was sent and a
key saying where the answer lives. Neither exists inside a run, which is why the
production side records provenance and says nothing about coverage, and the
judging happens here against witnesses a human wrote down.
"""
from __future__ import annotations

import json

import pytest

from react_review.contracts import repo_root
from react_review.eval_excerpt import (
    COVERED,
    FULLTEXT_UNLOCATABLE,
    NON_TEXT_UNASSESSABLE,
    WINDOW_MISSED,
    classify,
    locate,
    tally,
)

BENCH = repo_root() / "eval/benchmarks/melanoma_checkpoint_2017"
GOLD = BENCH / "excerpt_gold_v1.json"
SOURCE_PDF = BENCH / "raw/sources/larkin_2015.pdf"

# The two line-break shapes that appear in one sentence of the real paper.
WRAPPED = ("A total of 945 patients underwent ran-\ndomization: 316 patients "
           "were assigned to the nivolumab group, 314 to the nivolumab-plus-\n"
           "ipilimumab group, and 315 to the ipilimumab group.")


# --- locating a quote in text a PDF extractor produced ----------------------

def test_a_word_broken_across_lines_is_still_found():
    """The hyphen IS the break: "ran- domization" is randomization."""
    assert locate(WRAPPED, "A total of 945 patients underwent randomization")


def test_a_hyphenated_word_broken_across_lines_is_still_found():
    """The hyphen belongs to the word: "nivolumab-plus- ipilimumab"."""
    assert locate(WRAPPED, "314 to the nivolumab-plus-ipilimumab group")


def test_a_quote_containing_both_kinds_of_break_is_found():
    """The regression that decided the rule.

    Resolving line-break hyphens one way loses the first; the other way loses
    the second; trying both readings of the whole document loses any quote that
    spans one of each — and the partition sentence spans exactly that, so a
    paper that states the partition was reported as not containing it.
    """
    found = locate(WRAPPED, "A total of 945 patients underwent randomization: "
                            "316 patients were assigned to the nivolumab group, "
                            "314 to the nivolumab-plus-ipilimumab group, and "
                            "315 to the ipilimumab group")
    assert found is not None


def test_the_offsets_returned_are_offsets_into_the_source():
    """A run's spans are source offsets; a match reported in normalised
    coordinates would silently be compared against a different ruler."""
    start, end = locate(WRAPPED, "315 to the ipilimumab group")
    assert WRAPPED[start:end] == "315 to the ipilimumab group"


def test_a_quote_the_extracted_text_does_not_contain_is_not_found():
    """Not something to try harder at: a fuzzier matcher would turn "the key's
    quote is not in what we extracted" into a coordinate somebody then treats
    as evidence."""
    assert locate(WRAPPED, "412 patients withdrew consent") is None


# --- the four outcomes ------------------------------------------------------

def test_evidence_inside_a_window_is_covered():
    outcome = classify(WRAPPED, quote="315 to the ipilimumab group",
                       modality="text", spans=[(0, len(WRAPPED))])
    assert outcome.outcome == COVERED and outcome.assessable


def test_evidence_the_window_did_not_include_is_the_selector_s_fault():
    """The only outcome that indicts the window."""
    outcome = classify(WRAPPED, quote="315 to the ipilimumab group",
                       modality="text", spans=[(0, 40)])
    assert outcome.outcome == WINDOW_MISSED and outcome.assessable


def test_a_quote_the_extractor_mangled_is_not_the_window_s_fault():
    outcome = classify(WRAPPED, quote="a sentence the PDF text does not carry",
                       modality="text", spans=[(0, len(WRAPPED))])
    assert outcome.outcome == FULLTEXT_UNLOCATABLE and not outcome.assessable


def test_a_figure_is_not_assessed_at_all():
    """Counting it as missing would blame the window for the extractor's limit."""
    outcome = classify(WRAPPED, quote="whatever the figure shows",
                       modality="figure", spans=[])
    assert outcome.outcome == NON_TEXT_UNASSESSABLE and not outcome.assessable


# --- counting -----------------------------------------------------------------

def _outcomes(text, quotes, spans):
    return [classify(text, quote=q, modality="text", spans=spans, witness_id=q)
            for q in quotes]


def test_a_batch_is_covered_only_when_every_witness_in_it_is():
    """One reading answers several claims, so a window holding four of five
    passages has failed the claim resting on the fifth."""
    quotes = ["316 patients were assigned to the nivolumab group",
              "315 to the ipilimumab group"]
    whole = _outcomes(WRAPPED, quotes, [(0, len(WRAPPED))])
    partial = _outcomes(WRAPPED, quotes, [(0, 130)])

    assert tally([(whole, True)]).as_dict() == {
        "windowed_batches": 1, "gold_text_assessable_batches": 1,
        "gold_covered_batches": 1, "gold_missing_batches": 0,
        "gold_unlocatable_batches": 0}
    assert tally([(partial, True)]).as_dict() == {
        "windowed_batches": 1, "gold_text_assessable_batches": 1,
        "gold_covered_batches": 0, "gold_missing_batches": 1,
        "gold_unlocatable_batches": 0}


def test_a_batch_with_nothing_assessable_is_counted_in_neither_column():
    """Otherwise a figure-only batch is either a free pass or a free failure."""
    outcomes = [classify(WRAPPED, quote="x", modality="figure", spans=[])]
    counted = tally([(outcomes, True)]).as_dict()
    assert counted["gold_text_assessable_batches"] == 0
    assert counted["gold_covered_batches"] == counted["gold_missing_batches"] == 0


def test_a_paper_short_enough_to_send_whole_is_not_counted_as_windowed():
    outcomes = _outcomes(WRAPPED, ["315 to the ipilimumab group"],
                         [(0, len(WRAPPED))])
    assert tally([(outcomes, False)]).as_dict()["windowed_batches"] == 0


# --- the key itself -----------------------------------------------------------

def test_the_gold_covers_only_routes_that_are_actually_batched():
    """Under phase8_batch the arm_identity route is targeted_v4, one call per
    claim. Counting those rows would count a window decision never made."""
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    claimed = {a for batch in gold["batches"] for a in batch["audit_ids"]}
    assert claimed.isdisjoint({"MA003", "MA005", "MA007"})


def test_the_aggregable_batch_proves_the_partition_separately_from_the_total():
    """MA002's answer-key quote establishes an explicit total and cannot
    establish a partition. safe_sum_v5 needs both, and needs the census read
    from an anchored passage — a different sentence, so a different witness."""
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    batch = next(b for b in gold["batches"] if b["field_type"] == "sample_size")
    kinds = {w["witness_type"] for w in batch["witnesses"]}
    assert {"explicit", "partition", "component"} <= kinds

    explicit = next(w for w in batch["witnesses"] if w["witness_type"] == "explicit")
    partition = next(w for w in batch["witnesses"] if w["witness_type"] == "partition")
    assert explicit["source_quote"] != partition["source_quote"]
    assert len(partition["source_quote"]) > len(explicit["source_quote"])


def test_every_audit_id_the_gold_names_exists_in_the_answer_key():
    import csv

    rows = {r["audit_id"] for r in csv.DictReader(
        (BENCH / "audit_template.csv").open(encoding="utf-8-sig"))}
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    for batch in gold["batches"]:
        assert set(batch["audit_ids"]) <= rows, batch["batch_id"]


@pytest.mark.skipif(not SOURCE_PDF.is_file(),
                    reason="the source paper is copyrighted and not in the repo")
def test_every_witness_is_locatable_in_the_paper_it_names():
    """The key checks itself against the document it claims to describe.

    Not decoration: the first draft of this file asserted a partition sentence
    that could not be found, and the generator refused to write it.
    """
    from react_review.agents.collector import _document_sha256
    from react_review.retrieval.local_pdf import _pdf_text

    # The retriever's own extraction, because the gold's offsets are compared
    # against spans a run recorded — and a run reads this function's output.
    text = _pdf_text(SOURCE_PDF)
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    assert gold["document_sha256"] == _document_sha256(text), (
        "the gold describes a different extraction of this paper")
    for batch in gold["batches"]:
        for witness in batch["witnesses"]:
            assert locate(text, witness["source_quote"]) is not None, \
                witness["witness_id"]


# --- what production records, and what it must not claim --------------------

def test_the_run_records_which_regions_it_sent():
    """Recorded whether or not the paper was windowed.

    "Not windowed" is an answer to the question. A record appearing only when
    something was cut would leave every other reading silent about whether
    anything had been, which is the ambiguity this exists to remove.
    """
    from react_review.tools.extract_source import (
        SELECTION_METHOD_ID,
        SELECTION_VERSION,
        select_excerpt,
    )

    excerpt, spans = select_excerpt("a short paper", target="x", raw_label="x",
                                    field_type="x")
    assert excerpt == "a short paper" and spans == [(0, len("a short paper"))]
    assert SELECTION_METHOD_ID and SELECTION_VERSION


def test_a_windowed_paper_reports_the_regions_it_kept():
    long_text = ("A total of 945 patients underwent randomization. " + "filler. " * 4000
                 + "315 to the ipilimumab group.")
    from react_review.tools.extract_source import select_excerpt

    excerpt, spans = select_excerpt(long_text, target="sample size",
                                    raw_label="Total, n", field_type="sample_size")
    assert len(excerpt) < len(long_text)
    assert spans and all(0 <= a < b <= len(long_text) for a, b in spans)


def test_the_production_record_says_what_was_sent_and_never_whether_it_was_enough():
    """A run asserting "the evidence was missing" would be asserting something
    about the paper from inside the run that failed to find it."""
    from react_review.schemas.batch import ExcerptProvenance

    fields = set(ExcerptProvenance.model_fields)
    assert fields == {"windowed", "source_chars", "excerpt_chars", "spans",
                      "selection_method_id", "selection_version"}
    assert not any("cover" in name or "missing" in name for name in fields)


def test_a_reading_without_that_measurement_gains_no_key():
    """A missing measurement must never be readable as "the whole paper"."""
    from react_review.schemas.batch import BatchReadingRecord

    assert "excerpt_provenance" not in BatchReadingRecord(
        study_id="s").model_dump(mode="json")
