"""The baseline a version decision is made against cannot drift.

A version number is a claim about behaviour. Deciding it by reading the diff is
the judgement this project keeps finding unreliable: "only provenance wiring" is
what somebody believes before discovering the total moved from 944 to 945.

So the claim is settled against a frozen corpus, and everything that could make
that settlement meaningless is pinned: the corpus, the generator that measures
it, the vector's shape, and the commit it describes.
"""
from __future__ import annotations

import json

from react_review.contracts import repo_root

CORPUS = "eval/baselines/aggregation_corpus.json"
BASELINE = "eval/baselines/aggregation_behavior_baseline.json"

#: Changing either is a decision: it means the thing a version is judged against
#: has moved, and every earlier version's judgement was made against something
#: else.
PINNED = {
    CORPUS: "220AC3E0686D44F4",
    "eval/aggregation_behavior.py": "2C0A4D855DD5E11B",
}


def _baseline() -> dict:
    return json.loads((repo_root() / BASELINE).read_text(encoding="utf-8"))


def test_the_baseline_names_the_commit_it_describes():
    body = _baseline()
    assert len(body["baseline_commit"]) == 40
    assert body["behavior_vector_version"] == 1
    assert body["generator_id"] == "aggregation_behavior"


def test_the_baseline_agrees_with_itself_about_how_many_cases_it_has():
    body = _baseline()
    assert len(body["case_ids"]) == body["case_count"]
    assert len(set(body["case_ids"])) == body["case_count"]
    assert set(body["cases"]) == set(body["case_ids"])


def test_the_corpus_is_the_one_the_baseline_was_written_from():
    from react_review.tools.aggregation_identity import _canonical_bytes
    import hashlib

    body = _baseline()
    digest = hashlib.sha256(
        _canonical_bytes(repo_root() / CORPUS)).hexdigest().upper()
    assert body["corpus_sha256"] == digest


def test_every_case_declares_what_it_is_supposed_to_demonstrate():
    """Without this a case can be green for a reason unrelated to its name.

    Two were, in the first draft: one quoted a population sentence its document
    did not contain, the other a total its quote did not print. Both refused,
    both for the wrong reason, and both would have been frozen as correct.
    """
    corpus = json.loads((repo_root() / CORPUS).read_text(encoding="utf-8"))
    for case in corpus["cases"]:
        assert case.get("expect"), case["case_id"]
        assert "released" in case["expect"], case["case_id"]


def test_the_corpus_still_covers_every_reproduced_release_path():
    """Losing a case must be an error, not a quiet improvement in the numbers."""
    corpus = json.loads((repo_root() / CORPUS).read_text(encoding="utf-8"))
    ids = {c["case_id"] for c in corpus["cases"]}
    for round_prefix, expected in (("r2_", 5), ("r3_", 4), ("r4_", 2)):
        assert sum(1 for i in ids if i.startswith(round_prefix)) == expected


def test_the_corpus_and_its_generator_are_pinned():
    import hashlib

    from react_review.tools.aggregation_identity import _canonical_bytes

    for path, expected in PINNED.items():
        digest = hashlib.sha256(
            _canonical_bytes(repo_root() / path)).hexdigest().upper()[:16]
        assert digest == expected, path


def test_the_pinned_files_are_lf_so_the_hashes_are_reproducible():
    attributes = (repo_root() / ".gitattributes").read_text(encoding="utf-8")
    assert "eval/baselines/*.json text eol=lf" in attributes
    assert "eval/aggregation_behavior.py text eol=lf" in attributes
