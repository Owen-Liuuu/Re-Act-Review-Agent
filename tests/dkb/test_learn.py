"""Heavy unit tests for the DKB developer LEARN module (DKB-4)."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.dkb import (
    KnowledgeBase,
    KnowledgeEntry,
    LearningSession,
    Provenance,
    load_proposals,
    save_proposals,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _prop(field_type: str, synonyms: list[str] | None = None) -> KnowledgeEntry:
    return KnowledgeEntry(field_type=field_type, concept=field_type,
                          synonyms=synonyms if synonyms is not None else [field_type.upper()],
                          provenance=Provenance(source="llm"), status="provisional")


# --- ingest / repeated agreement ---

def test_single_ingest_adds_provisional_not_promoted():
    kb = KnowledgeBase()
    s = LearningSession(kb, threshold=3)
    promoted = s.ingest([_prop("hba1c")])
    assert promoted == []
    assert "hba1c" in kb.entries and kb.entries["hba1c"].status == "provisional"


def test_repeated_agreement_across_runs_promotes():
    kb = KnowledgeBase()
    s = LearningSession(kb, threshold=3)
    assert s.ingest([_prop("hba1c")]) == []                 # run 1
    assert s.ingest([_prop("hba1c")]) == []                 # run 2
    assert s.ingest([_prop("hba1c")]) == ["hba1c"]          # run 3 → promoted
    assert kb.entries["hba1c"].status == "authoritative"


def test_dedup_within_one_batch_counts_once():
    kb = KnowledgeBase()
    s = LearningSession(kb, threshold=2)
    # three proposals of the SAME concept in ONE run count as one agreement
    assert s.ingest([_prop("hba1c"), _prop("hba1c"), _prop("hba1c")]) == []
    assert kb.entries["hba1c"].status == "provisional"
    assert s.ingest([_prop("hba1c")]) == ["hba1c"]          # a second RUN promotes it


def test_threshold_one_promotes_on_first_ingest():
    kb = KnowledgeBase()
    assert LearningSession(kb, threshold=1).ingest([_prop("hba1c")]) == ["hba1c"]
    assert kb.entries["hba1c"].status == "authoritative"


def test_empty_ingest_is_noop():
    kb = KnowledgeBase()
    assert LearningSession(kb).ingest([]) == []
    assert kb.entries == {}


# --- confirm / pending ---

def test_confirm_promotes_immediately():
    kb = KnowledgeBase()
    s = LearningSession(kb, threshold=99)
    s.ingest([_prop("hba1c")])
    assert s.confirm("hba1c") is True
    assert kb.entries["hba1c"].status == "authoritative"
    assert s.confirm("hba1c") is False          # already authoritative → no-op


def test_pending_lists_only_provisional():
    kb = KnowledgeBase.from_json(CONFIGS / "knowledge.seed.json")   # all authoritative
    s = LearningSession(kb, threshold=3)
    s.ingest([_prop("hba1c"), _prop("ldl")])
    assert s.pending() == ["hba1c", "ldl"]      # sorted; seed concepts excluded


# --- merging semantics ---

def test_synonyms_unioned_across_proposals():
    kb = KnowledgeBase()
    s = LearningSession(kb, threshold=5)
    s.ingest([_prop("hba1c", ["A1c"])])
    s.ingest([_prop("hba1c", ["glycated hb"])])
    syns = kb.entries["hba1c"].synonyms
    assert "A1c" in syns and "glycated hb" in syns


def test_existing_authoritative_is_left_untouched():
    kb = KnowledgeBase.from_json(CONFIGS / "knowledge.seed.json")
    before = list(kb.entries["bmi"].synonyms)
    LearningSession(kb, threshold=1).ingest([_prop("bmi", ["bogus synonym"])])
    assert kb.entries["bmi"].status == "authoritative"       # unchanged
    assert kb.entries["bmi"].synonyms == before              # not modified by a proposal


# --- persistence ---

def test_kb_save_load_round_trip_preserves_status(tmp_path):
    kb = KnowledgeBase()
    s = LearningSession(kb, threshold=1)
    s.ingest([_prop("hba1c")])                   # promoted → authoritative
    p = tmp_path / "kb.json"
    s.save(p)
    assert KnowledgeBase.from_json(p).entries["hba1c"].status == "authoritative"


def test_proposals_save_load_round_trip(tmp_path):
    props = [_prop("hba1c", ["A1c"]), _prop("ldl")]
    p = tmp_path / "proposals.json"
    save_proposals(props, p)
    loaded = load_proposals(p)
    assert [e.field_type for e in loaded] == ["hba1c", "ldl"]
    assert loaded[0].synonyms == ["A1c"] and loaded[0].status == "provisional"


def test_learn_then_resolve_becomes_deterministic():
    # after promotion, the concept resolves via DKB-1 (no LLM) — the payoff
    kb = KnowledgeBase()
    LearningSession(kb, threshold=1).ingest([_prop("hba1c", ["HbA1c", "A1c"])])
    assert kb.resolve("A1c") == "hba1c"          # now a deterministic hit
