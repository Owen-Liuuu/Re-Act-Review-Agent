"""Tests for DKB promotion + ontology import (DKB-3)."""
from __future__ import annotations

from pathlib import Path

from react_review.dkb import (
    KnowledgeBase,
    KnowledgeEntry,
    Provenance,
    PromotionTracker,
    import_ontology,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _provisional_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry(field_type="hba1c", concept="glycated haemoglobin",
                          provenance=Provenance(source="llm"), status="provisional"))
    return kb


# --- promotion ---

def test_repeated_agreement_promotes_at_threshold():
    kb = _provisional_kb()
    tr = PromotionTracker(kb, threshold=3)
    assert tr.observe("hba1c") is False        # 1
    assert tr.observe("hba1c") is False         # 2
    assert kb.entries["hba1c"].status == "provisional"
    assert tr.observe("hba1c") is True          # 3 → promoted
    assert kb.entries["hba1c"].status == "authoritative"


def test_human_confirm_promotes_immediately():
    kb = _provisional_kb()
    assert PromotionTracker(kb).confirm("hba1c") is True
    assert kb.entries["hba1c"].status == "authoritative"


def test_observe_on_authoritative_is_noop():
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry(field_type="bmi", status="authoritative"))
    assert PromotionTracker(kb).observe("bmi") is False
    assert PromotionTracker(kb).observe("unknown") is False


# --- ontology import ---

def test_import_ontology_adds_authoritative():
    kb = KnowledgeBase.from_json(CONFIGS / "knowledge.seed.json")
    added, merged = import_ontology(kb, CONFIGS / "ontology" / "labs.json", source="LOINC")
    assert added == 3 and merged == 0
    e = kb.entries["hba1c"]
    assert e.status == "authoritative" and e.provenance.source == "ontology:LOINC"
    # importing an authoritative concept makes it a deterministic KB hit
    assert kb.resolve("HbA1c") == "hba1c"


def test_import_ontology_merges_and_promotes_existing():
    kb = _provisional_kb()                    # hba1c is provisional
    added, merged = import_ontology(kb, CONFIGS / "ontology" / "labs.json", source="LOINC")
    assert merged == 1                        # hba1c merged, not re-added
    assert kb.entries["hba1c"].status == "authoritative"   # lifted by the ontology
    assert "A1c" in kb.entries["hba1c"].synonyms            # synonyms unioned
