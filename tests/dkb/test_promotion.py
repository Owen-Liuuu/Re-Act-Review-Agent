"""Tests for DKB promotion + ontology import (DKB-3)."""
from __future__ import annotations

import json
from pathlib import Path

from react_review.dkb import (
    KnowledgeBase,
    KnowledgeEntry,
    Provenance,
    PromotionTracker,
    import_ontology,
    load_runtime_knowledge,
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
    assert kb.entries["hba1c"].provenance.source == "ontology:LOINC"
    record = kb.imports[0]
    assert record.merged_field_types == ["hba1c"]
    assert {c.field for c in record.conflicts} >= {"default_unit", "domain"}


def test_curated_ontology_explicitly_overrides_conflicting_seed_values(tmp_path):
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry(
        field_type="marker", concept="seed concept", value_type="numeric",
        default_unit="mm", scope="cohort", synonyms=["seed name"]))
    path = tmp_path / "curated.json"
    path.write_text(json.dumps({
        "marker": {
            "concept": "curated concept", "value_type": "numeric",
            "default_unit": "cm3", "scope": "study",
            "synonyms": ["curated name"],
        },
    }), encoding="utf-8")

    assert import_ontology(kb, path, source="curated") == (0, 1)
    marker = kb.entries["marker"]
    assert (marker.concept, marker.default_unit, marker.scope) == (
        "curated concept", "cm3", "study")
    assert marker.synonyms == ["seed name", "curated name"]
    conflicts = {c.field: c for c in kb.imports[0].conflicts}
    assert set(conflicts) == {"concept", "default_unit", "scope"}
    assert conflicts["default_unit"].resolution == "ontology_override"
    assert conflicts["default_unit"].seed_value == "mm"
    assert conflicts["default_unit"].ontology_value == "cm3"


def test_runtime_loader_is_deterministic_and_loads_all_ontology_slices():
    seed = CONFIGS / "knowledge.seed.json"
    directory = CONFIGS / "ontology"
    first = load_runtime_knowledge(seed, directory)
    second = load_runtime_knowledge(seed, directory)

    assert len(first.entries) == 12
    assert first.version == second.version == first.fingerprint()
    assert len(first.version) == 64
    assert [record.source for record in first.imports] == ["ontology:labs"]
    record = first.imports[0]
    assert record.added == 3 and record.merged == 0
    assert record.concepts_before == 9 and record.concepts_after == 12
    assert len(record.sha256) == 64
