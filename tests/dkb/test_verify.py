"""Deterministic verification of LLM-proposed field_type mappings (the gate)."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.dkb import (
    KnowledgeBase,
    KnowledgeEntry,
    evidence_contradicts,
    verify_candidate,
)
from react_review.normalize.units import unit_kind

SEED = Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json"


def _kb() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry(field_type="eat_thickness", concept="epicardial fat thickness",
                          default_unit="mm"))
    kb.add(KnowledgeEntry(field_type="eat_volume", concept="epicardial fat volume",
                          default_unit="cm3"))
    kb.add(KnowledgeEntry(field_type="bmi", concept="body mass index",
                          default_unit="kg/m2"))                       # compound → unknown kind
    kb.add(KnowledgeEntry(field_type="hba1c", concept="glycated haemoglobin",
                          default_unit="%", plausible_range=[4.0, 15.0]))
    return kb


def _verify(kb, ft, **kw):
    base = dict(unit="", value=None, is_new=False, confidence=0.9, grounded_on=["x"])
    base.update(kw)
    return verify_candidate(ft, kb=kb, **base)


# --- unit_kind ---

@pytest.mark.parametrize("unit, kind", [
    ("mm", "length"), ("cm", "length"), ("m", "length"),
    ("ml", "volume"), ("cm3", "volume"), ("cc", "volume"), ("L", "volume"),
    ("kg", "mass"), ("mg", "mass"),
    ("%", "percent"), ("years", "time"), ("yr", "time"),
    ("kg/m2", "unknown"), ("mg/dl", "unknown"),          # compound → cannot judge
    ("", "unknown"), ("widgets", "unknown"),
])
def test_unit_kind(unit, kind):
    assert unit_kind(unit) == kind


# --- grounding check ---

def test_grounding_passes_when_cited():
    v = _verify(_kb(), "eat_thickness", is_new=True, grounded_on=["eat_thickness"],
                confidence=0.05)
    assert v.ok and v.checks["grounding"]


def test_grounding_passes_on_confidence_alone():
    v = _verify(_kb(), "novel_marker", is_new=True, grounded_on=[], confidence=0.6)
    assert v.ok and v.checks["grounding"]


def test_grounding_fails_ungrounded_low_confidence():
    v = _verify(_kb(), "novel_marker", is_new=True, grounded_on=[], confidence=0.1)
    assert not v.ok and v.checks["grounding"] is False
    assert "ungrounded" in v.reason


def test_mapping_to_existing_concept_is_grounded_by_construction():
    # is_new=False means the LLM chose a retrieved concept → grounded even at conf 0
    v = _verify(_kb(), "eat_thickness", is_new=False, grounded_on=[], confidence=0.0,
                unit="mm")
    assert v.ok and v.checks["grounding"]


# --- unit-kind check ---

def test_unit_conflict_rejects_volume_for_a_length_concept():
    v = _verify(_kb(), "eat_thickness", unit="cm3")       # thickness is mm (length)
    assert not v.ok and v.checks["unit"] is False
    assert "different kind" in v.reason


def test_unit_same_kind_is_accepted():
    v = _verify(_kb(), "eat_thickness", unit="cm")        # cm vs mm — both length, fine
    assert v.ok and v.checks["unit"]


def test_equivalent_unit_is_accepted():
    v = _verify(_kb(), "eat_volume", unit="mL")           # mL == cm3 canonical
    assert v.ok and v.checks["unit"]


def test_compound_expected_unit_is_never_a_conflict():
    v = _verify(_kb(), "bmi", unit="anything")            # kg/m2 → unknown kind → skip
    assert v.ok and v.checks["unit"]


# --- range check ---

def test_value_outside_plausible_range_is_rejected():
    v = _verify(_kb(), "hba1c", unit="%", value="300")
    assert not v.ok and v.checks["range"] is False
    assert "plausible range" in v.reason


def test_value_within_range_is_accepted():
    v = _verify(_kb(), "hba1c", unit="%", value="7.0 ± 0.5")
    assert v.ok and v.checks["range"]


# --- unknown concept: nothing to check against ---

def test_new_concept_not_in_kb_skips_unit_and_range():
    v = _verify(_kb(), "brand_new_score", is_new=True, grounded_on=[], confidence=0.6,
                unit="cm3", value="9999")
    assert v.ok                                            # no expectation to violate


# --- evidence_contradicts (the source-side back-check) ---

def test_evidence_contradicts_on_unit_kind():
    reason = evidence_contradicts("eat_thickness", kb=_kb(), unit="cm3", value="52.3")
    assert "different kind" in reason


def test_evidence_agrees_when_unit_kind_matches():
    assert evidence_contradicts("eat_thickness", kb=_kb(), unit="mm", value="6.6") == ""


def test_evidence_contradicts_on_range():
    assert "plausible range" in evidence_contradicts("hba1c", kb=_kb(), unit="%", value="300")


def test_evidence_unknown_field_type_never_contradicts():
    assert evidence_contradicts("not_a_concept", kb=_kb(), unit="cm3", value="1") == ""


# --- the SHIPPED seed's plausible ranges are actually wired ---

def test_seed_ranges_reject_implausible_values():
    kb = KnowledgeBase.from_json(SEED)
    assert kb.entries["bmi"].plausible_range == [8, 100]          # bmi's only value gate
    # an impossible BMI mapped by the LLM is rejected by the range gate
    assert not verify_candidate("bmi", kb=kb, value="300", is_new=False,
                                confidence=0.9, grounded_on=["bmi"]).ok
    # a real benchmark BMI passes
    assert verify_candidate("bmi", kb=kb, value="20.57 ± 1.7", is_new=False,
                            confidence=0.9, grounded_on=["bmi"]).ok
    # an EAT volume (52 cm3) mislabelled as thickness is out of [0, 30] mm
    assert not verify_candidate("eat_thickness", kb=kb, value="52.3", is_new=False,
                                confidence=0.9, grounded_on=["eat_thickness"]).ok
