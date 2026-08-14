"""The frozen D4 transition is checked against produced claim-level rows."""
from __future__ import annotations

from copy import deepcopy

from eval.check_evidence_adequacy import (
    load_transition,
    validate_results,
    validate_transition,
)


def _produced():
    rows = []
    for expected in load_transition():
        failed = expected["required_failed_axis"]
        adequacy = {
            "status": expected["expected_adequacy"],
            "document_scope": expected["document_scope"],
            "axis_results": ({failed: {"status": "fail"}} if failed else {}),
        }
        rows.append({
            "audit_id": expected["audit_id"],
            "predicted_label": expected["expected_new_label"],
            "evidence_adequacy": adequacy,
        })
    rows.extend([
        {"audit_id": "A034", "predicted_label": "match",
         "evidence_adequacy": {"status": "sufficient",
                                "document_scope": "abstract_only"}},
        {"audit_id": "A035", "predicted_label": "match",
         "evidence_adequacy": {"status": "sufficient",
                                "document_scope": "abstract_only"}},
    ])
    return rows


def test_frozen_transition_and_positive_controls_pass_together():
    transition = load_transition()
    assert validate_transition(transition) == []
    assert validate_results(transition, _produced()) == []


def test_a_wrong_binding_cannot_still_arrive_as_mismatch():
    produced = _produced()
    next(row for row in produced if row["audit_id"] == "A028")[
        "predicted_label"] = "mismatch"

    errors = validate_results(load_transition(), produced)

    assert any("A028 label" in error for error in errors)


def test_positive_abstract_controls_cannot_be_blanket_rejected():
    produced = deepcopy(_produced())
    control = next(row for row in produced if row["audit_id"] == "A034")
    control["predicted_label"] = "not_comparable"
    control["evidence_adequacy"]["status"] = "insufficient"

    errors = validate_results(load_transition(), produced)

    assert any("A034 must remain sufficient/match" in error for error in errors)
