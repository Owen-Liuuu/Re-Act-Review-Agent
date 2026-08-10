"""Projection comparison — the acceptance tool every later unit leans on.

"Byte-identical to the previous run" stops working the moment results carry a
new field, and a check that has to be relaxed to keep passing is a check nobody
believes. What this measures instead is that nothing the OLD artifact recorded
has moved. It has to be tested, because every P8-0 acceptance claim rests on it.
"""
from __future__ import annotations

import json

from react_review.eval_projection import added_keys, compare, project


def _write(path, body):
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_an_added_field_is_information_not_a_failure(tmp_path):
    old = _write(tmp_path / "old.json", {"metrics": {"n": 2},
                                         "rows": [{"audit_id": "A1", "label": "match"}]})
    new = _write(tmp_path / "new.json", {"metrics": {"n": 2, "target": {"ok": 1}},
                                         "rows": [{"audit_id": "A1", "label": "match",
                                                   "scope_check": "ok"}]})
    result = compare(old, new)
    assert result["differences"] == []
    assert result["added_fields"] == ["metrics.target", "rows.[].scope_check"]


def test_a_changed_value_three_levels_down_is_caught(tmp_path):
    old = _write(tmp_path / "old.json",
                 {"rows": [{"audit_id": "A1", "numeric": {"ci": {"lower": 0.31}}}]})
    new = _write(tmp_path / "new.json",
                 {"rows": [{"audit_id": "A1", "numeric": {"ci": {"lower": 0.37,
                                                                 "level": 95}}}]})
    result = compare(old, new)
    assert result["differences"] == ["rows[0].numeric.ci.lower: 0.31 -> 0.37"]


def test_a_dropped_field_is_caught(tmp_path):
    old = _write(tmp_path / "old.json", {"rows": [{"audit_id": "A1", "label": "match"}]})
    new = _write(tmp_path / "new.json", {"rows": [{"audit_id": "A1"}]})
    assert "MISSING" in compare(old, new)["differences"][0]


def test_rows_are_matched_by_id_not_position(tmp_path):
    old = _write(tmp_path / "old.json", {"rows": [{"audit_id": "A2", "label": "match"},
                                                  {"audit_id": "A1", "label": "mismatch"}]})
    new = _write(tmp_path / "new.json", {"rows": [{"audit_id": "A1", "label": "mismatch"},
                                                  {"audit_id": "A2", "label": "match"}]})
    assert compare(old, new)["differences"] == []


def test_the_run_block_is_ignored(tmp_path):
    """Paths and execution modes differ by design between a record and a replay."""
    old = _write(tmp_path / "old.json", {"run": {"extraction_mode": "record"},
                                         "metrics": {"n": 1}})
    new = _write(tmp_path / "new.json", {"run": {"extraction_mode": "replay"},
                                         "metrics": {"n": 1}})
    assert compare(old, new)["differences"] == []


def test_project_keeps_only_the_old_shape():
    assert project({"a": 1}, {"a": 2, "b": 3}) == {"a": 2}
    assert added_keys({"a": 1}, {"a": 2, "b": 3}) == ["b"]
