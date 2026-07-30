"""End-to-end tests for the `react-review learn` developer subcommand."""
from __future__ import annotations

from pathlib import Path

from react_review.cli import _learn_main
from react_review.dkb import (
    KnowledgeBase,
    KnowledgeEntry,
    Provenance,
    save_proposals,
)

SEED = Path(__file__).resolve().parents[1] / "configs" / "knowledge.seed.json"


def _proposals_file(path: Path, field_type: str, synonyms: list[str]) -> Path:
    save_proposals(
        [KnowledgeEntry(field_type=field_type, concept=field_type, synonyms=synonyms,
                        provenance=Provenance(source="llm"), status="provisional")],
        path)
    return path


def test_learn_promotes_after_threshold_across_run_files(tmp_path, capsys):
    # two run batches (two files) of the same concept, threshold 2 → promoted
    run1 = _proposals_file(tmp_path / "run1.json", "hba1c", ["A1c"])
    run2 = _proposals_file(tmp_path / "run2.json", "hba1c", ["glycated hb"])
    out = tmp_path / "curated.json"

    _learn_main([str(run1), str(run2), "--kb", str(SEED), "--out", str(out), "--threshold", "2"])
    printed = capsys.readouterr().out
    assert "promoted ['hba1c']" in printed

    kb = KnowledgeBase.from_json(out)
    assert kb.entries["hba1c"].status == "authoritative"
    # synonyms from both run batches survived the merge
    assert {"A1c", "glycated hb"} <= set(kb.entries["hba1c"].synonyms)


def test_learn_below_threshold_stays_pending(tmp_path, capsys):
    run1 = _proposals_file(tmp_path / "run1.json", "novel_marker", ["NM"])
    out = tmp_path / "curated.json"

    _learn_main([str(run1), "--kb", str(SEED), "--out", str(out), "--threshold", "3"])
    printed = capsys.readouterr().out
    assert "novel_marker" in printed          # listed as pending

    kb = KnowledgeBase.from_json(out)
    assert kb.entries["novel_marker"].status == "provisional"


def test_learn_confirm_forces_promotion(tmp_path, capsys):
    run1 = _proposals_file(tmp_path / "run1.json", "novel_marker", ["NM"])
    out = tmp_path / "curated.json"

    _learn_main([str(run1), "--kb", str(SEED), "--out", str(out),
                 "--threshold", "99", "--confirm", "novel_marker"])
    assert "promoted" in capsys.readouterr().out

    kb = KnowledgeBase.from_json(out)
    assert kb.entries["novel_marker"].status == "authoritative"


def test_learn_list_does_not_write(tmp_path, capsys):
    run1 = _proposals_file(tmp_path / "run1.json", "novel_marker", ["NM"])
    out = tmp_path / "curated.json"

    _learn_main([str(run1), "--kb", str(SEED), "--out", str(out), "--list"])
    assert "novel_marker" in capsys.readouterr().out
    assert not out.exists()                   # --list is read-only
