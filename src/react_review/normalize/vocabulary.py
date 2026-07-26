"""Tier-2 controlled + extensible field_type vocabulary.

Maps a review's raw column name to a canonical ``field_type``. The vocabulary is
the shared target both sides normalise INTO, so review "EFT/ EAT" and a source
paper's "EAT thickness average" both resolve to ``eat_thickness`` and can match.

Resolution here is the DETERMINISTIC part (synonym lookup + unit disambiguation);
the LLM fallback that extends the vocabulary on a miss lives in the
``normalize_field`` tool. The vocabulary persists to JSON so it grows into a
living cross-domain dictionary rather than a hand-maintained static table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from react_review.normalize.units import normalize_unit


def _norm_name(s: str) -> str:
    """Normalised synonym key: lower-case, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class FieldTypeEntry(BaseModel):
    """One canonical concept in the vocabulary."""

    field_type: str
    concept: str = ""
    value_type: str = "numeric"          # numeric | text | categorical
    default_unit: str = ""
    synonyms: list[str] = Field(default_factory=list)

    def all_names(self) -> list[str]:
        """Every surface form this entry answers to (field_type + concept + synonyms)."""
        return [self.field_type, self.concept, *self.synonyms]


class Vocabulary(BaseModel):
    """A name-keyed set of :class:`FieldTypeEntry`."""

    entries: dict[str, FieldTypeEntry] = Field(default_factory=dict)

    # ---- persistence ----
    @classmethod
    def from_json(cls, path: Path | str) -> "Vocabulary":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        entries = {
            ft: FieldTypeEntry(field_type=ft, **{k: v for k, v in body.items() if k != "field_type"})
            for ft, body in data.items()
        }
        return cls(entries=entries)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        body = {
            ft: e.model_dump(exclude={"field_type"})
            for ft, e in self.entries.items()
        }
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- mutation ----
    def add(self, entry: FieldTypeEntry) -> None:
        """Register (or replace) a field_type entry."""
        self.entries[entry.field_type] = entry

    def field_types(self) -> list[str]:
        return sorted(self.entries)

    # ---- resolution (deterministic) ----
    def candidates(self, raw_name: str) -> list[str]:
        """field_types whose surface forms include ``raw_name`` (normalised)."""
        n = _norm_name(raw_name)
        if not n:
            return []
        return [
            ft
            for ft, e in self.entries.items()
            if any(_norm_name(s) == n for s in e.all_names())
        ]

    def resolve(self, raw_name: str, unit: str = "") -> str | None:
        """Resolve ``raw_name`` to a field_type, or None if unknown/ambiguous.

        One candidate → that field_type. Several candidates (e.g. "EFT/ EAT" maps
        to both eat_thickness and eat_volume) → disambiguate by the reported unit
        against each candidate's ``default_unit``. If still ambiguous, return None
        so the caller can fall back to the LLM.
        """
        cands = self.candidates(raw_name)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        if unit:
            nu = normalize_unit(unit)
            unit_matches = [
                ft for ft in cands
                if normalize_unit(self.entries[ft].default_unit) == nu
            ]
            if len(unit_matches) == 1:
                return unit_matches[0]
        return None
