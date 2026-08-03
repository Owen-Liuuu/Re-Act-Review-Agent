"""DKB knowledge base — deterministic load / resolve / scope (DKB-1).

The evolution of Tier-2 Vocabulary: same synonym + unit resolution, plus
multi-signal disambiguation (modality) and a per-concept `scope`. Still 100%
deterministic and LLM-free; the RAG retrieval + agent judgement + provisional
write-back land in DKB-2 (dkb/retrieval.py, dkb/agent.py).
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field

from react_review.dkb.schema import KnowledgeEntry
from react_review.schemas.knowledge import KnowledgeImportRecord
from react_review.normalize.units import normalize_unit


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


class KnowledgeBase(BaseModel):
    """A field_type-keyed set of :class:`KnowledgeEntry`."""

    entries: dict[str, KnowledgeEntry] = Field(default_factory=dict)
    version: str = ""              # snapshot id for reproducible runs (DKB-3)
    imports: list[KnowledgeImportRecord] = Field(default_factory=list)

    # ---- persistence (same JSON shape as the old vocabulary seed) ----
    @classmethod
    def from_json(cls, path: Path | str) -> "KnowledgeBase":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        entries = {
            ft: KnowledgeEntry(field_type=ft,
                               **{k: v for k, v in body.items() if k != "field_type"})
            for ft, body in data.items()
        }
        return cls(entries=entries)

    def save(self, path: Path | str) -> None:
        body = {ft: e.model_dump(exclude={"field_type"}) for ft, e in self.entries.items()}
        Path(path).write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")

    def fingerprint(self) -> str:
        """Content hash of the effective entries, independent of file ordering."""
        body = {
            ft: self.entries[ft].model_dump(mode="json", exclude={"field_type"})
            for ft in sorted(self.entries)
        }
        material = json.dumps(
            body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    # ---- mutation ----
    def add(self, entry: KnowledgeEntry) -> None:
        self.entries[entry.field_type] = entry

    def field_types(self) -> list[str]:
        return sorted(self.entries)

    # ---- knowledge accessors ----
    def scope_of(self, field_type: str) -> str:
        """"study" for study-level concepts (country, sample_size…), else "cohort"."""
        e = self.entries.get(field_type)
        return e.scope if e else "cohort"

    # ---- resolution (deterministic, multi-signal) ----
    def candidates(self, raw_name: str) -> list[str]:
        n = _norm_name(raw_name)
        if not n:
            return []
        return [ft for ft, e in self.entries.items()
                if any(_norm_name(s) == n for s in e.all_names())]

    def resolve(self, raw_name: str, unit: str = "", modality: str = "") -> str | None:
        """Resolve a raw name to a field_type using name → unit → modality signals.

        One candidate wins outright. For an ambiguous name (e.g. "EFT/ EAT" → both
        eat_thickness and eat_volume) disambiguate by an explicit modality rule
        first (CT→eat_volume, echo→eat_thickness), then by the reported unit. Still
        ambiguous → None so the caller can fall back to the LLM (DKB-2).
        """
        cands = self.candidates(raw_name)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        if modality:
            m = modality.strip().lower()
            for ft in cands:
                for signal, target in self.entries[ft].disambiguation.get("modality", {}).items():
                    if signal in m and target in cands:
                        return target
        if unit:
            nu = normalize_unit(unit)
            matches = [ft for ft in cands
                       if normalize_unit(self.entries[ft].default_unit) == nu]
            if len(matches) == 1:
                return matches[0]
        return None
