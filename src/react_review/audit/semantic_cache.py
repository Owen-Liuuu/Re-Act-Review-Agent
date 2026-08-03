"""A recorded set of semantic judgements, so an audit can be re-run offline.

An LLM call is not reproducible, which is a problem for a tool whose output is
meant to be checkable. Recording each judgement against the exact question that
produced it lets a run be repeated with no model at all — and lets a reviewer
see every judgement the audit relied on, in one file.

``cache-only`` deliberately FAILS on a miss instead of quietly calling the model:
a run that claims to be reproducing a recording must not silently stop being one.
"""
from __future__ import annotations

import json
from pathlib import Path

from react_review.schemas.semantic import SemanticVerdict


class SemanticCacheMiss(KeyError):
    """Asked for a recorded judgement that is not in the cache."""


class SemanticCache:
    """Judgements keyed by the full question that produced them."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self._entries: dict[str, dict] = {}
        # Which model produced these judgements. Part of every key, so a replay
        # with no model configured can still reconstruct the keys it recorded —
        # otherwise a cache-only run misses everything it just wrote.
        self.model_id = ""
        self.hits = 0
        self.misses = 0
        if self.path and self.path.is_file():
            try:
                body = json.loads(self.path.read_text(encoding="utf-8-sig"))
                self.model_id = body.get("model_id", "")
                self._entries = body.get("entries", {})
            except Exception:                                     # noqa: BLE001
                self._entries = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get(self, key: str) -> SemanticVerdict | None:
        body = self._entries.get(key)
        if body is None:
            self.misses += 1
            return None
        self.hits += 1
        return SemanticVerdict(**body)

    def verdicts(self) -> list[SemanticVerdict]:
        """Every judgement in the recording — the input to a sensitivity check."""
        return [SemanticVerdict(**b) for b in self._entries.values()]

    def put(self, key: str, verdict: SemanticVerdict) -> None:
        self._entries[key] = verdict.model_dump(mode="json")
        self.model_id = self.model_id or str(verdict.provenance.get("model_id") or "")

    def save(self) -> Path | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"model_id": self.model_id, "entries": self._entries},
                       indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8")
        return self.path
