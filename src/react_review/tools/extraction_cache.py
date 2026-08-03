"""Record raw extraction responses and replay them without an LLM.

The recording stores the model JSON before deterministic post-processing, so a
replay re-exercises validation and arithmetic instead of returning a saved final
answer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ExtractionCacheMiss(KeyError):
    """A replay asked for an extraction response that was never recorded."""


def extraction_cache_key(*, model_id: str, prompt_version: str, prompt: str,
                         attempt: int) -> str:
    body = {
        "model_id": model_id,
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "attempt": attempt,
        "seed": 42,
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExtractionCache:
    """Raw model JSON keyed by the exact extraction question and attempt."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else None
        self.model_id = ""
        self._entries: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        if self.path and self.path.is_file():
            try:
                body = json.loads(self.path.read_text(encoding="utf-8-sig"))
                self.model_id = str(body.get("model_id") or "")
                self._entries = dict(body.get("entries") or {})
            except Exception:  # noqa: BLE001
                self._entries = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def get(self, key: str) -> dict[str, Any] | None:
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(json.dumps(value, ensure_ascii=False))

    def put(self, key: str, value: dict[str, Any], *, model_id: str) -> None:
        self._entries[key] = json.loads(json.dumps(value, ensure_ascii=False))
        self.model_id = self.model_id or model_id
        # A live extraction run is expensive. Persist each completed response so
        # an interruption does not discard all earlier calls.
        if self.path is not None:
            self.save()

    def save(self) -> Path | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "format": "react-review-extraction-replay-v1",
            "model_id": self.model_id,
            "entries": self._entries,
        }, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return self.path
