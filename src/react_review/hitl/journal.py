"""The run journal — per-step artifacts on disk, written unconditionally.

This is the half of the advisor's requirement that does NOT depend on anyone
watching: even a ``--non-interactive`` run leaves, for every step, the file it
read and the full content it produced. It is what makes "the pipeline ran and
printed a good result" auditable after the fact, and it is what survives a
Ctrl-C at the prompt — :meth:`emit` writes before the gate is consulted.

    <run_dir>/journal.ndjson              one line per step, append-only
    <run_dir>/steps/003_collect_study.json  the full StepEvent
    <run_dir>/steps/003_collect_study.<name>   sidecars (rendered table, CSV…)
"""
from __future__ import annotations

import json
from pathlib import Path

from react_review.hitl.events import StepEvent


def _event_payload(event: StepEvent) -> dict:
    """Dump a step; omit unused backend-trace keys so unconfigured journals stay identical."""
    data = event.model_dump(mode="json")
    for key in ("backend_profile", "backend_model_id", "backend_reasoning"):
        if not data.get(key):
            data.pop(key, None)
    if data.get("backend_reasoning_tokens") is None:
        data.pop("backend_reasoning_tokens", None)
    return data


class RunJournal:
    """Append-only record of a run's steps."""

    def __init__(self, run_dir: Path | str) -> None:
        self._dir = Path(run_dir)
        self._steps = self._dir / "steps"
        self._ndjson = self._dir / "journal.ndjson"
        self._index = 0

    @property
    def run_dir(self) -> Path:
        return self._dir

    def next_index(self) -> int:
        self._index += 1
        return self._index

    def emit(self, event: StepEvent, *, sidecars: dict[str, str] | None = None) -> Path:
        """Persist ``event`` (and any sidecar files); returns the artifact path."""
        self._steps.mkdir(parents=True, exist_ok=True)
        path = self._steps / f"{event.slug}.json"
        path.write_text(
            json.dumps(_event_payload(event), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for name, body in (sidecars or {}).items():
            (self._steps / f"{event.slug}.{name}").write_text(body, encoding="utf-8")
        with self._ndjson.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "index": event.index, "stage": event.stage.value,
                "title": event.title, "subject": event.subject,
                "warnings": len(event.warnings), "artifact": path.name,
            }, ensure_ascii=False) + "\n")
        return path

    def record_decision(self, event: StepEvent) -> None:
        """Rewrite the step artifact once the gate's answer is known."""
        path = self._steps / f"{event.slug}.json"
        if path.is_file():
            path.write_text(
                json.dumps(_event_payload(event), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


class NullJournal:
    """No-op journal — the library default when no run directory is configured."""

    run_dir: Path | None = None

    def next_index(self) -> int:
        return 0

    def emit(self, event: StepEvent, *, sidecars: dict[str, str] | None = None) -> None:
        return None

    def record_decision(self, event: StepEvent) -> None:
        return None
