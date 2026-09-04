"""Read a run's existing files. No new models — StepEvent JSON as stored."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def safe_run_id(run_id: str) -> str:
    if not _SAFE_ID.match(run_id or ""):
        raise ValueError("invalid run id")
    return run_id


def run_dir(runs_root: Path, run_id: str) -> Path:
    return Path(runs_root) / safe_run_id(run_id)


def list_run_ids(runs_root: Path) -> list[str]:
    root = Path(runs_root)
    if not root.is_dir():
        return []
    ids = [p.name for p in root.iterdir()
           if p.is_dir() and (p / "journal.ndjson").is_file()]
    return sorted(ids, reverse=True)


def journal_index(directory: Path) -> list[dict[str, Any]]:
    path = Path(directory) / "journal.ndjson"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_step(directory: Path, artifact: str) -> dict[str, Any]:
    path = Path(directory) / "steps" / artifact
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_steps(directory: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for row in journal_index(directory):
        artifact = str(row.get("artifact") or "")
        body = load_step(directory, artifact) if artifact else {}
        if not body:
            body = {
                "index": row.get("index"),
                "stage": row.get("stage"),
                "title": row.get("title"),
                "warnings": [],
                "render_blocks": [],
                "offers": [],
                "decision": "",
                "interaction": "",
            }
        steps.append(body)
    return steps


def load_live(directory: Path) -> dict[str, Any] | None:
    """Current in-flight step, if the reporter has written live.json."""
    path = Path(directory) / "live.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def run_status(directory: Path, *, waiting_index: int | None = None) -> str:
    d = Path(directory)
    if (d / "package.json").is_file():
        return "complete"
    if (d / "fail.json").is_file():
        return "failed"
    if waiting_index is not None:
        return "waiting"
    if (d / "package.partial.json").is_file():
        return "stopped"
    if journal_index(d):
        return "running"
    return "unknown"


def blocked_step(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The step the gate is on: written, but not yet decided."""
    for step in reversed(steps):
        if not step.get("decision") and step.get("blocking", True):
            return step
    return None
