"""Compare a new eval artifact against an older one on the OLDER one's schema.

Once results carry new fields, "byte-identical to the previous run" stops being
a usable regression test: every added component makes it fail, so it would be
quietly relaxed until it meant nothing. What must hold instead is that nothing
the older artifact already recorded has moved.

So the new artifact is PROJECTED onto the old one's shape — recursively, because
a dict nested three levels down is where an added key hides — and only the keys
the old artifact actually had are compared. New keys are reported separately, as
information rather than as a failure.

Pure and importable; ``eval/compare_projection.py`` is the command line over it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project(old: Any, new: Any) -> Any:
    """``new`` restricted to the keys ``old`` has, at every depth."""
    if isinstance(old, dict) and isinstance(new, dict):
        return {k: project(v, new[k]) for k, v in old.items() if k in new}
    if isinstance(old, list) and isinstance(new, list):
        return [project(o, n) for o, n in zip(old, new)]
    return new


def added_keys(old: Any, new: Any, prefix: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        found = [f"{prefix}{k}" for k in new if k not in old]
        for k, v in old.items():
            if k in new:
                found += added_keys(v, new[k], prefix=f"{prefix}{k}.")
        return found
    if isinstance(old, list) and isinstance(new, list) and old and new:
        return added_keys(old[0], new[0], prefix=f"{prefix}[].")
    return []


def _differences(old: Any, new: Any, prefix: str = "") -> list[str]:
    projected = project(old, new)
    if projected == old:
        return []
    if isinstance(old, dict):
        out: list[str] = []
        for k, v in old.items():
            if k not in projected:
                out.append(f"{prefix}{k}: MISSING from the new artifact")
            else:
                out += _differences(v, projected[k], prefix=f"{prefix}{k}.")
        return out
    if isinstance(old, list):
        if len(old) != len(new):
            return [f"{prefix}: {len(old)} items vs {len(new)}"]
        out = []
        for i, (o, n) in enumerate(zip(old, projected)):
            out += _differences(o, n, prefix=f"{prefix.rstrip('.')}[{i}].")
        return out
    return [f"{prefix.rstrip('.')}: {old!r} -> {new!r}"]


def compare(old_path: Path, new_path: Path, *, key: str = "audit_id",
            ignore: tuple[str, ...] = ("run",)) -> dict[str, Any]:
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    for section in ignore:                 # paths and modes differ by design
        old.pop(section, None)
        new.pop(section, None)

    # Rows are matched by their identifier, never by position: a comparison that
    # silently pairs row 4 with row 5 reports nonsense with total confidence.
    for body in (old, new):
        rows = body.get("rows")
        if isinstance(rows, list) and rows and key in (rows[0] or {}):
            body["rows"] = sorted(rows, key=lambda r: str(r.get(key, "")))

    return {
        "old": str(old_path), "new": str(new_path),
        "differences": _differences(old, new),
        "added_fields": sorted(set(added_keys(old, new))),
    }
