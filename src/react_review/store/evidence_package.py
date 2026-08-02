"""JSON-file Evidence Package Store: one directory per run.

    <base_dir>/<run_id>/package.json

Chosen over a database (decision 6) because a run's evidence is small, the JSON
is human-inspectable, and it doubles as a recorded fixture for tests. Writes are
atomic (temp file + replace) so a crash mid-write can't corrupt a package.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from react_review.schemas.package import EvidencePackage


class EvidencePackageStore:
    """Persist and load :class:`EvidencePackage` objects under a base directory."""

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir)

    def run_dir(self, run_id: str) -> Path:
        return self._base / run_id

    def _package_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "package.json"

    def exists(self, run_id: str) -> bool:
        return self._package_path(run_id).is_file()

    def list_runs(self) -> list[str]:
        """All run_ids that have a stored package, sorted."""
        if not self._base.is_dir():
            return []
        return sorted(
            p.name for p in self._base.iterdir()
            if p.is_dir() and (p / "package.json").is_file()
        )

    def save(self, package: EvidencePackage) -> Path:
        """Write ``package`` atomically; returns the package.json path."""
        return self._write(package, self._package_path(package.run_id))

    def save_partial(self, package: EvidencePackage) -> Path:
        """Write progress so far to ``package.partial.json``.

        Called after each study group so an interrupted run still leaves the
        evidence it had already collected, instead of losing the whole run.
        """
        return self._write(package, self.run_dir(package.run_id) / "package.partial.json")

    def _write(self, package: EvidencePackage, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(
            package.model_dump(mode="json"), indent=2, ensure_ascii=False
        )
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)  # atomic on the same filesystem
        return path

    def load(self, run_id: str) -> EvidencePackage:
        """Load the package for ``run_id`` (FileNotFoundError if absent)."""
        path = self._package_path(run_id)
        if not path.is_file():
            raise FileNotFoundError(f"no evidence package for run {run_id!r} at {path}")
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return EvidencePackage.model_validate(data)
