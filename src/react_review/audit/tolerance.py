"""The tolerance table: relative-error band per field_type (default 1%)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from react_review.schemas.audit import ToleranceRule


class ToleranceTable:
    """Resolve the comparison tolerance for a field_type.

    MVP is a single global band (``default_rel_tolerance``); the optional
    ``per_field_type`` map overrides specific concepts. Kept tiny and
    data-driven so calibrating tolerances (later, with clinical input) is a
    config change, not a code change.
    """

    def __init__(
        self,
        default_rel_tolerance: float = 0.01,
        per_field_type: dict[str, float] | None = None,
    ) -> None:
        self._default = float(default_rel_tolerance)
        self._per_field: dict[str, float] = {
            k: float(v) for k, v in (per_field_type or {}).items()
        }

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ToleranceTable":
        """Load a tolerance table from a YAML config file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            default_rel_tolerance=data.get("default_rel_tolerance", 0.01),
            per_field_type=data.get("per_field_type") or {},
        )

    def rel_tolerance(self, field_type: str) -> float:
        """Relative-error MATCH bound for ``field_type`` (falls back to default)."""
        return self._per_field.get((field_type or "").strip().lower(), self._default)

    def rule_for(self, field_type: str) -> ToleranceRule:
        """Return the resolved :class:`ToleranceRule` for ``field_type``."""
        key = (field_type or "").strip().lower()
        return ToleranceRule(
            field_type=key if key in self._per_field else "*",
            rel_tolerance=self.rel_tolerance(field_type),
            comparison_family="numeric",
        )
