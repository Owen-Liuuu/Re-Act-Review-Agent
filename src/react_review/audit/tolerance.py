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
        default_sd_rel_tolerance: float = 0.03,
        per_field_type_sd: dict[str, float] | None = None,
        p_value_abs_tolerance: float = 0.0,
        null_value_by_field_type: dict[str, float] | None = None,
        semantic_min_confidence: float = 0.70,
    ) -> None:
        self._default = float(default_rel_tolerance)
        self._per_field: dict[str, float] = {
            k.strip().lower(): float(v) for k, v in (per_field_type or {}).items()
        }
        self._default_sd = float(default_sd_rel_tolerance)
        self._per_field_sd: dict[str, float] = {
            k.strip().lower(): float(v) for k, v in (per_field_type_sd or {}).items()
        }
        # A p-value needs an ABSOLUTE band: 0.001 vs 0.002 is a 100% relative
        # error but a 0.001 absolute one, and only the second reading matches how
        # the number is used.
        self._p_abs = float(p_value_abs_tolerance)
        # The value a confidence interval must exclude for the finding to be
        # "significant" — 1.0 for a ratio, 0.0 for a difference. Configured, not
        # inferred: which measures are ratios is domain knowledge, not code's.
        self._null: dict[str, float] = {
            k.strip().lower(): float(v)
            for k, v in (null_value_by_field_type or {}).items()
        }
        # Provisional; recorded on every run so a result states the band it used.
        self.semantic_min_confidence = float(semantic_min_confidence)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ToleranceTable":
        """Load a tolerance table from a YAML config file."""
        path = Path(path)
        with open(path, encoding="utf-8-sig") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            default_rel_tolerance=data.get("default_rel_tolerance", 0.01),
            per_field_type=data.get("per_field_type") or {},
            default_sd_rel_tolerance=data.get("default_sd_rel_tolerance", 0.03),
            per_field_type_sd=data.get("per_field_type_sd") or {},
            p_value_abs_tolerance=data.get("p_value_abs_tolerance", 0.0),
            null_value_by_field_type=data.get("null_value_by_field_type") or {},
            semantic_min_confidence=data.get("semantic_min_confidence", 0.70),
        )

    def rel_tolerance(self, field_type: str) -> float:
        """Relative-error MATCH bound for the MEAN (falls back to default)."""
        return self._per_field.get((field_type or "").strip().lower(), self._default)

    def sd_rel_tolerance(self, field_type: str) -> float:
        """Relative-error MATCH bound for the SD (falls back to default)."""
        return self._per_field_sd.get(
            (field_type or "").strip().lower(), self._default_sd
        )

    def p_value_abs_tolerance(self, field_type: str = "") -> float:
        """Absolute band for a reported bound / p-value threshold."""
        del field_type
        return self._p_abs

    def null_value(self, field_type: str) -> float | None:
        """The no-effect value for this measure, or None when not configured."""
        return self._null.get((field_type or "").strip().lower())

    def rule_for(self, field_type: str) -> ToleranceRule:
        """Return the resolved :class:`ToleranceRule` for ``field_type``."""
        key = (field_type or "").strip().lower()
        return ToleranceRule(
            field_type=key if key in self._per_field else "*",
            rel_tolerance=self.rel_tolerance(field_type),
            comparison_family="numeric",
        )
