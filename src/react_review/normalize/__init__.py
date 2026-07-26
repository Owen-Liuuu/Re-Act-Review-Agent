"""Tier-1 (syntax) normalization: deterministic, domain-universal, no LLM.

Operates on VALUES and UNITS — the layer the tolerance compare and
unit_mismatch detection run on. The Tier-2 semantic layer (field_type / group
mapping via LLM + vocabulary) lives elsewhere and is a P2 concern; see
docs/normalization_pipeline.md.
"""
from react_review.normalize.numeric import NumericValue, parse_numeric, primary_number
from react_review.normalize.units import normalize_unit, units_differ

__all__ = [
    "NumericValue",
    "parse_numeric",
    "primary_number",
    "normalize_unit",
    "units_differ",
]
