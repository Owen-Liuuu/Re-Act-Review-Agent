"""Deterministic group (cohort) normalization: raw label → t1dm / control / all.

The review's own group labels are consistent within one review, so a small
synonym map covers the MVP. (An LLM/vocab fallback like field_type's, for
cross-domain group naming, is deferred.)
"""
from __future__ import annotations

_CONTROL = ("control", "healthy", "non-diabetic", "nondiabetic", "comparison")
_T1DM = ("t1dm", "type 1", "type-1", "dm", "diabet", "patient", "case")


def normalize_group(raw: str) -> str:
    """Map a raw cohort label to ``t1dm`` | ``control`` | ``all``.

    ``all`` is used when there is no diabetes/control split (a single reported
    cohort) or the label is blank/unrecognised.
    """
    g = (raw or "").strip().lower()
    if not g or g in ("all", "-", "combined", "total", "overall", "pooled"):
        return "all"
    # control before t1dm: "non-diabetic control" should be control, and
    # "diabetes" tokens must not steal a control label.
    if any(k in g for k in _CONTROL):
        return "control"
    if any(k in g for k in _T1DM):
        return "t1dm"
    return "all"
