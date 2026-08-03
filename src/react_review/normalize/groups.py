"""LEGACY cohort normalization. Not used by the pipeline — see ``cohorts.py``.

This mapped any label onto ``t1dm | control | all`` and, critically, returned
``all`` for anything it did not recognise (line below). On a review with
Treatment and Placebo arms both became ``all``, their claims collided on the
join key, and the audit paired one arm's value against the other's evidence
without any error or flag. The replacement discovers a review's cohorts from its
own table and marks an unplaceable label ``unknown`` instead of guessing.

Kept only so its behaviour stays documented and testable; ``tests/normalize/
test_groups.py`` pins what the old rule did. ``tests/normalize/test_cohorts.py``
asserts the parser no longer calls it. Do not use it in new code.
"""
from __future__ import annotations

_CONTROL = ("control", "healthy", "non-diabetic", "nondiabetic", "comparison")
_T1DM = ("t1dm", "type 1", "type-1", "dm", "diabet", "patient", "case")


def normalize_group(raw: str) -> str:
    """LEGACY: map a raw cohort label to ``t1dm`` | ``control`` | ``all``.

    .. deprecated::
        Use :func:`react_review.normalize.cohorts.build_cohort_registry` and
        :meth:`CohortRegistry.resolve`, which keep the review's own labels and
        report an unplaceable one rather than folding it into ``all``.
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
    return "all"          # ← the silent collapse this module was replaced for
