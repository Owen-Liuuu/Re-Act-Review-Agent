"""Deterministic audit core: tolerance table + value comparison (Tier-3).

This is the piece the benchmark validates directly — feeding review/source
values through :func:`compare_values` must reproduce the hand-labelled
``expected_label`` on ``audit_template.csv`` and catch the seeded discrepancies.
"""
from react_review.audit.tolerance import ToleranceTable
from react_review.audit.compare import compare_values

__all__ = ["ToleranceTable", "compare_values"]
