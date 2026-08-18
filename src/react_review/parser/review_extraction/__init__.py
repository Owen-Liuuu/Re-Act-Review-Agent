"""Review Extraction — which review cells are source-paper claims.

A systematic review prints two kinds of number: values copied from an included
paper, and values the review computed (pooled OR, GRADE, forest-plot weights).
Only the first kind is this product's audit target.

``ReviewExtraction.run`` is the testable unit. ``ReviewParser.parse`` calls it
when the table-capture profile is not a frozen v1/v2 contract, then continues
with cohorts, field resolution, and the checklist.
"""
from __future__ import annotations

from react_review.parser.review_extraction.pipeline import ReviewExtraction
from react_review.parser.review_extraction.schemas import (
    DisplayHit,
    ExtractionResult,
    OriginLabel,
    ReviewClaim,
    ReviewLens,
)

__all__ = [
    "DisplayHit",
    "ExtractionResult",
    "OriginLabel",
    "ReviewClaim",
    "ReviewExtraction",
    "ReviewLens",
]
