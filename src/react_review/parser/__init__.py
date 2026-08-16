"""Deterministic Parser Module: review PDF → review long table (ReviewDataItem[]).

Two-stage LLM extraction (Stage 1 identify table structure, Stage 2 unpivot to
long rows) followed by deterministic normalization (field_type via the DKB
FieldResolver, cohorts via the review's own labels — see
``react_review.normalize.cohorts``). See docs/normalization_pipeline.md.
"""
from react_review.parser.review_parser import ParsedReview, ReviewParser

__all__ = ["ReviewParser", "ParsedReview"]
