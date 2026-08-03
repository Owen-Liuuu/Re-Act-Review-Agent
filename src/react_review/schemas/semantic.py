"""A model's claim that two values mean the same thing — before it is believed.

Lives with the other schemas rather than beside the tool that produces it: the
deterministic controls in :mod:`react_review.audit.semantic_control` judge these
verdicts and must not depend on the tool layer to do so.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SemanticVerdict(BaseModel):
    """What the model said, and enough provenance to dispute it later."""

    relation: str = "unknown"      # same | review_broader | source_broader | different | unknown
    equivalent: bool = False
    confidence: float = 0.0
    rationale: str = ""
    review_normalized: str = ""
    source_normalized: str = ""
    # The span of the source quote the model says it relied on. Checked against
    # the quote, so a claim cannot rest on text the paper does not contain.
    evidence_span: str = ""
    provenance: dict = Field(default_factory=dict)
