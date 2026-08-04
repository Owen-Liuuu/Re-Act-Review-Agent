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
    # Which side says MORE. Asked for separately from ``relation`` because a
    # model that names the direction correctly in its rationale can still emit
    # the opposite label; two structured answers can be checked against each
    # other, a label and a sentence cannot.
    #
    # Empty means NOT STATED — a verdict recorded before this contract existed.
    # That is deliberately distinct from an explicit "unknown": a model that was
    # asked and could not tell has answered, and its answer must agree with the
    # relation it also gave.
    more_specific_side: str = ""   # "" | review | source | neither | unknown
    equivalent: bool = False
    confidence: float = 0.0
    rationale: str = ""
    review_normalized: str = ""
    source_normalized: str = ""
    # The span of the source quote the model says it relied on. Checked against
    # the quote, so a claim cannot rest on text the paper does not contain.
    evidence_span: str = ""
    provenance: dict = Field(default_factory=dict)
