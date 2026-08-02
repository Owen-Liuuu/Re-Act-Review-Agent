"""Why something is missing, uncertain, or unusable — in words, not just a code.

An audit that reports "not found" without saying *why* leaves a reader unable to
tell an inaccessible paper from a fabricated value from a parser that never
looked. Where the reason comes from matters too, so it is recorded: a
deterministic check, a model explaining its own difficulty, or an exception.
"""
from __future__ import annotations

from pydantic import BaseModel


class ReasonRecord(BaseModel):
    """One explanation attached to an item."""

    code: str                       # placeholder_cell | cohort_unknown | …
    message: str = ""               # human-readable; may be written by the model
    source: str = "deterministic"   # deterministic | llm | exception
    stage: str = ""                 # which pipeline stage produced it
    detail: dict = {}

    def __str__(self) -> str:
        return self.message or self.code
