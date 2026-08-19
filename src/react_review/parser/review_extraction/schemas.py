"""Structured objects the Review Extraction unit passes between steps.

Later steps receive a ``ReviewLens``, never the raw abstract. Origin labels are
the only thing that decides whether a captured cell becomes an audit claim.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from react_review.schemas.evidence import ReviewDataItem
from react_review.schemas.table import CapturedTableSet

ValueSource = Literal["source_paper", "review_computed", "bibliographic"]
DisplayKind = Literal["pdf_table", "forest_plot", "other"]


class ReviewClaim(ReviewDataItem):
    """Origin-labelled review cell.

    Subclass rather than editing ``ReviewDataItem``: that class is inside the
    frozen aggregation / evidence-adequacy evaluator trees.
    """

    value_source: str = ""
    outcome: str = ""
    display_kind: str = ""
    study_label_raw: str = ""


class ReviewLens(BaseModel):
    """A compressed ruler for the rest of extraction — not a second abstract."""

    lens_one_line: str = ""
    domain: str = ""
    population: str = ""
    comparison: str = ""
    outcomes: list[str] = Field(default_factory=list)
    not_audit_focus: list[str] = Field(default_factory=list)
    difficulties: list[str] = Field(default_factory=list)

    def as_ruler(self) -> str:
        """The compact form later prompts are allowed to see."""
        outcomes = "; ".join(self.outcomes) if self.outcomes else ""
        parts = [
            f"one line: {self.lens_one_line}" if self.lens_one_line else "",
            f"domain: {self.domain}" if self.domain else "",
            f"population: {self.population}" if self.population else "",
            f"comparison: {self.comparison}" if self.comparison else "",
            f"outcomes: {outcomes}" if outcomes else "",
            ("not audit focus: " + "; ".join(self.not_audit_focus)
             if self.not_audit_focus else ""),
        ]
        return "\n".join(p for p in parts if p)


class DisplayHit(BaseModel):
    """One table or figure the localizer considered."""

    display_id: str
    kind: DisplayKind = "other"
    caption: str = ""
    page_hint: str = ""
    evidence_chain: bool = False
    reason: str = ""


class OriginLabel(BaseModel):
    """Whether one column (or one cell) is a source-paper claim."""

    table_id: str
    column_path: str = ""
    row: int | None = None
    value_source: ValueSource = "source_paper"
    outcome: str = ""
    reason: str = ""


class ExtractionResult(BaseModel):
    """What Review Extraction hands back to the parser tail."""

    lens: ReviewLens = Field(default_factory=ReviewLens)
    hits: list[DisplayHit] = Field(default_factory=list)
    tables: CapturedTableSet = Field(default_factory=CapturedTableSet)
    origin_labels: list[OriginLabel] = Field(default_factory=list)
    research_context: str = ""
    dropped_non_source: list[str] = Field(default_factory=list)
