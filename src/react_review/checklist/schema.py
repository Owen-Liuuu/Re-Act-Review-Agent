"""Typed contract for a clinician-editable audit checklist."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

ChecklistScope = Literal["review", "per_study", "per_cohort"]
ChecklistValueKind = Literal["presence", "numeric", "categorical", "text"]
ChecklistWhere = Literal["review_text", "review_table", "source_paper"]
ChecklistComponent = Literal["operator", "ci", "events", "pct", "sd"]


class ChecklistItem(BaseModel):
    """One question whose coverage must be visible before auditing continues."""

    id: str
    question: str
    required: bool = False
    scope: ChecklistScope = "review"
    where: list[ChecklistWhere] = Field(default_factory=lambda: ["review_table"])
    value_kind: ChecklistValueKind = "presence"
    # A concrete ReviewDataItem satisfies the question when either its canonical
    # type is listed here or its verbatim header matches one of the aliases.
    field_types: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    # Optional structured numeric parts recognised by normalize.numeric, e.g.
    # ``ci`` or ``events``. Every listed component must be present in the cell.
    required_components: list[ChecklistComponent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_selector(self) -> "ChecklistItem":
        if not self.id.strip():
            raise ValueError("checklist item id cannot be empty")
        if not self.question.strip():
            raise ValueError(f"checklist item {self.id!r} needs a question")
        if self.value_kind == "presence" and not (self.aliases or self.field_types):
            raise ValueError(
                f"presence item {self.id!r} needs aliases or field_types to search for")
        return self


class Checklist(BaseModel):
    """A versioned checklist plus the exact YAML artifact it came from."""

    name: str = "default"
    version: str = "1"
    source_file: str = ""
    sha256: str = ""
    items: list[ChecklistItem] = Field(default_factory=list)

    def require_supported_execution_contract(self) -> None:
        """Reject locations the current executor cannot actually inspect.

        ``source_paper`` remains in the serialized Literal so historical
        evidence packages stay readable.  New YAML and every live execution
        fail loudly instead of silently ignoring that location.
        """
        unsupported = [item.id for item in self.items if "source_paper" in item.where]
        if unsupported:
            raise ValueError(
                "checklist where=source_paper is not supported by the current "
                "review-side checklist executor; affected item(s): "
                + ", ".join(unsupported))

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Checklist":
        source = Path(path).resolve()
        raw = source.read_bytes()
        body = yaml.safe_load(raw.decode("utf-8-sig")) or {}
        payload = {key: value for key, value in body.items()
                   if key not in {"source_file", "sha256"}}
        checklist = cls(
            **payload,
            source_file=str(source),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        ids = [item.id for item in checklist.items]
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate checklist item id(s): {', '.join(duplicates)}")
        checklist.require_supported_execution_contract()
        return checklist


class ChecklistEvidence(BaseModel):
    """Why one checklist question is considered covered."""

    source: str                         # review_item | review_text | captured_table
    checklist_id: str = ""
    study_id: str = ""
    group: str = "-"
    field_type: str = ""
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    excerpt: str = ""


class ChecklistAssessment(BaseModel):
    """Coverage result for one checklist item."""

    checklist_id: str
    question: str
    required: bool = False
    scope: ChecklistScope = "review"
    value_kind: ChecklistValueKind = "presence"
    status: str                         # covered | partial | missing_required | missing_optional
    expected: int = 1
    found: int = 0
    evidence: list[ChecklistEvidence] = Field(default_factory=list)
    reason: str = ""
    evaluation_pass: str = ""


class ChecklistGap(BaseModel):
    """One required scope target that the review did not cover."""

    checklist_id: str
    question: str
    scope: ChecklistScope
    study_id: str = ""
    group: str = "-"
    reason: str


class ChecklistApplication(BaseModel):
    """Run-level checklist artifact persisted with the evidence package."""

    name: str
    version: str
    source_file: str
    sha256: str
    items: list[ChecklistItem] = Field(default_factory=list)
    assessments: list[ChecklistAssessment] = Field(default_factory=list)
    gaps: list[ChecklistGap] = Field(default_factory=list)
    completed_passes: list[str] = Field(default_factory=list)
