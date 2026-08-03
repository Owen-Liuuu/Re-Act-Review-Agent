"""Auditable provenance for the knowledge loaded into one review run."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeConflictRecord(BaseModel):
    """One seed/ontology field conflict and the policy that resolved it."""

    field_type: str
    field: str
    seed_value: Any = None
    ontology_value: Any = None
    resolution: str = "ontology_override"
    reason: str = "curated ontology overrides the seed value"


class KnowledgeImportRecord(BaseModel):
    """What one ontology file changed in the runtime knowledge base."""

    source: str
    path: str
    sha256: str
    concepts_before: int = 0
    concepts_after: int = 0
    added: int = 0
    merged: int = 0
    added_field_types: list[str] = Field(default_factory=list)
    merged_field_types: list[str] = Field(default_factory=list)
    conflicts: list[KnowledgeConflictRecord] = Field(default_factory=list)
