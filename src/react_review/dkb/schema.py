"""DKB knowledge-entry schema (DKB-1).

A KnowledgeEntry generalises the old Tier-2 FieldTypeEntry: it keeps the same
core (field_type / synonyms / default_unit) and adds domain knowledge as DATA
instead of hardcoded rules — `scope` (study vs cohort, replaces the parser's
hardcoded _STUDY_LEVEL_FIELDS), `disambiguation` (one measurement method implying
one of several concepts, the A2 rule), unit equivalences, plausible ranges — plus
the `provenance` + `status` that DKB-2/3 need for provisional→authoritative
promotion.

Every domain-specific fact lives in the KB data files, not in this module. That
is the point: a new field of study ships a new knowledge base, not a code change.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Where a piece of knowledge came from (drives trust / promotion later)."""

    source: str = "curated"        # curated | ontology:<name> | learned | llm
    confidence: float = 1.0
    citation: str = ""


class KnowledgeEntry(BaseModel):
    """One canonical measurement concept."""

    field_type: str                                     # canonical key
    concept: str = ""                                   # human description
    value_type: str = "numeric"                         # numeric | text | categorical
    default_unit: str = ""
    synonyms: list[str] = Field(default_factory=list)
    # --- domain knowledge as data ---
    domain: str = ""                                    # e.g. "cardiology/imaging"
    scope: str = "cohort"                               # study | cohort
    unit_equivalences: list[str] = Field(default_factory=list)
    # discriminator -> {signal value -> field_type}, e.g.
    # {"modality": {"<method a>": "<concept measured that way>", ...}}
    disambiguation: dict[str, dict[str, str]] = Field(default_factory=dict)
    plausible_range: list[float] | None = None          # [lo, hi] sanity band
    # --- trust ---
    provenance: Provenance = Field(default_factory=Provenance)
    status: str = "authoritative"                       # authoritative | provisional

    def all_names(self) -> list[str]:
        """Every surface form this entry answers to."""
        return [self.field_type, self.concept, *self.synonyms]
