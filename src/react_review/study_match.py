"""Match a table row's identity to a cited paper.

The table prints its own words (``Li J et al.`` plus a year). The reference
list and ``included_studies.csv`` usually store a compact alias (``li_2015``,
``ahmad_2022``). Pairing is year-aware matching against the alias AND the
printed citation, so two papers that share a first word stay distinct and a
review from another field does not need a special slugger. Zero or two hits
is a refusal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from react_review.normalize.study_key import best_identity_match
from react_review.schemas.evidence import IncludedStudy, ReviewDataItem
from react_review.schemas.reason import ReasonRecord
from react_review.schemas.resolution import FieldResolutionRecord
from react_review.steps.paper_verification.schemas import ReferenceEntry

if TYPE_CHECKING:
    from react_review.parser.review_parser import ParsedStudy


# Marks a reference the review's list did not actually supply, so downstream can
# tell "no citation for this study" from "a citation we have yet to resolve".
_UNRESOLVED_PREFIX = "[no citation for] "


def resolve_study(parser_study_id: str, studies: list[IncludedStudy]) -> IncludedStudy | None:
    """Best canonical match for a parser study id, or None if ambiguous/none."""
    by_id = {s.study_id: s for s in studies}
    best = best_identity_match(
        parser_study_id, [(s.study_id, s.review_citation) for s in studies])
    return by_id.get(best) if best else None


def resolve_studies(
    review_items: list[ReviewDataItem],
    studies: list[IncludedStudy],
) -> tuple[list[ReviewDataItem], dict[str, IncludedStudy]]:
    """Relabel review items to canonical study_ids; return (items, canonical->study).

    Unresolved items keep their original study_id (they'll surface as unmatched).
    """
    cache: dict[str, IncludedStudy | None] = {}
    sid_to_study: dict[str, IncludedStudy] = {}
    relabelled: list[ReviewDataItem] = []
    for item in review_items:
        if item.study_id not in cache:
            cache[item.study_id] = resolve_study(item.study_id, studies)
        match = cache[item.study_id]
        if match is not None:
            sid_to_study[match.study_id] = match
            relabelled.append(item.model_copy(update={"study_id": match.study_id}))
        else:
            relabelled.append(item)
    return relabelled, sid_to_study


def is_resolvable(reference: ReferenceEntry) -> bool:
    """Whether this reference is worth trying to fetch.

    A DOI is not required — a real citation with authors and a year can still be
    reconciled online through the gated resolver, and refusing those would throw
    away most of a review whose reference list prints no DOIs. What cannot work
    is a PLACEHOLDER: an id standing in for a citation the parser never found,
    where a title search has nothing to search for.
    """
    if (reference.doi or "").strip():
        return True
    title = (reference.title or "").strip()
    return bool(title) and not title.startswith(_UNRESOLVED_PREFIX)


def build_reference_resolver_from_parsed(studies: "list[ParsedStudy]"):
    """A study_id -> ReferenceEntry resolver built from the PARSER's own reference
    list (no included_studies.csv). The citation becomes the title and any printed
    DOI is carried; a missing DOI is later reconciled online by the Collector.
    """
    by_id = {s.study_id: s for s in studies}

    def resolver(study_id: str) -> ReferenceEntry:
        # exact id first (fast path), then fuzzy year+surname.
        s = by_id.get(study_id)
        if s is None:
            best = best_identity_match(
                study_id, [(p.study_id, p.citation) for p in studies])
            s = by_id.get(best) if best else None
        if s is None:
            # No citation for this study. Returning a bare ReferenceEntry whose
            # title is the study id looks like a real reference and sends the
            # resolver hunting for a paper called "ahmad_2022"; mark it instead.
            return ReferenceEntry(title=f"{_UNRESOLVED_PREFIX}{study_id}")
        return ReferenceEntry(
            title=s.citation or study_id,
            doi=(s.doi or None),
            pmid=(s.pmid or None),
        )
    return resolver


def build_reference_resolver(sid_to_study: dict[str, IncludedStudy]):
    """A study_id -> ReferenceEntry resolver for the AuditPipeline."""
    def resolver(study_id: str) -> ReferenceEntry:
        s = sid_to_study.get(study_id)
        if s is None:
            return ReferenceEntry(title=study_id)
        return ReferenceEntry(title=s.review_citation or study_id, doi=s.doi or None)
    return resolver


def apply_modality_disambiguation(
    review_items,
    sid_to_study,
    kb,
    field_resolutions: list[FieldResolutionRecord] | None = None,
):
    """Relabel a field_type using the study's modality + the DKB disambiguation rule.

    A header that names the quantity but not how it was measured is ambiguous at
    parse time, because the parser does not yet know the study's modality. Now
    that studies are resolved to included_studies, use each study's modality
    against the concept's ``disambiguation`` rule — the A2 fix, driven by KB DATA
    rather than by anything hardcoded here.
    """
    out = []
    resolutions = {r.resolution_key: r for r in (field_resolutions or [])}
    changed_keys: set[str] = set()
    for item in review_items:
        study = sid_to_study.get(item.study_id)
        entry = kb.entries.get(item.field_type)
        rule = entry.disambiguation.get("modality", {}) if entry else {}
        if study and study.modality and rule:
            m = study.modality.strip().lower()
            for signal, target in rule.items():
                if signal in m and target in kb.entries and target != item.field_type:
                    previous = item.field_type
                    item = item.model_copy(update={"field_type": target})
                    record = resolutions.get(item.resolution_key)
                    if record is not None:
                        changed_keys.add(record.resolution_key)
                        record.checks["study_modality_disambiguation"] = True
                        record.field_types_seen = list(dict.fromkeys(
                            [*record.field_types_seen, previous, target]))
                        reason = ReasonRecord(
                            code="modality_disambiguation", stage="field_resolution",
                            message=(f"study modality {study.modality!r} changed "
                                     f"{previous} to {target}"),
                            detail={"study_id": item.study_id, "from": previous,
                                    "to": target, "modality": study.modality},
                        )
                        known = {(r.code, r.message) for r in record.reasons}
                        if (reason.code, reason.message) not in known:
                            record.reasons.append(reason)
                        for cell in record.affected_cells:
                            same_cell = (
                                bool(item.table_id) and item.cell_ref is not None
                                and cell.table_id == item.table_id
                                and cell.cell_ref == item.cell_ref)
                            if same_cell:
                                cell.field_type = target
                                cell.status = "authoritative"
                                cell.reason = reason.message
                    break
        out.append(item)

    # Keep the run-level decision truthful after this downstream deterministic
    # refinement.  One record can cover several studies: if their modalities
    # diverge, the per-cell decisions remain explicit and the aggregate is mixed.
    for key in changed_keys:
        record = resolutions[key]
        cell_types = {c.field_type for c in record.affected_cells if c.field_type}
        if len(cell_types) == 1:
            record.field_type = next(iter(cell_types))
            record.status = "authoritative"
            record.source = "deterministic_modality"
        else:
            record.field_type = None
            record.status = "mixed"
            record.source = "mixed"
    return out
