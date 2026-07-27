"""Match parser-produced study ids to the canonical included-studies registry.

The parser slugs a study name into e.g. ``yaz_2011`` / ``de_2018`` (imperfect for
non-ASCII / particle names). To audit against the source papers we must relabel
each review item to the canonical ``study_id`` from included_studies.csv
(``yazici_2011`` / ``de_gonzalo_calvo_2018``) and resolve its reference. Matching
is by publication year + surname prefix (year disambiguates same-surname-prefix).
"""
from __future__ import annotations

import re

from react_review.schemas.evidence import IncludedStudy, ReviewDataItem
from react_review.steps.paper_verification.schemas import ReferenceEntry


def _parts(study_id: str) -> tuple[str, str]:
    """(surname, year) from a study_id like 'yazici_2011' -> ('yazici', '2011')."""
    sid = (study_id or "").strip().lower()
    year_m = re.search(r"(19|20)\d{2}", sid)
    surname = sid.split("_")[0] if sid else ""
    return surname, (year_m.group(0) if year_m else "")


def resolve_study(parser_study_id: str, studies: list[IncludedStudy]) -> IncludedStudy | None:
    """Best canonical match for a parser study id, or None if ambiguous/none."""
    p_sur, p_year = _parts(parser_study_id)
    if not p_sur:
        return None
    candidates = []
    for s in studies:
        c_sur, c_year = _parts(s.study_id)
        if p_year and c_year and p_year != c_year:
            continue
        if c_sur == p_sur or c_sur.startswith(p_sur) or p_sur.startswith(c_sur):
            candidates.append(s)
    if len(candidates) == 1:
        return candidates[0]
    # Tie-break on an exact surname match if the year narrowed to several.
    exact = [s for s in candidates if _parts(s.study_id)[0] == p_sur]
    return exact[0] if len(exact) == 1 else None


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


def build_reference_resolver(sid_to_study: dict[str, IncludedStudy]):
    """A study_id -> ReferenceEntry resolver for the AuditPipeline."""
    def resolver(study_id: str) -> ReferenceEntry:
        s = sid_to_study.get(study_id)
        if s is None:
            return ReferenceEntry(title=study_id)
        return ReferenceEntry(title=s.review_citation or study_id, doi=s.doi or None)
    return resolver
