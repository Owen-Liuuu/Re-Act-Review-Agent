"""Match-table construction: join review claims to source evidence.

Both sides are joined on the canonical 4-tuple
``(study_id, group, timepoint, field_type)`` (decision 3). In P1 the values are
already canonical (from the hand-labelled benchmark); in P2 the semantic
normaliser produces those keys before this join runs.
"""
from __future__ import annotations

from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem

MatchKey = tuple[str, str, str, str]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def match_key(study_id: str, group: str, timepoint: str, field_type: str) -> MatchKey:
    """The canonical join key for both sides."""
    return (_norm(study_id), _norm(group), _norm(timepoint), _norm(field_type))


def _review_key(item: ReviewDataItem) -> MatchKey:
    return match_key(item.study_id, item.group, item.timepoint, item.field_type)


def _source_key(item: SourceEvidenceItem) -> MatchKey:
    return match_key(item.study_id, item.group, item.timepoint, item.field_type)


def build_pairs(
    review_items: list[ReviewDataItem],
    source_items: list[SourceEvidenceItem],
) -> tuple[
    list[tuple[ReviewDataItem, SourceEvidenceItem]],
    list[ReviewDataItem],
    list[SourceEvidenceItem],
]:
    """Join review and source items on the 4-tuple key.

    Returns ``(pairs, unmatched_review, unmatched_source)``. Each source item is
    consumed at most once (first-come); review items with no source, and source
    items no review claimed, are returned separately so the report can flag them
    rather than silently dropping evidence.
    """
    source_by_key: dict[MatchKey, list[SourceEvidenceItem]] = {}
    for s in source_items:
        source_by_key.setdefault(_source_key(s), []).append(s)

    pairs: list[tuple[ReviewDataItem, SourceEvidenceItem]] = []
    unmatched_review: list[ReviewDataItem] = []
    consumed: set[int] = set()

    for r in review_items:
        bucket = source_by_key.get(_review_key(r), [])
        match = next((s for s in bucket if id(s) not in consumed), None)
        if match is not None:
            consumed.add(id(match))
            pairs.append((r, match))
        else:
            unmatched_review.append(r)

    unmatched_source = [s for s in source_items if id(s) not in consumed]
    return pairs, unmatched_review, unmatched_source
