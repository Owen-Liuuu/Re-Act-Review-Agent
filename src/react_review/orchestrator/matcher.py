"""Match-table construction: join review claims to source evidence.

Both sides join on the canonical 4-tuple
``(study_id, group, timepoint, field_type)`` (decision 3).

That key is not always unique. Two rows of the same study, cohort and field —
different arms that collapsed to one label, or the same measure at two
timepoints the parser could not tell apart — produce the same key, and pairing
them by arrival order silently attributes one row's evidence to another row's
claim. Nothing downstream can detect that: the verdict looks clean.

So a duplicated key is only paired when BOTH sides carry the same non-empty
``(table_id, cell_ref)`` — the cell the value actually came from. Otherwise the
whole group is refused and reported as ``ambiguous_match_key``. Refusing to
guess is a finding; guessing wrong is not.
"""
from __future__ import annotations

from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.schemas.report import UnmatchedClaim

MatchKey = tuple[str, str, str, str]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def match_key(study_id: str, group: str, timepoint: str, field_type: str) -> MatchKey:
    """The canonical join key for both sides."""
    return (_norm(study_id), _norm(group), _norm(timepoint), _norm(field_type))


def _key_of(item: ReviewDataItem | SourceEvidenceItem) -> MatchKey:
    return match_key(item.study_id, item.group, item.timepoint, item.field_type)


def cell_id(item: ReviewDataItem | SourceEvidenceItem) -> str:
    """The table cell and/or checklist assertion identity ("" if unknown).

    ``cell_ref`` alone is not enough — row 0, column 3 exists in every table.
    """
    table_id = getattr(item, "table_id", "") or ""
    cell = getattr(item, "cell_ref", None)
    checklist_id = getattr(item, "checklist_id", "") or ""
    parts: list[str] = []
    if table_id and cell is not None:
        parts.append(f"{table_id}#{cell[0]},{cell[1]}")
    if checklist_id:
        parts.append(f"checklist#{checklist_id}")
    return "|".join(parts)


def _unmatched(item, reason_code: str, message: str) -> UnmatchedClaim:
    return UnmatchedClaim(
        study_id=item.study_id, group=item.group, timepoint=item.timepoint,
        field_type=item.field_type, table_id=getattr(item, "table_id", "") or "",
        cell_ref=getattr(item, "cell_ref", None),
        checklist_id=getattr(item, "checklist_id", "") or "",
        reason_code=reason_code, message=message,
    )


def detect_duplicate_keys(items: list) -> dict[MatchKey, list[int]]:
    """Join keys held by more than one item → their positions."""
    seen: dict[MatchKey, list[int]] = {}
    for i, item in enumerate(items):
        seen.setdefault(_key_of(item), []).append(i)
    return {k: idx for k, idx in seen.items() if len(idx) > 1}


def build_pairs(
    review_items: list[ReviewDataItem],
    source_items: list[SourceEvidenceItem],
) -> tuple[
    list[tuple[ReviewDataItem, SourceEvidenceItem]],
    list[UnmatchedClaim],
    list[UnmatchedClaim],
]:
    """Join review and source items, refusing to guess on an ambiguous key.

    Returns ``(pairs, unmatched_review, unmatched_source)``; the unmatched
    entries carry WHY they were not paired.
    """
    review_by_key: dict[MatchKey, list[ReviewDataItem]] = {}
    for r in review_items:
        review_by_key.setdefault(_key_of(r), []).append(r)
    source_by_key: dict[MatchKey, list[SourceEvidenceItem]] = {}
    for s in source_items:
        source_by_key.setdefault(_key_of(s), []).append(s)

    pairs: list[tuple[ReviewDataItem, SourceEvidenceItem]] = []
    unmatched_review: list[UnmatchedClaim] = []
    claimed: set[int] = set()

    for key, claims in review_by_key.items():
        bucket = source_by_key.get(key, [])
        unclaimed = [s for s in bucket if id(s) not in claimed]

        # Unambiguous: one claim on this key. Pair with the one source value.
        if len(claims) == 1 and len(bucket) <= 1:
            if unclaimed:
                claimed.add(id(unclaimed[0]))
                pairs.append((claims[0], unclaimed[0]))
            else:
                unmatched_review.append(_unmatched(
                    claims[0], "no_source_evidence",
                    "no source evidence was collected for this claim"))
            continue

        # Ambiguous: the key alone cannot say which value answers which claim.
        # Pair only where both sides name the same cell; refuse the rest.
        by_cell = {cell_id(s): s for s in unclaimed if cell_id(s)}
        for claim in claims:
            cid = cell_id(claim)
            match = by_cell.pop(cid, None) if cid else None
            if match is not None:
                claimed.add(id(match))
                pairs.append((claim, match))
            else:
                unmatched_review.append(_unmatched(
                    claim, "ambiguous_match_key",
                    f"{len(claims)} claims share the key "
                    f"{'/'.join(key)}; not paired because the source cell is "
                    "unknown or does not match — refusing to guess"))

    unmatched_source = [
        _unmatched(s, "unclaimed_source",
                   "a source value no review claim corresponds to")
        for s in source_items if id(s) not in claimed
    ]
    return pairs, unmatched_review, unmatched_source
