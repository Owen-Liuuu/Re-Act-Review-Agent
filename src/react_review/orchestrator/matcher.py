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

from react_review.claim_ids import declared_claim_id
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
        audit_id=declared_claim_id(item),
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


def _identity_locator(item: ReviewDataItem | SourceEvidenceItem) -> tuple:
    """The key and source coordinate an explicit identity claims to name."""

    return (
        _key_of(item),
        getattr(item, "table_id", "") or "",
        getattr(item, "cell_ref", None),
        getattr(item, "checklist_id", "") or "",
    )


def _explicit_groups(items: list) -> dict[str, list]:
    groups: dict[str, list] = {}
    for item in items:
        claim_id = declared_claim_id(item)
        if claim_id:
            groups.setdefault(claim_id, []).append(item)
    return groups


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
    pairs: list[tuple[ReviewDataItem, SourceEvidenceItem]] = []
    claimed: set[int] = set()
    review_reasons: dict[int, UnmatchedClaim] = {}
    source_reasons: dict[int, UnmatchedClaim] = {}

    # Explicit identity is global, not scoped to a MatchKey bucket. A duplicate
    # anywhere invalidates every row carrying it on either side.
    review_ids = _explicit_groups(review_items)
    source_ids = _explicit_groups(source_items)
    duplicate_ids = {
        claim_id for claim_id in set(review_ids) | set(source_ids)
        if len(review_ids.get(claim_id, [])) > 1
        or len(source_ids.get(claim_id, [])) > 1
    }
    for claim_id in duplicate_ids:
        message = (
            f"claim id {claim_id!r} is not globally unique in this run; "
            "no row carrying it was paired")
        for item in review_ids.get(claim_id, []):
            review_reasons[id(item)] = _unmatched(
                item, "duplicate_claim_id", message)
        for item in source_ids.get(claim_id, []):
            source_reasons[id(item)] = _unmatched(
                item, "duplicate_claim_id", message)

    # New rows pair by explicit ID first, across the whole run. The ID does not
    # excuse a conflicting key or locator: that is a protocol conflict, not a
    # reason to fall through and guess by the old key.
    for review in review_items:
        claim_id = declared_claim_id(review)
        if not claim_id or claim_id in duplicate_ids:
            continue
        candidates = source_ids.get(claim_id, [])
        if not candidates:
            message = (
                f"no source evidence carries explicit claim id {claim_id!r}; "
                "legacy locator matching is disabled for an identified claim")
            review_reasons[id(review)] = _unmatched(
                review, "claim_identity_missing", message)
            for source in source_items:
                if (not declared_claim_id(source)
                        and _identity_locator(source) == _identity_locator(review)):
                    source_reasons[id(source)] = _unmatched(
                        source, "claim_identity_missing", message)
            continue
        source = candidates[0]
        if _identity_locator(review) != _identity_locator(source):
            message = (
                f"claim id {claim_id!r} names different match keys or locators "
                "on the review and source sides")
            review_reasons[id(review)] = _unmatched(
                review, "claim_identity_key_conflict", message)
            source_reasons[id(source)] = _unmatched(
                source, "claim_identity_key_conflict", message)
            continue
        claimed.add(id(source))
        pairs.append((review, source))

    # An identified source that no review claim names is not a legacy row and
    # may not be borrowed by a nearby key.
    for source in source_items:
        claim_id = declared_claim_id(source)
        if (not claim_id or claim_id in duplicate_ids or id(source) in claimed
                or id(source) in source_reasons):
            continue
        source_reasons[id(source)] = _unmatched(
            source, "claim_identity_missing",
            f"source evidence carries explicit claim id {claim_id!r}, but no "
            "review claim carries that identity")

    # Only rows with no explicit identity on either side use the historical
    # key/cell join. Mixed explicit/legacy rows never enter these buckets.
    legacy_review = [r for r in review_items if not declared_claim_id(r)]
    legacy_source = [s for s in source_items if not declared_claim_id(s)]
    review_by_key: dict[MatchKey, list[ReviewDataItem]] = {}
    for item in legacy_review:
        review_by_key.setdefault(_key_of(item), []).append(item)
    source_by_key: dict[MatchKey, list[SourceEvidenceItem]] = {}
    for item in legacy_source:
        source_by_key.setdefault(_key_of(item), []).append(item)

    for key, claims in review_by_key.items():
        bucket = source_by_key.get(key, [])
        unclaimed = [source for source in bucket if id(source) not in claimed]
        if len(claims) == 1 and len(bucket) <= 1:
            if unclaimed:
                claimed.add(id(unclaimed[0]))
                pairs.append((claims[0], unclaimed[0]))
            else:
                review_reasons[id(claims[0])] = _unmatched(
                    claims[0], "no_source_evidence",
                    "no source evidence was collected for this claim")
            continue

        by_cell = {cell_id(source): source for source in unclaimed
                   if cell_id(source)}
        for claim in claims:
            coordinate = cell_id(claim)
            match = by_cell.pop(coordinate, None) if coordinate else None
            if match is not None:
                claimed.add(id(match))
                pairs.append((claim, match))
            else:
                review_reasons[id(claim)] = _unmatched(
                    claim, "ambiguous_match_key",
                    f"{len(claims)} claims share the key "
                    f"{'/'.join(key)}; not paired because the source cell is "
                    "unknown or does not match — refusing to guess")

    unmatched_review = [
        review_reasons[id(item)] for item in review_items
        if id(item) in review_reasons
    ]
    unmatched_source = []
    for source in source_items:
        if id(source) in claimed:
            continue
        unmatched_source.append(source_reasons.get(id(source)) or _unmatched(
            source, "unclaimed_source",
            "a source value no review claim corresponds to"))
    return pairs, unmatched_review, unmatched_source
