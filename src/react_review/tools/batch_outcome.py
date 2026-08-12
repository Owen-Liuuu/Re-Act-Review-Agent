"""What a projection means for the human who reads the audit.

The projection speaks in its own vocabulary — ok, derived, scope_unresolved,
contradictory — and the audit speaks in outcomes, which is what decides whether
a reviewer is told "the paper does not say this" or "we could not read it". The
two are not the same language and the translation is where an accusation can be
manufactured by accident: report a malformed response as MISSING_SOURCE and the
artifact says the review may have invented a number, on the strength of an
extractor fault.

So the mapping is written down once, here, in precedence order, rather than
being decided at each call site by whoever is wiring that path:

    1. released                     -> FOUND
    2. the batch itself failed      -> EXTRACTION_FAILED
    3. entries present, all refused -> EXTRACTION_FAILED
    4. a legitimate empty response  -> MISSING_SOURCE
    5. anything not uniquely read   -> EXTRACTION_UNRESOLVED

Order matters at both ends. `released` first, so a printed total that came with
an unrelated aggregation warning is not demoted for a fault in a part of the
response it never used. And MISSING_SOURCE last but one, reachable only when the
model said plainly that it looked and found nothing — because that is the only
state in which the audit is entitled to say the PAPER is silent.
"""
from __future__ import annotations

from react_review.core.enums import CollectionOutcome
from react_review.tools.batch_parse import BatchReading
from react_review.tools.batch_project import (
    BATCH_FAILED,
    DERIVED,
    OK,
    Projection,
)


def outcome_for(projection: Projection, reading: BatchReading) -> CollectionOutcome:
    """The one place a projection becomes something a reviewer is told."""
    if projection.status in (OK, DERIVED):
        return CollectionOutcome.FOUND
    if projection.status == BATCH_FAILED or reading.batch_error:
        return CollectionOutcome.EXTRACTION_FAILED
    if (reading.entries or reading.rejected) and not reading.usable:
        # The model returned readings and every one failed a deterministic
        # check. That is a fact about the RESPONSE. Calling it MISSING_SOURCE
        # would be the audit asserting, without evidence, that the paper is
        # silent — an accusation it has no grounds for.
        #
        # `rejected` as well as `entries`: a refused reading never reaches
        # `entries`, so testing that alone would leave the all-refused case —
        # the one this row exists for — falling through to a later branch.
        return CollectionOutcome.EXTRACTION_FAILED
    if _legitimately_empty(reading):
        return CollectionOutcome.MISSING_SOURCE
    return CollectionOutcome.EXTRACTION_UNRESOLVED


def _legitimately_empty(reading: BatchReading) -> bool:
    """The model looked, said so, and nothing it returned was refused.

    All three conditions. An empty list with no explanation is a response that
    failed to answer, not a paper that stays silent; and an empty list beside a
    rejected entry is a reading that broke, not a paper with nothing in it.
    """
    return (not reading.entries and not reading.rejected
            and bool(reading.nothing_reported_reason))
