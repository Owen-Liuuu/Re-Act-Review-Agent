"""Does this phrase describe THIS value, or merely occur in the same paper?

A population phrase that appears somewhere in a trial report proves nothing
about a number in a table three pages away. The model can hand back a genuine
sentence — "316 patients were assigned to the nivolumab group" — as the
population evidence for a number it read out of an analysis-set table, and both
halves are real text. Only their relationship is invented.

So a phrase binds to a value in one of two ways, and nothing else counts:

1. **Same quote.** The phrase and the value are inside one contiguous quote
   that has already been checked against the document. Nothing further is
   needed, because the paper itself put them together.

2. **Same block, close by.** The phrase and the value are both located in the
   document, in the same paragraph or table block, within a bounded distance.
   The bound is deliberately small: "same section" is not a relationship, it is
   a coincidence with a bigger radius.

Anything else — including both phrases being findable in the document — is
refused. The refusal is the point.
"""
from __future__ import annotations

import re

from react_review.normalize.anchors import flatten, normalised_contains

#: How far apart a phrase and its value may sit and still be one statement.
#: A sentence is tens of characters; a paragraph a few hundred. Beyond that the
#: claim that they belong together is doing the work, not the text.
MAX_BINDING_DISTANCE = 400

SAME_QUOTE = "same_quote"
SAME_BLOCK = "same_block"
UNBOUND = "unbound"

#: What separates one block from the next in flattened source text: a blank
#: line, or the excerpt marker the extractor inserts when it selects regions.
_BLOCK_BREAK = re.compile(r"\n\s*\n|\[SOURCE EXCERPT \d+:\d+\]")


def binding_verdict(phrase: str, value: str, *, quote: str,
                    document: str = "",
                    max_distance: int = MAX_BINDING_DISTANCE) -> tuple[str, str]:
    """How ``phrase`` is tied to ``value``. Returns ``(verdict, reason)``."""
    if not phrase:
        return UNBOUND, "no phrase was given, so nothing is bound to this value"
    if quote and normalised_contains(quote, phrase) and normalised_contains(quote, value):
        return SAME_QUOTE, ""

    if not document:
        return UNBOUND, (
            f"{phrase!r} is not inside the quote that supports {value!r}, and "
            "there is no document to place them in")

    flat = flatten(document)
    phrase_at = _positions(flat, phrase)
    value_at = _positions(flat, value if not quote else quote)
    if not phrase_at:
        return UNBOUND, f"{phrase!r} does not occur in the document"
    if not value_at:
        return UNBOUND, f"the evidence for {value!r} does not occur in the document"

    best = min(((abs(p - v), p, v) for p in phrase_at for v in value_at),
               key=lambda triple: triple[0])
    distance, phrase_pos, value_pos = best
    if distance > max_distance:
        return UNBOUND, (
            f"{phrase!r} is {distance} characters from the evidence for "
            f"{value!r}; occurring in the same paper is not a relationship")
    if _BLOCK_BREAK.search(document, *sorted((phrase_pos, value_pos))):
        return UNBOUND, (
            f"{phrase!r} and the evidence for {value!r} are in different blocks "
            "of the document")
    return SAME_BLOCK, ""


def bound(verdict: str) -> bool:
    return verdict in (SAME_QUOTE, SAME_BLOCK)


def _positions(flat: str, needle: str) -> list[int]:
    """Where a phrase occurs, compared on words rather than on bytes."""
    target = flatten(needle)
    if not target:
        return []
    out, start = [], 0
    while True:
        found = flat.find(target, start)
        if found < 0:
            return out
        out.append(found)
        start = found + 1
