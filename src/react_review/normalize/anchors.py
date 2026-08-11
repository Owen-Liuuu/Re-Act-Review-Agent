"""Shared text anchoring — does this quote really occur, and contain this value.

Extracted so the extraction tool and the deterministic target assignment apply
the SAME notion of "the paper says this". Two slightly different substring rules
in two places would mean a quote could anchor for one check and not the other,
and the disagreement would be invisible.

PDF text is the reason none of this is a plain ``in``: a printed line wrap
leaves ``pres-\\nent``, and the space around ``±`` is often U+2009.
"""
from __future__ import annotations

import re
import unicodedata


def normalise(text: str) -> str:
    """Lowercased alphanumeric words, with printed line-wrap hyphens joined."""
    text = unicodedata.normalize("NFKD", text or "")
    text = re.sub(r"(?<=[A-Za-z])[-­]\s*\n\s*(?=[A-Za-z])", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def alnum_compact(text: str) -> str:
    """Every separator dropped — the last resort for a hyphenated line break."""
    folded = unicodedata.normalize("NFKD", text or "").lower()
    return "".join(re.findall(r"[a-z0-9]+", folded))


def normalised_contains(haystack: str, needle: str) -> bool:
    """Whether ``needle`` occurs in ``haystack`` once both are normalised."""
    normal_needle = normalise(needle)
    if not normal_needle:
        return False
    if normal_needle in normalise(haystack):
        return True
    compact_needle = alnum_compact(needle)
    return bool(compact_needle) and compact_needle in alnum_compact(haystack)


def quote_contains_value(quote: str, value: str) -> bool:
    """Whether the quote actually prints the value it is offered as support for."""
    normal_value, normal_quote = normalise(value), normalise(quote)
    if normal_value and normal_value in normal_quote:
        return True
    compact_value = alnum_compact(value)
    return bool(compact_value) and compact_value in alnum_compact(quote)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def value_supported_by_quote(quote: str, value: str) -> bool:
    """Whether the quote supports this value — numbers strictly, wording not.

    A paper prints one arm's interval in full ("95% confidence interval [CI],
    4.3 to 9.5") and the next in short ("95% CI, 8.9 to 16.7"); a response that
    regularises the wording is reporting the same measurement. Demanding the
    exact bytes rejected complete, correct extractions for a parenthesis.

    What may NOT move is the arithmetic: every number in the value must appear
    in the quote, in the same order and with nothing else between them. So
    "11.5 (95% CI 8.9-16.7)" is supported by that sentence and "11.5 (95% CI
    2.8-3.4)" — every number of which also occurs somewhere in it — is not.
    """
    wanted = _NUMBER.findall(value or "")
    if not wanted:
        return quote_contains_value(quote, value)
    printed = _NUMBER.findall(flatten(quote))
    span = len(wanted)
    return any(
        all(_same_number(a, b) for a, b in zip(wanted, printed[start:start + span]))
        for start in range(0, max(0, len(printed) - span) + 1))


def _same_number(one: str, other: str) -> bool:
    try:
        return float(one) == float(other)
    except ValueError:
        return one == other


def flatten(text: str) -> str:
    """Whitespace collapsed and case folded, keeping every character's ORDER.

    Used where an offset matters (how far a number is from an arm label). Note
    that offsets are preserved in order but NOT in value: a run of whitespace
    becomes one space, so a flattened position is shorter than the raw one it
    came from. Anything that needs to look back at the original text must use
    :func:`flatten_with_offsets`, which says where each character came from.
    """
    return _fold(text)[0]


def flatten_with_offsets(text: str) -> tuple[str, list[int], str]:
    """Flattened text, where each character came from, and the folded source.

    The map exists because a flattened offset is not an offset into anything
    else. Using one to index the original document reads a window a few
    characters off — invisibly at first, and further out the longer the document
    gets, which is exactly the failure mode that hides in small fixtures and
    appears in real papers.

    Returns ``(flat, offsets, folded)`` where ``offsets[i]`` is the index in
    ``folded`` that ``flat[i]`` came from, and ``folded`` is the case-folded
    text with its line breaks intact — the only form in which block boundaries
    still exist.
    """
    return _fold(text)


def _fold(text: str) -> tuple[str, list[int], str]:
    folded = unicodedata.normalize("NFKD", text or "").lower()
    folded = re.sub(r"(?<=[a-z])[-­]\s*\n\s*(?=[a-z])", "-", folded)
    out: list[str] = []
    offsets: list[int] = []
    index, size = 0, len(folded)
    while index < size:
        if folded[index].isspace():
            run = index
            while run < size and folded[run].isspace():
                run += 1
            out.append(" ")
            offsets.append(index)
            index = run
        else:
            out.append(folded[index])
            offsets.append(index)
            index += 1
    return "".join(out), offsets, folded
