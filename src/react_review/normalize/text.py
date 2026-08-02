"""Deterministic cleanup of text extracted from PDFs (Tier-1, universal).

Some PDFs mis-encode glyphs as C0 control characters — most importantly the
plus-minus sign ``±`` comes through as ``\\x01`` in certain fonts. Left in, that
raw control char (a) breaks the SD split in numeric parsing and (b) makes any
LLM JSON that quotes the value invalid (``Invalid control character``), which
silently turned whole source papers into "value not found".

Typesetting whitespace is folded for the same reason. A journal's text layer
often sets "12.90 ± 1.30" with THIN SPACEs (U+2009) or NARROW NO-BREAK SPACEs
around the sign. Those characters are invisible, so a value transcribed
faithfully from the PDF looks identical to the one a person typed — and compares
unequal to it. Folding them to an ordinary space is lossless for an audit and
removes a whole class of phantom mismatches.
"""
from __future__ import annotations

import re

# C0 control chars except tab / newline / carriage-return.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Unicode space separators that a PDF may use instead of U+0020: NBSP, the
# en/em/thin/hair family, figure & punctuation spaces, and the narrow NBSP.
_SPACES = re.compile(r"[   -   　]")


def clean_pdf_text(text: str) -> str:
    """Restore the ``±`` mis-encoding, fold exotic spaces, drop control chars."""
    if not text:
        return text
    # \x01 between numbers is, in this corpus, always a mangled ± ("mean ± SD").
    text = text.replace("\x01", "±")
    text = _SPACES.sub(" ", text)
    return _CTRL.sub("", text)
