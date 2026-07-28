"""Deterministic cleanup of text extracted from PDFs (Tier-1, universal).

Some PDFs mis-encode glyphs as C0 control characters — most importantly the
plus-minus sign ``±`` comes through as ``\\x01`` in certain fonts. Left in, that
raw control char (a) breaks the SD split in numeric parsing and (b) makes any
LLM JSON that quotes the value invalid (``Invalid control character``), which
silently turned whole source papers into "value not found".
"""
from __future__ import annotations

import re

# C0 control chars except tab / newline / carriage-return.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_pdf_text(text: str) -> str:
    """Restore the common ``±`` mis-encoding and drop other control chars."""
    if not text:
        return text
    # \x01 between numbers is, in this corpus, always a mangled ± ("mean ± SD").
    text = text.replace("\x01", "±")
    return _CTRL.sub("", text)
