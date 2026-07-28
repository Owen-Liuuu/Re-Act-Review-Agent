"""Tests for PDF text sanitization + JSON-parser robustness to control chars."""
from __future__ import annotations

from react_review.llm.base import parse_llm_response
from react_review.normalize.text import clean_pdf_text


def test_restores_plusminus_mis_encoding():
    # PyMuPDF renders ± as \x01 in some fonts (aslan_2015, iacobellis_2014).
    assert clean_pdf_text("Age 30.6 \x01 10.3 years") == "Age 30.6 ± 10.3 years"


def test_strips_other_control_chars_but_keeps_whitespace():
    assert clean_pdf_text("a\x00b\x07c") == "abc"
    assert clean_pdf_text("keep\ttab\nand\rreturn") == "keep\ttab\nand\rreturn"


def test_clean_pdf_text_empty():
    assert clean_pdf_text("") == ""


def test_parser_tolerates_control_char_in_string():
    # Defense in depth: even an un-sanitised control char must not hard-fail.
    raw = '```json\n{"found": true, "value": "30.6 \x01 10.3"}\n```'
    data = parse_llm_response(raw, "stub")
    assert data["found"] is True
