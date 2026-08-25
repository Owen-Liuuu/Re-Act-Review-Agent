"""Swallowed forest-OCR failures leave a warning; return values stay empty."""
from __future__ import annotations

import builtins
import sys

import pytest

from react_review.observe import records, reset
from react_review.tools import forest_ocr as fo


MISSING = "Z:/no-such-react-review-forest.pdf"


def _block_fitz(monkeypatch):
    real = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fitz" or str(name).startswith("fitz."):
            raise ImportError("No module named 'fitz'")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)


def _events(*names: str) -> set[str]:
    return {str(r["event"]) for r in records() if r["event"] in names}


def test_figure_text_open_failure_returns_empty_and_warns():
    reset()
    assert fo._figure_text(MISSING, "Figure 2", "") == ""
    assert "forest_ocr_pdf_open_failed" in _events("forest_ocr_pdf_open_failed")


def test_locate_figure_images_open_failure_returns_empty_and_warns():
    reset()
    assert fo._locate_figure_images(MISSING) == []
    assert "forest_ocr_pdf_open_failed" in _events("forest_ocr_pdf_open_failed")


def test_paired_forest_images_open_failure_returns_empty_and_warns():
    reset()
    assert fo._paired_forest_images(MISSING) == []
    assert "forest_ocr_pdf_open_failed" in _events("forest_ocr_pdf_open_failed")


def test_read_png_open_failure_returns_empty_and_warns():
    reset()
    assert fo._read_png(MISSING, 1) == b""
    assert "forest_ocr_pdf_open_failed" in _events("forest_ocr_pdf_open_failed")


def test_image_rects_failure_returns_empty_and_warns():
    reset()

    class Page:
        def get_images(self, full=True):
            return [(1, 0, 400, 400)]

        def get_image_rects(self, xref):
            raise RuntimeError("rects boom")

    assert fo._locate_in_doc([Page()]) == []
    blob = " ".join(str(r["message"]) for r in records())
    assert "rects boom" in blob
    assert "forest_ocr_image_rects_failed" in _events("forest_ocr_image_rects_failed")


def test_clip_nearby_failure_returns_empty_and_warns():
    reset()

    class Rect:
        width = 100
        height = 100

    class Page:
        rect = Rect()

        def get_text(self, *a, **k):
            raise RuntimeError("clip boom")

    assert fo._clip_nearby(Page(), 10.0, 20.0) == ""
    assert "forest_ocr_clip_failed" in _events("forest_ocr_clip_failed")


def test_read_png_pixmap_failure_returns_empty_and_warns(monkeypatch):
    reset()

    class Doc:
        def close(self):
            pass

    class FakeFitz:
        csRGB = object()

        @staticmethod
        def open(path):
            return Doc()

        class Pixmap:
            def __init__(self, *a, **k):
                raise RuntimeError("pixmap boom")

    monkeypatch.setitem(sys.modules, "fitz", FakeFitz)
    assert fo._read_png("any.pdf", 9) == b""
    assert "forest_ocr_png_failed" in _events("forest_ocr_png_failed")


@pytest.mark.parametrize("fn, empty", [
    (lambda: fo._figure_text("x.pdf", "c", ""), ""),
    (lambda: fo._locate_figure_images("x.pdf"), []),
    (lambda: fo._paired_forest_images("x.pdf"), []),
    (lambda: fo._read_png("x.pdf", 1), b""),
])
def test_pymupdf_missing_returns_empty_and_warns_once_per_site(monkeypatch, fn, empty):
    reset()
    _block_fitz(monkeypatch)
    assert fn() == empty
    assert "forest_ocr_dependency_missing" in _events("forest_ocr_dependency_missing")
    blob = " ".join(str(r["message"]) for r in records())
    assert "PyMuPDF" in blob
