"""PDF parser module: extracts StudentReviewInput from a systematic review PDF.

This is the "Step 0" before the main 4-step pipeline. It uses PyMuPDF to
extract text and an LLM to identify the search strategy, selected papers,
and data extraction tables.
"""
from react_review.steps.pdf_parsing.parser import PDFParser

__all__ = ["PDFParser"]
