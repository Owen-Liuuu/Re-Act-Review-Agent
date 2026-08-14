"""Step 3: Data extraction and table generation."""

from react_review.steps.data_extraction.interfaces import Extractor
from react_review.steps.data_extraction.schemas import (
    DocumentScope,
    ExtractedField,
    ExtractedTable,
    PaperDocument,
)

__all__ = [
    "DocumentScope",
    "Extractor",
    "ExtractedField",
    "ExtractedTable",
    "PaperDocument",
]
