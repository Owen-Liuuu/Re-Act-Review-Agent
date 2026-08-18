"""Paper documents and extracted-table models.

Shared by the audit pipeline, the retrieval chain and the evidence schemas: a
``PaperDocument`` plus how much of the paper it represents (``DocumentScope``).
"""

from react_review.steps.data_extraction.schemas import (
    DocumentScope,
    ExtractedField,
    ExtractedTable,
    PaperDocument,
)

__all__ = [
    "DocumentScope",
    "ExtractedField",
    "ExtractedTable",
    "PaperDocument",
]
