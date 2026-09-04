"""Source retrieval implementations (local PDFs, composite, etc.)."""
from react_review.retrieval.composite import CompositeRetriever
from react_review.retrieval.labels import origin_label
from react_review.retrieval.local_pdf import LocalPdfRetriever

__all__ = ["CompositeRetriever", "LocalPdfRetriever", "origin_label"]
