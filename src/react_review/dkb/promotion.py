"""DKB promotion — provisional → authoritative (DKB-3).

Semi-automatic (the user-chosen policy): a provisional entry becomes authoritative
when a HUMAN confirms it, or when it has been independently produced K times with
a consistent field_type (REPEATED AGREEMENT). Deterministic; mutates the KB in
place. Pair it with a persisted counter store to survive across runs (future).
"""
from __future__ import annotations

import structlog

from react_review.dkb.base import KnowledgeBase

logger = structlog.get_logger(__name__)


class PromotionTracker:
    """Track provisional mappings and promote them per the semi-auto policy."""

    def __init__(self, kb: KnowledgeBase, *, threshold: int = 3) -> None:
        self._kb = kb
        self._threshold = max(1, threshold)
        self._counts: dict[str, int] = {}

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def observe(self, field_type: str) -> bool:
        """Record one consistent sighting; promote once it hits the threshold.

        Returns True iff this observation promoted the entry.
        """
        entry = self._kb.entries.get(field_type)
        if entry is None or entry.status != "provisional":
            return False
        self._counts[field_type] = self._counts.get(field_type, 0) + 1
        if self._counts[field_type] >= self._threshold:
            return self._promote(field_type, f"repeated agreement (x{self._counts[field_type]})")
        return False

    def confirm(self, field_type: str) -> bool:
        """A human confirms a provisional mapping → authoritative."""
        return self._promote(field_type, "human confirmed")

    def _promote(self, field_type: str, reason: str) -> bool:
        entry = self._kb.entries.get(field_type)
        if entry is None or entry.status != "provisional":
            return False
        entry.status = "authoritative"
        note = f"promoted ({reason})"
        entry.provenance.citation = f"{entry.provenance.citation} | {note}".strip(" |")
        logger.info("dkb_promote", field_type=field_type, reason=reason)
        return True
