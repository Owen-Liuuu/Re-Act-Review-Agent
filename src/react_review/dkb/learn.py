"""DKB developer LEARN module (DKB-4) — separate from the audit product.

The audit runs read-only and only COLLECTS candidate concepts as proposals
(``FieldResolver.proposals``). This module is the deliberate, developer-side
curation loop that turns those proposals into trusted knowledge:

    ingest(run proposals)  →  add as provisional + record ONE agreement per concept
    K agreements (repeated across runs) OR confirm(field_type)  →  authoritative
    save()  →  a new curated KB version

Kept out of the client audit path on purpose (reproducibility + trust): a
self-added mapping is only promoted after repeated agreement or a human confirm.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

from react_review.dkb.base import KnowledgeBase
from react_review.dkb.promotion import PromotionTracker
from react_review.dkb.schema import KnowledgeEntry

logger = structlog.get_logger(__name__)


def load_proposals(path: Path | str) -> list[KnowledgeEntry]:
    """Load a proposals JSON (a list of KnowledgeEntry bodies) saved by a run."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return [KnowledgeEntry(**body) for body in data]


def save_proposals(proposals: list[KnowledgeEntry], path: Path | str) -> None:
    Path(path).write_text(
        json.dumps([p.model_dump() for p in proposals], ensure_ascii=False, indent=2),
        encoding="utf-8")


class LearningSession:
    """Curate audit proposals into KB knowledge (developer mode)."""

    def __init__(self, kb: KnowledgeBase, *, threshold: int = 3) -> None:
        self._kb = kb
        self._tracker = PromotionTracker(kb, threshold=threshold)

    @property
    def kb(self) -> KnowledgeBase:
        return self._kb

    def ingest(self, proposals: list[KnowledgeEntry]) -> list[str]:
        """Ingest ONE run's proposals; return the field_types promoted by this batch.

        A concept new to the KB is added as provisional; an existing provisional
        one has its synonyms unioned; an authoritative one is left alone. Each
        distinct concept counts as ONE agreement per run (deduped within the batch)
        so promotion means agreement ACROSS runs, not repeats within one.
        """
        promoted: list[str] = []
        counted: set[str] = set()
        for p in proposals:
            existing = self._kb.entries.get(p.field_type)
            if existing is None:
                self._kb.add(p)                                    # new provisional
            elif existing.status == "provisional":
                existing.synonyms = list(dict.fromkeys([*existing.synonyms, *p.synonyms]))
            # authoritative already → leave it
            if p.field_type not in counted:
                counted.add(p.field_type)
                if self._tracker.observe(p.field_type):
                    promoted.append(p.field_type)
        logger.info("dkb_learn_ingest", n=len(proposals), promoted=promoted)
        return promoted

    def confirm(self, field_type: str) -> bool:
        """A developer confirms one provisional concept → authoritative."""
        return self._tracker.confirm(field_type)

    def pending(self) -> list[str]:
        """Provisional concepts still awaiting promotion."""
        return sorted(ft for ft, e in self._kb.entries.items() if e.status == "provisional")

    def save(self, path: Path | str) -> None:
        """Persist the curated KB as a new version."""
        self._kb.save(path)
