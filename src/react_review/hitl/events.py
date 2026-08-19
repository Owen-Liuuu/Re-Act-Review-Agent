"""Step events — what a pipeline stage reports to the human before continuing.

Every stage that a reviewer must be able to inspect emits ONE :class:`StepEvent`.
The event carries three things the advisor review demanded of every step:

    subject      WHICH FILE this step is processing (an absolute path)
    payload      the FULL content it produced — never a summary
    warnings     what the deterministic checks (and the model itself) flagged

The event is written to disk BEFORE the gate is consulted, so a run that is
stopped — or killed — at the prompt still leaves a complete audit trail.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepStage(str, Enum):
    """The inspectable stages of an audit run, in execution order."""

    REVIEW_PDF_LOADED = "review_pdf_loaded"
    REVIEW_LENS = "review_lens"
    EVIDENCE_LOCALIZE = "evidence_localize"
    TABLE_CAPTURE = "review_table_capture"
    FOREST_OCR = "forest_ocr"
    CLAIM_ORIGIN = "claim_origin"
    COHORT_REGISTRY = "cohort_registry"
    FIELD_RESOLUTION = "field_resolution"
    CHECKLIST_REVIEW = "checklist_review"
    LONG_FORMAT_ROWS = "long_format_rows"
    REFERENCE_COVERAGE = "reference_coverage"
    CHECKLIST_STUDY_COVERAGE = "checklist_study_coverage"
    # Historical Phase-5 journals used this value.  Keep it readable, but new
    # runs emit the two explicit passes above.
    CHECKLIST = "checklist"
    # Per-paper evidence: SHOWN in full as it is collected, but not gated — a
    # review may include 9 source papers or 80, and the checkpoint count must
    # not scale with it. The aggregate below is where the run pauses.
    COLLECT_STUDY = "collect_study"
    COLLECTION_REVIEW = "collection_review"
    AUDIT_SUMMARY = "audit_summary"
    JUDGE_FLAGS = "judge_flags"


class SubjectKind(str, Enum):
    """What the ``subject`` path refers to."""

    REVIEW_PDF = "review_pdf"
    SOURCE_PDF = "source_pdf"
    ONLINE = "online"
    NONE = "none"


class StepEvent(BaseModel):
    """One inspectable checkpoint in a run."""

    run_id: str
    index: int
    screen: int = 0
    stage: StepStage
    title: str = ""
    # The advisor's minimum requirement #1: every step says which file it read.
    subject: str = ""
    subject_kind: SubjectKind = SubjectKind.NONE
    # Minimum requirement #2: the step's FULL content, not a digest.
    payload: dict[str, Any] = Field(default_factory=dict)
    # Pre-rendered terminal blocks (tables etc.) so rendering stays out of the gate.
    render_blocks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Extra decisions this stage can honour beyond continue/stop (e.g. retry).
    offers: list[str] = Field(default_factory=list)
    # Name of a payload key holding a list of {"id", "label"} entries the human
    # may remove at this checkpoint (tables now; references later). Kept generic
    # so the drop mechanism is not tied to one stage.
    selectable: str = ""
    # Which ids were removed — human intervention is itself an auditable step.
    dropped: list[str] = Field(default_factory=list)
    blocking: bool = True
    decision: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    elapsed_ms: int = 0

    @property
    def slug(self) -> str:
        """Filename stem for this event's artifact, e.g. ``003_collect_study``."""
        return f"{self.index:03d}_{self.stage.value}"

    def selectable_items(self) -> list[dict]:
        """The entries a human may drop here (empty when the stage offers none)."""
        if not self.selectable:
            return []
        items = self.payload.get(self.selectable)
        return items if isinstance(items, list) else []

    def drop(self, index: int) -> str:
        """Remove the 1-based ``index`` entry; returns its id ("" if out of range)."""
        items = self.selectable_items()
        if not 1 <= index <= len(items):
            return ""
        removed = items.pop(index - 1)
        rid = str(removed.get("id") or removed.get("table_id") or index)
        self.dropped.append(rid)
        return rid

    def set_on(self, index: int) -> str:
        """Set the 1-based entry ON. Idempotent. Returns its id ("" if out of range)."""
        return self._set_flag(index, True)

    def set_off(self, index: int) -> str:
        """Set the 1-based entry OFF. Idempotent. Returns its id ("" if out of range)."""
        return self._set_flag(index, False)

    def _set_flag(self, index: int, on: bool) -> str:
        items = self.selectable_items()
        if not 1 <= index <= len(items):
            return ""
        item = items[index - 1]
        rid = str(item.get("id") or item.get("table_id") or item.get("display_id")
                  or index)
        if "evidence_chain" in item:
            item["evidence_chain"] = on
        if on:
            self.dropped = [d for d in self.dropped if d != rid]
        elif rid not in self.dropped:
            self.dropped.append(rid)
        item["label"] = _flag_label(str(item.get("label") or rid), on)
        return rid


def _flag_label(label: str, on: bool) -> str:
    """Ensure a selectable label starts with ``[on]`` / ``[off]``."""
    text = label.strip()
    for prefix in ("[on]", "[off]", "[ON]", "[OFF]"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].lstrip()
            break
    flag = "on" if on else "off"
    return f"[{flag}] {text}".strip()
