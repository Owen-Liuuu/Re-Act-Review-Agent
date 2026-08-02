"""How much friction each stage gets.

The advisor asked for a checkpoint at every step but also for it to cost only
"one press of C". Those pull against each other on a 60-value run, so the knob
is per-stage: the structural stages always GATE, while per-item work is SHOWN in
full inside its study block and gated once per source paper.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from react_review.hitl.events import StepStage


class Mode(str, Enum):
    GATE = "gate"      # print everything, then wait for a key
    SHOW = "show"      # print everything, continue automatically
    SILENT = "silent"  # journal only (still written to disk)


_ALL_STAGES = tuple(StepStage)


class CheckpointPolicy(BaseModel):
    """Per-stage interaction mode."""

    by_stage: dict[StepStage, Mode] = Field(default_factory=dict)
    default: Mode = Mode.SHOW

    def mode_for(self, stage: StepStage) -> Mode:
        return self.by_stage.get(stage, self.default)

    @classmethod
    def key_stages(cls) -> "CheckpointPolicy":
        """Default: gate the structural stages; SHOW each paper, gate the batch.

        Per-paper evidence is printed in full but not gated, so the number of
        keypresses is the same whether the review includes 9 source papers or 80.
        """
        by_stage = {s: Mode.GATE for s in _ALL_STAGES}
        by_stage[StepStage.COLLECT_STUDY] = Mode.SHOW
        return cls(by_stage=by_stage, default=Mode.GATE)

    @classmethod
    def all_stages(cls) -> "CheckpointPolicy":
        """Gate everything, including each individual source paper."""
        return cls(by_stage={s: Mode.GATE for s in _ALL_STAGES}, default=Mode.GATE)

    @classmethod
    def none(cls) -> "CheckpointPolicy":
        """Never block. The journal is still written — that is not optional."""
        return cls(by_stage={s: Mode.SILENT for s in _ALL_STAGES}, default=Mode.SILENT)

    @classmethod
    def from_name(cls, name: str) -> "CheckpointPolicy":
        return {"key": cls.key_stages, "all": cls.all_stages, "none": cls.none}[name]()
