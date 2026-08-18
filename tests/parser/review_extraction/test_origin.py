"""Origin labels: Events/Total are source_paper; OR/Weight are review_computed."""
from __future__ import annotations

import json

import pytest

from react_review.llm.base import LLMBackend
from react_review.parser.review_extraction.origin import (
    drop_non_source,
    fields_for_cell,
    label_table,
    match_origin,
)
from react_review.parser.review_extraction.prompts import ExtractionPromptContract
from react_review.parser.review_extraction.schemas import OriginLabel, ReviewLens
from react_review.schemas.table import CapturedTable

LENS = ReviewLens(
    lens_one_line="elderly ESCC, MIE vs OE, postoperative complications",
    outcomes=["overall complications"],
)

FOREST = CapturedTable(
    table_id="fig_3_3_1",
    caption="Figure 3.3.1 Forest plot of overall complications",
    header_rows=[["Study or Subgroup", "Events", "Total", "Odds ratio", "Weight"]],
    rows=[["Li J 2015", "23", "58", "0.40", "12%"],
          ["Total (95% CI)", "58", "102", "0.40", "100%"]],
    row_axis_columns=["Study or Subgroup"],
    display_kind="forest_plot",
)

LABELS = [
    {"table_id": "fig_3_3_1", "column_path": "Events",
     "value_source": "source_paper", "outcome": "overall complications"},
    {"table_id": "fig_3_3_1", "column_path": "Total",
     "value_source": "source_paper", "outcome": "overall complications"},
    {"table_id": "fig_3_3_1", "column_path": "Odds ratio",
     "value_source": "review_computed", "reason": "RevMan OR"},
    {"table_id": "fig_3_3_1", "column_path": "Weight",
     "value_source": "review_computed", "reason": "plot weight"},
    {"table_id": "fig_3_3_1", "column_path": "Events", "row": 1,
     "value_source": "review_computed", "reason": "pooled footer"},
]


class QueueBackend(LLMBackend):
    def __init__(self, responses: list) -> None:
        super().__init__()
        self._responses = [r if isinstance(r, str) else json.dumps(r) for r in responses]
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "queue"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else "{}"


def test_origin_contract_does_not_drift():
    assert ExtractionPromptContract.load("claim_origin_v1").drifts() == []


@pytest.mark.asyncio
async def test_forest_events_and_total_are_source_paper_or_and_weight_are_not():
    backend = QueueBackend([{"labels": LABELS}])
    labels = await label_table(backend, LENS, FOREST)
    by_col = {lab.column_path: lab.value_source for lab in labels if lab.row is None}
    assert by_col["Events"] == "source_paper"
    assert by_col["Total"] == "source_paper"
    assert by_col["Odds ratio"] == "review_computed"
    assert by_col["Weight"] == "review_computed"
    prompt = backend.prompts[0]
    assert LENS.lens_one_line in prompt
    assert "Odds ratio" in prompt
    assert "Abstract" not in prompt


def test_fields_for_cell_drops_review_computed_columns():
    parsed = [OriginLabel.model_validate(item) for item in LABELS]
    events = fields_for_cell(parsed, FOREST, {"column_header": "Events", "row": 0})
    odds = fields_for_cell(parsed, FOREST, {"column_header": "Odds ratio", "row": 0})
    pooled = fields_for_cell(parsed, FOREST, {"column_header": "Events", "row": 1})
    assert events["value_source"] == "source_paper"
    assert not drop_non_source(events["value_source"])
    assert drop_non_source(odds["value_source"])
    assert drop_non_source(pooled["value_source"])
    assert match_origin(parsed, "fig_3_3_1", "Events", 0).value_source == "source_paper"


def test_column_wide_origin_conflict_keeps_the_first_label():
    """Conflicting whole-column labels stay first-wins; the clash is logged."""
    labels = [
        OriginLabel(table_id="fig_3_3_1", column_path="Events",
                    value_source="source_paper"),
        OriginLabel(table_id="fig_3_3_1", column_path="Events",
                    value_source="review_computed"),
    ]
    hit = match_origin(labels, "fig_3_3_1", "Events")
    assert hit is not None
    assert hit.value_source == "source_paper"

