"""Localize: Table 1 on, Table 2 off, four forests on. Not an ESCC/EAT regex."""
from __future__ import annotations

import json

import pytest

from react_review.llm.base import LLMBackend
from react_review.parser.review_extraction.localize import _hit, localize, selected
from react_review.parser.review_extraction.prompts import ExtractionPromptContract
from react_review.parser.review_extraction.schemas import ReviewLens
from react_review.parser.review_extraction.windows import capture_window, results_window


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


DOC05 = """
Abstract SECRET_ABSTRACT_TOKEN elderly ESCC MIE vs OE.

Introduction
Methods

Results
Table 1 Characteristics of the included studies.
Figure 3.3.1 Forest plot of overall complications
Figure 3.3.2 Forest plot of pulmonary complications
Figure 3.3.3 Forest plot of 30-day mortality
Figure 3.3.4 Forest plot of anastomotic leak
Table 2 GRADE summary of findings. Outcome OR Quality.

Discussion
References
1. Li J 2015
"""

EAT = """
Abstract EAT thickness in T1DM versus healthy controls.

Introduction

Results
Table 1 Characteristics of included studies. Study, Country, EAT thickness.
Ahmad 2022

Discussion
References
"""

LENS_ESCC = ReviewLens(
    lens_one_line="elderly ESCC, MIE vs OE, postoperative complications",
    domain="esophageal cancer surgery",
    population="elderly ESCC",
    comparison="MIE vs OE",
    outcomes=["overall complications", "pulmonary", "30-day mortality",
              "anastomotic leak"],
    not_audit_focus=["pooled GRADE"],
)

DOC05_HITS = [
    {"display_id": "table_1", "kind": "pdf_table",
     "caption": "Table 1 Characteristics of the included studies",
     "evidence_chain": True, "reason": "per-study N for included papers"},
    {"display_id": "table_2", "kind": "pdf_table",
     "caption": "Table 2 GRADE summary of findings",
     "evidence_chain": False, "reason": "pooled OR / GRADE, not per-study raw counts"},
    {"display_id": "fig_3_3_1", "kind": "forest_plot",
     "caption": "Figure 3.3.1 Forest plot of overall complications",
     "evidence_chain": True, "reason": "per-study events for overall complications"},
    {"display_id": "fig_3_3_2", "kind": "forest_plot",
     "caption": "Figure 3.3.2 Forest plot of pulmonary complications",
     "evidence_chain": True, "reason": "per-study events for pulmonary"},
    {"display_id": "fig_3_3_3", "kind": "forest_plot",
     "caption": "Figure 3.3.3 Forest plot of 30-day mortality",
     "evidence_chain": True, "reason": "per-study events for 30-day mortality"},
    {"display_id": "fig_3_3_4", "kind": "forest_plot",
     "caption": "Figure 3.3.4 Forest plot of anastomotic leak",
     "evidence_chain": True, "reason": "per-study events for anastomotic leak"},
]


def test_results_window_excludes_abstract_and_references():
    window = results_window(DOC05)
    assert "Table 1" in window and "Figure 3.3.1" in window
    assert "SECRET_ABSTRACT_TOKEN" not in window
    assert "References" not in window


def test_capture_window_starts_after_the_abstract():
    window = capture_window(DOC05)
    assert "Table 1" in window
    assert "SECRET_ABSTRACT_TOKEN" not in window


def test_localize_contract_does_not_drift():
    assert ExtractionPromptContract.load("evidence_localize_v1").drifts() == []


def test_page_hint_must_be_digits_or_empty():
    echoed = _hit({"display_id": "figure_2", "kind": "forest_plot",
                   "page_hint": "printed page or empty",
                   "evidence_chain": True}, 1)
    assert echoed is not None and echoed.page_hint == ""
    numeric = _hit({"display_id": "figure_2", "kind": "forest_plot",
                    "page_hint": "10", "evidence_chain": True}, 1)
    assert numeric is not None and numeric.page_hint == "10"
    blank = _hit({"display_id": "figure_2", "kind": "forest_plot",
                  "page_hint": "", "evidence_chain": True}, 1)
    assert blank is not None and blank.page_hint == ""


@pytest.mark.asyncio
async def test_doc05_fixture_keeps_table1_and_four_forests_drops_table2():
    backend = QueueBackend([{"displays": DOC05_HITS}])
    hits = await localize(backend, LENS_ESCC, DOC05)
    on = {(h.display_id, h.kind) for h in hits if h.evidence_chain}
    off = {h.display_id for h in hits if not h.evidence_chain}
    assert ("table_1", "pdf_table") in on
    assert {h.display_id for h in selected(hits, kind="forest_plot")} == {
        "fig_3_3_1", "fig_3_3_2", "fig_3_3_3", "fig_3_3_4"}
    assert "table_2" in off
    prompt = backend.prompts[0]
    assert LENS_ESCC.lens_one_line in prompt
    assert "SECRET_ABSTRACT_TOKEN" not in prompt
    assert "evidence_chain=true only when" in prompt


@pytest.mark.asyncio
async def test_eat_fixture_does_not_invent_escc_forests():
    backend = QueueBackend([{"displays": [
        {"display_id": "table_1", "kind": "pdf_table",
         "caption": "Table 1 Characteristics of included studies",
         "evidence_chain": True, "reason": "per-study EAT for included papers"},
    ]}])
    eat_lens = ReviewLens(
        lens_one_line="EAT thickness in T1DM vs healthy controls",
        domain="cardiometabolic imaging",
        population="T1DM",
        comparison="T1DM vs healthy controls",
        outcomes=["EAT thickness"],
    )
    hits = await localize(backend, eat_lens, EAT)
    assert [h.display_id for h in hits if h.evidence_chain] == ["table_1"]
    assert not selected(hits, kind="forest_plot")
    assert "esophageal" not in backend.prompts[0].lower()
    assert "EAT thickness in T1DM" in backend.prompts[0]
