"""P6/P7 live on SplitAwareCollector, not the hash-frozen collector.py."""
from __future__ import annotations

import pytest

from react_review.agents.collector import Collector
from react_review.agents.split_collector import SplitAwareCollector
from react_review.core.enums import CollectionOutcome, ReflectionDecision
from tests.agents.test_collector import (
    StubBackend,
    _DocRetriever,
    _REVIEW,
    _REF,
    _catalogue,
)


@pytest.mark.asyncio
async def test_split_collector_records_a_human_only_field_mapping():
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "source_field_name": "EFT",
                           "location": "Table 2"})
    collector = SplitAwareCollector(_catalogue(_DocRetriever(), backend))
    res = await collector.collect(_REVIEW, _REF, research_context="EAT in T1DM")
    assert res.decision == ReflectionDecision.ACCEPT
    assert res.source_item.collection_outcome == CollectionOutcome.FOUND
    mapping = [r for r in res.source_item.reasons if r.code == "source_field_mapping"]
    assert len(mapping) == 1
    assert mapping[0].source == "llm"
    assert mapping[0].detail == {}
    assert "eat_thickness" in mapping[0].message
    assert "EFT" in mapping[0].message
    assert "human check only" in mapping[0].message
    assert "source_field_name" not in res.source_item.model_dump()
    assert res.record.final["reasons"][-1]["code"] == "source_field_mapping"


@pytest.mark.asyncio
async def test_mapping_is_absent_when_the_paper_label_was_not_returned():
    """An empty paper label must not mint a mapping reason — old packages stay quiet."""
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "location": "Table 2"})
    collector = SplitAwareCollector(_catalogue(_DocRetriever(), backend))
    res = await collector.collect(_REVIEW, _REF)
    assert not any(r.code == "source_field_mapping" for r in res.source_item.reasons)


@pytest.mark.asyncio
async def test_frozen_collector_does_not_emit_mapping_even_when_the_paper_names_the_field():
    """P7 must not leak into collector.py: that file is inside adequacy 1.0.0."""
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "source_field_name": "EFT",
                           "location": "Table 2"})
    collector = Collector(_catalogue(_DocRetriever(), backend))
    res = await collector.collect(_REVIEW, _REF)
    assert not any(r.code == "source_field_mapping" for r in res.source_item.reasons)
    assert "source_field_name" not in res.source_item.model_dump()
