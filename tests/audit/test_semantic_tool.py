"""When the semantic path is reached, and how a recorded run stays recorded."""
from __future__ import annotations

import json

import pytest

from react_review.audit import ToleranceTable
from react_review.audit.semantic_cache import SemanticCache, SemanticCacheMiss
from react_review.core.enums import AuditLabel
from react_review.llm.base import LLMBackend
from react_review.tools.compare import CompareValuesTool
from react_review.tools.models import CompareInput
from react_review.tools.semantic_compare import SemanticCompareTool


class _Stub(LLMBackend):
    def __init__(self, payload) -> None:
        super().__init__()
        self._payload = payload
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return json.dumps(self._payload)


_SAME = {"relation": "same", "equivalent": True, "confidence": 0.95,
         "rationale": "an abbreviation of the same term",
         "evidence_span": "intensive care unit"}


def _tool(payload=_SAME, *, mode="on", cache=None):
    backend = _Stub(payload)
    tool = CompareValuesTool(ToleranceTable(), semantic=SemanticCompareTool(backend),
                             semantic_mode=mode, semantic_cache=cache)
    return tool, backend


def _text_pair(review="ICU", source="intensive care unit",
               quote="Recruited in the intensive care unit.") -> CompareInput:
    return CompareInput(field_type="setting", review_value=review, source_value=source,
                        column_header="Setting", source_quote=quote,
                        research_context="critical care")


# --- when the semantic path runs at all ---

@pytest.mark.asyncio
async def test_text_that_cannot_be_read_as_numbers_escalates():
    tool, backend = _tool()
    out = await tool.run(_text_pair())
    assert out.label is AuditLabel.MATCH and out.match_mode == "semantic"
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_numbers_never_escalate():
    # A numeric pair is decided deterministically; the model is not consulted.
    tool, backend = _tool()
    out = await tool.run(CompareInput(field_type="bmi", review_value="24.1",
                                      source_value="24.1"))
    assert out.match_mode == "numeric" and backend.calls == 0


@pytest.mark.asyncio
async def test_a_decided_structured_value_never_escalates():
    # "p < 0.001" vs "= 0.0009" is not_comparable BY DECISION, not by failure to
    # parse — handing it to a model would overturn a deliberate verdict.
    tool, backend = _tool()
    out = await tool.run(CompareInput(field_type="p_value", review_value="p < 0.001",
                                      source_value="p = 0.0009"))
    assert out.label is AuditLabel.NOT_COMPARABLE and backend.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("review, source", [
    ("NR", "not reported"), ("—", "N/A"), ("ICU", ""), ("", "ward"),
])
async def test_absent_values_are_not_sent_for_equivalence(review, source):
    # Two cells that both say "nothing here" are not evidence of agreement.
    tool, backend = _tool()
    out = await tool.run(_text_pair(review, source, quote=""))
    assert backend.calls == 0 and out.label is AuditLabel.NOT_COMPARABLE


@pytest.mark.asyncio
async def test_off_is_the_default_and_leaves_the_audit_deterministic():
    tool = CompareValuesTool(ToleranceTable())
    out = await tool.run(_text_pair())
    assert out.label is AuditLabel.NOT_COMPARABLE and out.match_mode == "numeric"


# --- the controls still bind at the tool level ---

@pytest.mark.asyncio
async def test_values_that_parse_as_numbers_are_settled_before_any_model_sees_them():
    # "Grade 3" / "Grade 4" carry readable numbers, so the deterministic path
    # decides and the model is never asked — the numeric guarantee holds by
    # never reaching the semantic path at all.
    tool, backend = _tool({"relation": "same", "confidence": 0.99})
    out = await tool.run(_text_pair("Grade 3", "Grade 4", quote="Toxicity Grade 4."))
    assert out.label is AuditLabel.MISMATCH and backend.calls == 0


@pytest.mark.asyncio
async def test_an_ungrounded_claim_is_refused_at_the_tool_level():
    # The controls are not advisory: a claim whose cited span is absent from the
    # quote does not become a verdict, however confident the model is.
    tool, _ = _tool({"relation": "same", "confidence": 0.99,
                     "rationale": "same setting",
                     "evidence_span": "a coronary care unit"})
    out = await tool.run(_text_pair(quote="Recruited in the intensive care unit."))
    assert out.label is AuditLabel.NOT_COMPARABLE
    assert out.semantic_controls["anchor"] is False and out.review_required


@pytest.mark.asyncio
async def test_a_broader_match_is_recorded_as_needing_review():
    tool, _ = _tool({"relation": "source_broader", "confidence": 0.9,
                     "rationale": "the source is more specific",
                     "evidence_span": "France, surgical ICU"})
    out = await tool.run(_text_pair("France", "France, surgical ICU",
                                    quote="Conducted in France, surgical ICU."))
    assert out.label is AuditLabel.MATCH and out.review_required is True
    assert out.semantic_relation == "source_broader"


# --- recording and replaying ---

@pytest.mark.asyncio
async def test_a_judgement_is_recorded_and_then_reused(tmp_path):
    cache = SemanticCache(tmp_path / "semantic_cache.json")
    tool, backend = _tool(cache=cache)
    await tool.run(_text_pair())
    await tool.run(_text_pair())
    assert backend.calls == 1 and cache.hits == 1
    assert cache.save().is_file()


@pytest.mark.asyncio
async def test_a_replay_needs_no_model_at_all(tmp_path):
    path = tmp_path / "semantic_cache.json"
    record_tool, backend = _tool(cache=SemanticCache(path))
    await record_tool.run(_text_pair())
    record_tool._cache.save()                       # noqa: SLF001 — recording step

    replay = CompareValuesTool(ToleranceTable(), semantic=None,
                               semantic_mode="cache-only",
                               semantic_cache=SemanticCache(path))
    out = await replay.run(_text_pair())
    assert out.label is AuditLabel.MATCH and out.match_mode == "semantic"


@pytest.mark.asyncio
async def test_cache_only_fails_loudly_instead_of_calling_the_model(tmp_path):
    # A run claiming to reproduce a recording must not quietly stop being one.
    tool, backend = _tool(mode="cache-only", cache=SemanticCache(tmp_path / "c.json"))
    with pytest.raises(SemanticCacheMiss):
        await tool.run(_text_pair())
    assert backend.calls == 0


def test_the_cache_key_covers_everything_that_changes_the_answer():
    from react_review.tools.semantic_compare import cache_key
    base = dict(model_id="m", prompt_version="v1", field_type="setting",
                column_header="Setting", research_context="ctx",
                review_value="ICU", review_unit="", source_value="ward",
                source_unit="", source_quote="q", seed=42)
    for changed in ("model_id", "prompt_version", "field_type", "column_header",
                    "research_context", "review_value", "source_value",
                    "source_quote", "seed"):
        assert cache_key(base) != cache_key({**base, changed: "different"}), changed
