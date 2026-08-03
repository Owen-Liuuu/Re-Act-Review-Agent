"""Which document was read, and why an outcome happened — recorded, not logged."""
from __future__ import annotations

import json

import pytest

from react_review.agents.collector import Collector, _provenance
from react_review.core.enums import CollectionOutcome
from react_review.llm.base import LLMBackend
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry

_REF = ReferenceEntry(title="Ahmad 2022", doi="10.1/x")
_REVIEW = ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                         field_type="eat_thickness", value="6.60 ± 0.71", unit="mm")


class _Stub(LLMBackend):
    def __init__(self, payload) -> None:
        super().__init__()
        self._payload = payload

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        if isinstance(self._payload, Exception):
            raise self._payload
        return json.dumps(self._payload)


class _Retriever(PaperRetriever):
    def __init__(self, metadata: dict) -> None:
        self._metadata = metadata

    async def retrieve(self, reference):
        return PaperDocument(paper_id=reference.doi or "x", reference=reference,
                             full_text="EFT 6.60 ± 0.71 mm in diabetic children.",
                             metadata=self._metadata)


def _collector(metadata: dict, payload) -> Collector:
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(_Retriever(metadata)))
    reg.register(ExtractSourceValueTool(_Stub(payload)))
    return Collector(reg)


# --- the location of each retriever tier lands in one field ---

@pytest.mark.parametrize("metadata, uri, kind", [
    ({"source": "local_pdf", "path": "D:/pdf/Ahmad.pdf"}, "D:/pdf/Ahmad.pdf", "local_pdf"),
    ({"source": "pmc", "pmc_id": "PMC123"}, "PMC123", "pmc"),
    ({"source": "unpaywall", "oa_url": "https://x/oa.pdf"}, "https://x/oa.pdf", "unpaywall"),
    ({"source": "openalex_pdf", "pdf_url": "https://x/w.pdf"}, "https://x/w.pdf", "openalex_pdf"),
    ({"source": "pubmed_abstract", "pmid": "99"}, "99", "pubmed_abstract"),
])
def test_every_retriever_tier_reports_where_it_read_from(metadata, uri, kind):
    doc = PaperDocument(paper_id="p", reference=_REF, metadata=metadata)
    prov = _provenance(doc)
    assert prov["source_uri"] == uri and prov["retriever_kind"] == kind
    assert prov["source_doi"] == "10.1/x"


def test_a_local_run_records_the_file_path():
    doc = PaperDocument(paper_id="p", reference=_REF,
                        metadata={"source": "local_pdf", "path": "D:/pdf/Ahmad.pdf"})
    assert _provenance(doc)["source_file"] == "D:/pdf/Ahmad.pdf"


def test_an_online_run_has_no_file_but_still_has_a_location():
    doc = PaperDocument(paper_id="p", reference=_REF,
                        metadata={"source": "pmc", "pmc_id": "PMC7"})
    prov = _provenance(doc)
    assert prov["source_file"] == "" and prov["source_uri"] == "PMC7"


@pytest.mark.asyncio
async def test_evidence_says_which_document_it_came_from():
    collector = _collector({"source": "local_pdf", "path": "D:/pdf/Ahmad.pdf"},
                           {"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                            "group_label_in_paper": "T1DM", "quote": "EFT 6.60"})
    res = await collector.collect(_REVIEW, _REF)
    assert res.source_item.source_file == "D:/pdf/Ahmad.pdf"
    assert res.source_item.retriever_kind == "local_pdf"


# --- failures carry a reason, in words ---

@pytest.mark.asyncio
async def test_a_transport_error_is_recorded_not_just_logged():
    # Without this the exception text lived only in a log line, and a network
    # failure was indistinguishable from a paper that omits the value.
    collector = _collector({"source": "local_pdf", "path": "p.pdf"},
                           RuntimeError("connection reset by peer"))
    res = await collector.collect(_REVIEW, _REF)
    codes = {r.code for r in res.source_item.reasons}
    assert "extraction_error" in codes
    assert any("connection reset" in r.message for r in res.source_item.reasons)


@pytest.mark.asyncio
async def test_the_models_own_explanation_survives():
    collector = _collector({"source": "local_pdf", "path": "p.pdf"},
                           {"found": False, "value": None,
                            "not_found_reason": "the paper reports EAT only for the "
                                                "whole cohort, not per arm",
                            "cohorts_seen": ["T1DM", "Control"]})
    res = await collector.collect(_REVIEW, _REF)
    assert res.source_item.collection_outcome is CollectionOutcome.MISSING_SOURCE
    messages = " ".join(r.message for r in res.source_item.reasons)
    assert "only for the whole cohort" in messages
    assert res.source_item.cohorts_seen == ["T1DM", "Control"]


@pytest.mark.asyncio
async def test_a_study_with_no_citation_is_not_searched_for_by_its_id():
    # Searching for a paper called "ahmad_2022" would match something unrelated.
    from react_review.parser.review_parser import ParsedStudy
    from react_review.study_match import build_reference_resolver_from_parsed

    resolve = build_reference_resolver_from_parsed([ParsedStudy(study_id="other_1999")])
    collector = _collector({"source": "local_pdf", "path": "p.pdf"},
                           {"found": True, "value": "x"})
    res = await collector.collect(_REVIEW, resolve("ahmad_2022"))
    assert res.source_item.collection_outcome is CollectionOutcome.UNRESOLVED_SOURCE
    assert any("no citation" in r.message for r in res.source_item.reasons)
