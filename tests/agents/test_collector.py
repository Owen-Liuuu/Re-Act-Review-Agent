"""Tests for extract_source_value + the Collector agent (stub backend, no net)."""
from __future__ import annotations

import json

import pytest

from react_review.agents.collector import Collector
from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.dkb import KnowledgeBase, KnowledgeEntry
from react_review.llm.base import LLMBackend
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
    cohort_conflicts,
    cohort_description,
)
from react_review.tools.registry import ToolRegistry


class StubBackend(LLMBackend):
    def __init__(self, payload) -> None:
        super().__init__()
        self._payload = payload
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "stub"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)


class _DocRetriever(PaperRetriever):
    async def retrieve(self, reference):
        return PaperDocument(
            paper_id=reference.doi or "x", reference=reference,
            full_text="Table 2 reports EFT of 6.60 ± 0.71 mm in diabetic children.",
        )


class _NoneRetriever(PaperRetriever):
    async def retrieve(self, reference):
        return None


def _catalogue(retriever, extract_backend) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(retriever))
    reg.register(ExtractSourceValueTool(extract_backend))
    return reg


_REVIEW = ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                         field_type="eat_thickness", value="6.60 ± 0.71", unit="mm")
_REF = ReferenceEntry(title="Ahmad 2022", doi="10.1/x")


# --- extract_source_value tool ---

@pytest.mark.asyncio
async def test_extract_tool_found():
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "source_field_name": "EFT",
                           "location": "Table 2"})
    tool = ExtractSourceValueTool(backend)
    doc = PaperDocument(paper_id="x", reference=_REF, full_text="…")
    out = await tool.run(ExtractSourceValueInput(document=doc, field_type="eat_thickness", group="t1dm"))
    assert out.found is True and out.value == "6.60 ± 0.71" and out.unit == "mm"


@pytest.mark.asyncio
async def test_extract_tool_not_found_nullifies_value():
    backend = StubBackend({"found": False, "value": "null"})
    out = await ExtractSourceValueTool(backend).run(
        ExtractSourceValueInput(document=PaperDocument(paper_id="x", reference=_REF),
                                field_type="bmi", group="control"))
    assert out.found is False and out.value is None


@pytest.mark.asyncio
async def test_extract_tool_unparseable_is_not_found():
    out = await ExtractSourceValueTool(StubBackend("not json")).run(
        ExtractSourceValueInput(document=PaperDocument(paper_id="x", reference=_REF),
                                field_type="bmi"))
    assert out.found is False


# --- cohort guard: checked against the REVIEW's own labels, not a disease list ---

# What this review was found to report — no diabetes vocabulary in the code.
_COHORTS = {"t1dm": ["T1DM", "Diabetic children"],
            "control": ["Control", "Healthy controls"]}


@pytest.mark.parametrize("target, label, verdict", [
    ("control", "Diabetic children", "wrong_cohort"),   # read the other arm's column
    ("t1dm", "Control group", "wrong_cohort"),
    ("control", "Controls", "ok"),
    ("control", "healthy controls", "ok"),
    ("t1dm", "T1DM patients", "ok"),
    ("t1dm", "Diabetic children", "ok"),
    ("control", "", "ok"),                              # no reported label → no guard
    ("all", "Diabetic children", "ok"),                 # non-split ask → skip
    # NOT ok: the guard could not confirm the arm, so it must not pass as clean.
    ("control", "Cohort B", "ambiguous"),
    ("placebo", "Placebo arm", "ambiguous"),            # target not in this registry
])
def test_cohort_conflicts(target, label, verdict):
    assert cohort_conflicts(target, label, cohorts=_COHORTS)[0] == verdict


def test_no_registry_means_the_guard_is_not_configured_not_that_it_failed():
    # Auditing two CSVs whose groups are already canonical supplies no cohorts.
    # That is "nothing to check against", not "could not confirm" — flagging
    # every row there would bury the cases that genuinely need a look.
    assert cohort_conflicts("control", "Diabetic children", cohorts={})[0] == "ok"
    # But a registry that does NOT contain the target IS suspicious.
    assert cohort_conflicts("placebo", "Placebo arm", cohorts=_COHORTS)[0] == "ambiguous"


def test_cohort_description_uses_the_reviews_own_words():
    desc = cohort_description("treatment", display="Nivolumab arm",
                              variants=["Nivolumab arm", "Treatment"])
    assert '"Nivolumab arm" cohort' in desc and "Treatment" in desc
    assert "diabet" not in desc.lower()          # no disease vocabulary anywhere


def test_unidentified_cohort_is_not_described_as_the_whole_study():
    # Saying "whole study cohort" here would have the model fetch a pooled
    # number for a claim that is about one arm.
    desc = cohort_description("")
    assert "did not identify" in desc and "whole study" not in desc
    assert "whole study cohort" in cohort_description("all")


@pytest.mark.asyncio
async def test_extract_tool_rejects_wrong_cohort_value():
    backend = StubBackend({"found": True, "value": "12.90 ± 1.30", "unit": "years",
                           "group_label_in_paper": "Diabetic children",
                           "quote": "Age 12.90 ± 1.30", "location": "Table 1"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="control", cohorts=_COHORTS))
    assert out.found is False and out.value is None      # rejected as wrong cohort
    assert out.group_label_in_paper == "Diabetic children"
    assert out.cohort_check == "wrong_cohort"
    assert "Diabetic children" in out.not_found_reason   # says WHY it was rejected


@pytest.mark.asyncio
async def test_unverifiable_cohort_is_kept_but_marked_not_passed_as_clean():
    # The paper names a cohort this review does not report. The value is still
    # useful evidence, but nobody has confirmed which arm it belongs to, so it
    # must not read as a clean result.
    backend = StubBackend({"found": True, "value": "12.90 ± 1.30", "unit": "years",
                           "group_label_in_paper": "Cohort B", "quote": "Age 12.90"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="control", cohorts=_COHORTS))
    assert out.found is True                      # evidence kept
    assert out.cohort_check == "ambiguous"        # but explicitly unconfirmed
    assert out.cohort_reason


@pytest.mark.asyncio
async def test_a_quote_that_resembles_no_cohort_name_is_not_treated_as_a_problem():
    # A quote is prose. Not matching a short cohort label is normal for a
    # sentence and must not downgrade a correct extraction — only positive
    # evidence of ANOTHER arm counts against it.
    backend = StubBackend({"found": True, "value": "12.90 ± 1.30", "unit": "years",
                           "group_label_in_paper": "T1DM",
                           "quote": "The mean age of the study population was "
                                    "12.90 ± 1.30 years overall."})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="t1dm", cohorts={"t1dm": ["T1DM"],
                                                 "control": ["Control"]}))
    assert out.found is True and out.cohort_check == "ok"


@pytest.mark.asyncio
async def test_extract_tool_accepts_matching_cohort():
    backend = StubBackend({"found": True, "value": "12.96 ± 1.12", "unit": "years",
                           "group_label_in_paper": "Healthy controls",
                           "quote": "Controls 12.96 ± 1.12", "location": "Table 1"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="control"))
    assert out.found is True and out.value == "12.96 ± 1.12"


@pytest.mark.asyncio
async def test_extract_tool_rejects_when_quote_names_wrong_cohort():
    # The model faked a matching label but its quote is a diabetic-only sentence
    # (the real Ahmad failure: inferring control from "no significant difference").
    backend = StubBackend({"found": True, "value": "12.90 ± 1.30", "unit": "years",
                           "group_label_in_paper": "healthy controls",
                           "quote": "Regarding diabetic children, the mean age was "
                                    "12.90 ± 1.30 years.", "location": "Results"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF),
        field_type="age", group="control", cohorts=_COHORTS))
    assert out.found is False and out.value is None


# --- Collector ---

@pytest.mark.asyncio
async def test_collector_produces_source_item():
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "source_field_name": "EFT",
                           "location": "Table 2"})
    collector = Collector(_catalogue(_DocRetriever(), backend))
    res = await collector.collect(_REVIEW, _REF, research_context="EAT in T1DM")
    assert res.decision == ReflectionDecision.ACCEPT
    assert res.source_item.source_value == "6.60 ± 0.71"
    assert res.source_item.source_unit == "mm"
    assert res.source_item.study_id == "ahmad_2022"
    assert res.source_item.collection_outcome == CollectionOutcome.FOUND
    assert backend.calls == 1  # found on first attempt, no retry
    # trajectory recorded: fetch + one extract
    assert [s.tool for s in res.record.steps] == ["fetch_fulltext", "extract_source_value"]


@pytest.mark.asyncio
async def test_collector_retries_then_escalates_when_not_found():
    backend = StubBackend({"found": False, "value": None})
    collector = Collector(_catalogue(_DocRetriever(), backend), max_attempts=3)
    res = await collector.collect(_REVIEW, _REF)
    assert res.decision == ReflectionDecision.ESCALATE
    assert res.source_item.source_value is None
    # retrieved the paper but never located the value → potential fabrication
    assert res.source_item.collection_outcome == CollectionOutcome.MISSING_SOURCE
    assert backend.calls == 3  # tried max_attempts times


def _kb_thickness() -> KnowledgeBase:
    kb = KnowledgeBase()
    kb.add(KnowledgeEntry(field_type="eat_thickness",
                          concept="epicardial fat thickness", default_unit="mm"))
    return kb


@pytest.mark.asyncio
async def test_collector_back_check_flags_contradicting_concept():
    # A CANDIDATE mapping (eat_thickness, expects mm) but the source reports a
    # volume (cm3) → the source evidence refutes the parse-time translation.
    backend = StubBackend({"found": True, "value": "52.3", "unit": "cm3",
                           "quote": "EAT volume 52.3 cm3", "location": "Table 1"})
    review = ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                            field_type="eat_thickness", raw_field_name="EAT",
                            value="52.3", unit="cm3", resolution_status="candidate")
    collector = Collector(_catalogue(_DocRetriever(), backend), knowledge=_kb_thickness())
    res = await collector.collect(review, _REF)
    assert res.source_item.concept_mismatch is True
    assert "different kind" in res.source_item.concept_mismatch_reason


@pytest.mark.asyncio
async def test_collector_no_back_check_for_resolved_items():
    # A trusted (resolved) concept is never second-guessed by the back-check.
    backend = StubBackend({"found": True, "value": "52.3", "unit": "cm3",
                           "quote": "x", "location": "y"})
    review = ReviewDataItem(study_id="ahmad_2022", group="t1dm",
                            field_type="eat_thickness", value="52.3", unit="cm3",
                            resolution_status="resolved")
    collector = Collector(_catalogue(_DocRetriever(), backend), knowledge=_kb_thickness())
    res = await collector.collect(review, _REF)
    assert res.source_item.concept_mismatch is False


def _catalogue_with_resolve(retriever, extract_backend, candidates):
    from react_review.tools.search import (
        ReferenceReconciler, ResolveReferenceTool, StaticResolver)
    reg = _catalogue(retriever, extract_backend)
    reg.register(ResolveReferenceTool(
        ReferenceReconciler([StaticResolver("crossref", candidates)])))
    return reg


@pytest.mark.asyncio
async def test_collector_resolves_missing_doi_then_fetches():
    from react_review.tools.search import CandidateWork
    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "location": "Table 2"})
    cand = CandidateWork(doi="10.1/x", title="Ahmad 2022 EAT study", authors=["Ahmad A"],
                         year=2022, journal="J Cardiol", source="crossref")
    ref = ReferenceEntry(title="Ahmad 2022 EAT study", authors=["Ahmad A"],
                         year=2022, journal="J Cardiol")               # NO doi
    reg = _catalogue_with_resolve(_DocRetriever(), backend, [cand])
    res = await Collector(reg).collect(_REVIEW, ref)
    assert res.source_item.source_value == "6.60 ± 0.71"
    assert res.record.steps[0].tool == "resolve_reference"            # resolved before fetch


@pytest.mark.asyncio
async def test_collector_unresolved_source_when_citation_cannot_resolve():
    backend = StubBackend({"found": True, "value": "x"})
    ref = ReferenceEntry(title="An uncited grey-literature report", year=2010)   # NO doi
    reg = _catalogue_with_resolve(_DocRetriever(), backend, [])        # resolver finds nothing
    res = await Collector(reg).collect(_REVIEW, ref)
    assert res.source_item.collection_outcome == CollectionOutcome.UNRESOLVED_SOURCE
    assert res.source_item.source_value is None and backend.calls == 0   # never fetched/extracted


@pytest.mark.asyncio
async def test_chain_parsed_study_to_collector_resolves_and_fetches():
    # End-to-end (offline): the parser's DOI-less reference list → a ReferenceEntry
    # → the Collector reconciles its DOI from the citation → fetches → extracts.
    from react_review.parser.review_parser import ParsedStudy
    from react_review.study_match import build_reference_resolver_from_parsed
    from react_review.tools.search import CandidateWork

    citation = "Ahmad A. Epicardial fat thickness in type 1 diabetes. J Cardiol. 2022."
    resolve_ref = build_reference_resolver_from_parsed(
        [ParsedStudy(study_id="ahmad_2022", citation=citation, doi="")])   # NO printed DOI
    ref = resolve_ref("ahmad_2022")
    assert ref.doi is None

    backend = StubBackend({"found": True, "value": "6.60 ± 0.71", "unit": "mm",
                           "quote": "EFT of 6.60 ± 0.71 mm", "location": "Table 2"})
    cand = CandidateWork(doi="10.1/x", title=citation, source="crossref")   # matches the citation
    reg = _catalogue_with_resolve(_DocRetriever(), backend, [cand])
    res = await Collector(reg).collect(_REVIEW, ref)
    assert res.record.steps[0].tool == "resolve_reference"       # DOI reconciled first
    assert res.source_item.source_value == "6.60 ± 0.71"         # then fetched + extracted


@pytest.mark.asyncio
async def test_collector_escalates_when_paper_not_retrieved():
    backend = StubBackend({"found": True, "value": "x"})
    collector = Collector(_catalogue(_NoneRetriever(), backend))
    res = await collector.collect(_REVIEW, _REF)
    assert res.decision == ReflectionDecision.RETRY or res.decision == ReflectionDecision.ESCALATE
    assert res.source_item.source_value is None
    # never got the paper → access failure, NOT a fabrication signal
    assert res.source_item.collection_outcome == CollectionOutcome.SOURCE_ACCESS_FAILED
    assert backend.calls == 0  # never reached extraction
    assert res.record.steps[0].observation["retrieved"] is False
