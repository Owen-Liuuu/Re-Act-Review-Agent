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
    _paper_excerpt,
    cohort_conflicts,
    cohort_description,
)
from react_review.tools.extraction_cache import ExtractionCache, ExtractionCacheMiss
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
            full_text=("Table 2 reports EFT of 6.60 ± 0.71 mm in diabetic children. "
                       "EAT volume 52.3 cm3."),
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
    doc = PaperDocument(paper_id="x", reference=_REF,
                        full_text="EFT of 6.60 ± 0.71 mm")
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


@pytest.mark.asyncio
async def test_direct_value_is_rejected_when_quote_is_not_in_source_document():
    backend = StubBackend({"found": True, "value": "4.8 (4.2–5.4)", "unit": "mm",
                           "quote": "EFT was 4.8 (4.2–5.4) mm."})
    out = await ExtractSourceValueTool(backend).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF,
                                   full_text="The actual EFT was 0.7 (0.6–0.9) cm."),
            field_type="eat_thickness", group="t1dm"))
    assert not out.found and out.value is None
    assert out.evidence_check == "protocol_error"
    assert "verbatim quote" in out.evidence_reason


def test_paper_excerpt_keeps_target_dense_late_table_under_size_cap():
    text = ("abstract and methods " * 1200
            + "\nTable 2\nEpicardial fat thickness EFT (cm) "
              "0.7 (0.6–0.9) 0.6 (0.5–0.7)\n"
            + "discussion " * 1200)
    excerpt = _paper_excerpt(
        text, target="epicardial adipose tissue thickness",
        raw_label="EAT thickness", field_type="eat_thickness",
        variants=["EFT", "epicardial fat thickness"])
    assert len(excerpt) <= 20000
    assert "abstract and methods" in excerpt
    assert "EFT (cm) 0.7" in excerpt


def _sample_size_payload(**overrides):
    sentence = "The study population was composed of 15 T1DM patients and 15 healthy controls."
    data = {
        "found": False, "value": None, "unit": "participants",
        "cohorts_seen": ["T1DM patients", "healthy controls"],
        "cohort_counts": [
            {"label": "T1DM patients", "count": 15, "quote": sentence},
            {"label": "healthy controls", "count": 15, "quote": sentence},
        ],
        "cohort_partition_complete": True,
        "cohort_partition_mutually_exclusive": True,
        "cohort_partition_quote": sentence,
        "cohort_partition_reason": "the study population is composed of the two arms",
        "not_found_reason": "no total was printed",
    }
    data.update(overrides)
    return sentence, data


@pytest.mark.asyncio
async def test_whole_study_sample_size_sums_only_explicit_complete_disjoint_arms():
    sentence, data = _sample_size_payload()
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=sentence),
            field_type="sample_size", group="-"))
    assert out.found and out.value == "30"
    assert out.value_origin == "derived_sum" and out.aggregation_status == "derived"
    assert out.derivation == "15 + 15 = 30 (T1DM patients; healthy controls)"
    assert [c.count for c in out.cohort_counts] == [15, 15]


@pytest.mark.asyncio
async def test_anchored_arm_quotes_and_true_partition_flags_do_not_require_a_third_quote():
    sentence, data = _sample_size_payload(
        cohort_partition_quote="15 in arm A ... and 15 in arm B")
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=sentence),
            field_type="sample_size", group="-"))
    assert out.value == "30" and out.aggregation_status == "derived"
    assert "no separate partition quote" in out.aggregation_reason


@pytest.mark.asyncio
@pytest.mark.parametrize("change, expected, status", [
    ({"cohort_partition_complete": False}, "not confirmed to cover", "rejected"),
    ({"cohort_partition_mutually_exclusive": False},
     "not confirmed to be mutually exclusive", "rejected"),
    ({"cohort_counts": [{"label": "T1DM patients", "count": 15,
                          "quote": "15 T1DM patients"}]},
     "fewer than two", "protocol_error"),
])
async def test_whole_study_sample_size_refuses_incomplete_overlap_or_missing_arm(
        change, expected, status):
    sentence, data = _sample_size_payload(**change)
    # Include the one-arm quote in the document for the missing-arm case.
    text = sentence + " 15 T1DM patients"
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=text),
            field_type="sample_size", group="-"))
    assert not out.found and out.value is None
    assert out.value_origin == "unresolved" and out.aggregation_status == status
    assert expected in out.aggregation_reason


@pytest.mark.asyncio
async def test_whole_study_sample_size_accepts_an_anchored_explicit_total():
    quote = "A total of 30 participants were enrolled."
    data = {"found": True, "value": "30", "unit": "participants",
            "quote": quote, "explicit_total_reported": True}
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=quote),
            field_type="sample_size", group="-"))
    assert out.found and out.value == "30" and out.value_origin == "verbatim"
    assert out.aggregation_status == "not_applicable"


@pytest.mark.asyncio
async def test_computed_total_masquerading_as_explicit_is_a_protocol_error():
    quote = "15 subjects with T1DM and 15 non-diabetic controls"
    data = {"found": True, "value": "30", "explicit_total_reported": True,
            "quote": quote, "cohorts_seen": ["T1DM", "controls"],
            "cohort_counts": []}
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=quote),
            field_type="sample_size", group="-"))
    assert not out.found and out.value is None
    assert out.aggregation_status == "protocol_error"
    assert "violated the sample-size contract" in out.aggregation_reason


@pytest.mark.asyncio
async def test_spelled_out_arm_count_and_pdf_line_wrap_are_anchored():
    paper = ("Thirty-six type 1 diabetic patients were included. The control "
             "group consisted of 43 healthy people. A total of 79 partici-\n"
             "pants were included.")
    data = {
        "found": False, "value": None, "cohorts_seen": ["DM", "CONTROL"],
        "cohort_counts": [
            {"label": "DM", "count": 36,
             "quote": "Thirty-six type 1 diabetic patients were included."},
            {"label": "control group", "count": 43,
             "quote": "The control group consisted of 43 healthy people."},
        ],
        "cohort_partition_complete": True,
        "cohort_partition_mutually_exclusive": True,
        "cohort_partition_quote": (
            "Thirty-six type 1 diabetic patients were included. The control "
            "group consisted of 43 healthy people."),
    }
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=paper),
            field_type="sample_size", group="-"))
    assert out.value == "79" and out.value_origin == "derived_sum"


@pytest.mark.asyncio
async def test_explicit_total_quote_survives_pdf_hyphenated_line_wraps():
    paper = ("A total of 97 participants were registered in the pres-\n"
             "ent study (63 diabetic patients and 34 age- and sex-\nmatched controls).")
    quote = ("A total of 97 participants were registered in the present study "
             "(63 diabetic patients and 34 age- and sex-matched controls).")
    data = {"found": True, "value": "97", "quote": quote,
            "explicit_total_reported": True,
            "cohorts_seen": ["diabetic patients", "controls"]}
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=paper),
            field_type="sample_size", group="-"))
    assert out.value == "97" and out.value_origin == "verbatim"


@pytest.mark.asyncio
async def test_spelled_out_single_cohort_total_is_accepted():
    quote = "Seventy-three patients agreed to participate and were evaluated."
    data = {"found": True, "value": "73", "unit": "count", "quote": quote,
            "explicit_total_reported": False, "cohorts_seen": ["patients"]}
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=quote),
            field_type="sample_size", group="-"))
    assert out.value == "73" and out.value_origin == "verbatim"


@pytest.mark.asyncio
async def test_direct_quote_survives_pdf_font_ligature_normalisation():
    paper = "The EFT was signiﬁcantly greater [0.7 (0.6–0.9) cm]."
    quote = "The EFT was significantly greater [0.7 (0.6–0.9) cm]."
    data = {"found": True, "value": "0.7 (0.6–0.9)", "unit": "cm",
            "quote": quote, "group_label_in_paper": "DM"}
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=paper),
            field_type="eat_thickness", group="t1dm"))
    assert out.found and out.value == "0.7 (0.6–0.9)"


@pytest.mark.asyncio
async def test_direct_value_rejects_a_unit_the_quote_does_not_print():
    quote = "Body mass index (kg/m3) 25.8 ± 3.9 25.5 ± 4.2"
    data = {"found": True, "value": "25.5 ± 4.2", "unit": "kg/m2",
            "quote": quote, "group_label_in_paper": "Controls"}
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=quote),
            field_type="bmi", group="control"))
    assert not out.found and out.evidence_check == "protocol_error"
    assert "unit" in out.evidence_reason


@pytest.mark.asyncio
async def test_single_cohort_anchored_count_does_not_depend_on_model_boolean():
    quote = "Finally, 72 patients underwent CCTA."
    data = {"found": True, "value": "72", "quote": quote,
            "explicit_total_reported": False,
            "cohorts_seen": ["T1DM patients"]}
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text=quote),
            field_type="sample_size", group="-"))
    assert out.value == "72" and out.value_origin == "verbatim"


@pytest.mark.asyncio
async def test_whole_study_sample_size_rejects_unanchored_model_counts():
    sentence, data = _sample_size_payload()
    out = await ExtractSourceValueTool(StubBackend(data)).run(
        ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF,
                                   full_text="This paper does not contain that sentence."),
            field_type="sample_size", group="-"))
    assert not out.found and "not anchored" in out.aggregation_reason


@pytest.mark.asyncio
async def test_extraction_recording_replays_raw_json_through_derivation(tmp_path):
    sentence, data = _sample_size_payload()
    path = tmp_path / "extract.json"
    cache = ExtractionCache(path)
    backend = StubBackend(data)
    payload = ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF, full_text=sentence),
        field_type="sample_size", group="-")
    recorded = await ExtractSourceValueTool(
        backend, cache=cache, cache_mode="record").run(payload)
    cache.save()
    body = json.loads(path.read_text(encoding="utf-8"))
    raw = next(iter(body["entries"].values()))
    assert raw["found"] is False and raw["value"] is None  # not the derived 30

    resume_backend = StubBackend({"found": True, "value": "999"})
    resumed = await ExtractSourceValueTool(
        resume_backend, cache=ExtractionCache(path), cache_mode="record").run(payload)
    assert resumed.value == "30" and resume_backend.calls == 0

    replayed = await ExtractSourceValueTool(
        None, cache=ExtractionCache(path), cache_mode="replay").run(payload)
    assert recorded.value == replayed.value == "30"
    assert recorded.derivation == replayed.derivation
    assert backend.calls == 1


@pytest.mark.asyncio
async def test_extraction_replay_miss_fails_loudly(tmp_path):
    cache = ExtractionCache(tmp_path / "missing.json")
    tool = ExtractSourceValueTool(None, cache=cache, cache_mode="replay")
    with pytest.raises(ExtractionCacheMiss):
        await tool.run(ExtractSourceValueInput(
            document=PaperDocument(paper_id="x", reference=_REF, full_text="x"),
            field_type="bmi", group="control"))


@pytest.mark.asyncio
async def test_collector_carries_derivation_and_does_not_retry_a_rejected_sum():
    sentence, data = _sample_size_payload(cohort_partition_complete=False)

    class _CountsRetriever(PaperRetriever):
        async def retrieve(self, reference):
            return PaperDocument(paper_id="counts", reference=reference,
                                 full_text=sentence)

    backend = StubBackend(data)
    review = ReviewDataItem(study_id="iacobellis_2014", group="-",
                            field_type="sample_size", value="30")
    res = await Collector(_catalogue(_CountsRetriever(), backend), max_attempts=3).collect(
        review, _REF)
    assert backend.calls == 1
    assert res.source_item.collection_outcome == CollectionOutcome.MISSING_SOURCE
    assert res.source_item.value_origin == "unresolved"
    assert res.source_item.aggregation_status == "rejected"
    assert res.source_item.cohort_counts[0].count == 15
    assert res.source_item.reasons[0].source == "deterministic"


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
        document=PaperDocument(paper_id="x", reference=_REF,
                               full_text="Age 12.90"),
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
    quote = "The mean age of the study population was 12.90 ± 1.30 years overall."
    backend = StubBackend({"found": True, "value": "12.90 ± 1.30", "unit": "years",
                           "group_label_in_paper": "Cohort B", "quote": quote})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(
            paper_id="x", reference=_REF,
            full_text=quote),
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
        document=PaperDocument(
            paper_id="x", reference=_REF,
            full_text="The mean age of the study population was 12.90 ± 1.30 years overall."),
        field_type="age", group="t1dm", cohorts={"t1dm": ["T1DM"],
                                                 "control": ["Control"]}))
    assert out.found is True and out.cohort_check == "ok"


@pytest.mark.asyncio
async def test_extract_tool_accepts_matching_cohort():
    backend = StubBackend({"found": True, "value": "12.96 ± 1.12", "unit": "years",
                           "group_label_in_paper": "Healthy controls",
                           "quote": "Controls 12.96 ± 1.12 years", "location": "Table 1"})
    out = await ExtractSourceValueTool(backend).run(ExtractSourceValueInput(
        document=PaperDocument(paper_id="x", reference=_REF,
                               full_text="Controls 12.96 ± 1.12 years"),
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
                           "quote": "EAT volume 52.3 cm3", "location": "y"})
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


# --- what KIND of target a claim is (Phase 7A) ---

def test_a_cell_that_repeats_the_cohorts_own_name_is_an_identity_claim():
    """Structural, not a vocabulary: no rule here knows what a drug is."""
    from react_review.agents.collector import _target_kind

    arm_row = ReviewDataItem(
        study_id="larkin_2015", group="ipilimumab_plus_placebo",
        field_type="treatment_arm", value="Ipilimumab (3 mg/kg) + placebo",
        cohort_label="Ipilimumab (3 mg/kg) + placebo")
    assert _target_kind(arm_row) == "arm_identity"

    count_row = arm_row.model_copy(update={"field_type": "cohort_n", "value": "315"})
    assert _target_kind(count_row) == "value"

    study_row = ReviewDataItem(study_id="larkin_2015", group="-",
                               field_type="sample_size", value="945")
    assert _target_kind(study_row) == "value"


def test_identity_detection_ignores_case_and_spacing():
    from react_review.agents.collector import _target_kind

    item = ReviewDataItem(
        study_id="s", group="g", field_type="treatment_arm",
        value="Nivolumab (3 mg/kg)  + placebo",
        cohort_label="nivolumab (3 mg/kg) + placebo")
    assert _target_kind(item) == "arm_identity"


# --- one paper, one retrieval (P8 D1-1) ---

class _CountingRetriever(PaperRetriever):
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, reference):
        self.calls += 1
        return PaperDocument(
            paper_id=reference.doi or "x", reference=reference,
            full_text="Table 1. Age (years) 12.90 ± 1.30 12.96 ± 1.12")


@pytest.mark.asyncio
async def test_a_study_is_fetched_once_however_many_claims_it_has():
    """Nine claims about one paper used to mean nine retrievals of it."""
    retriever = _CountingRetriever()
    backend = StubBackend({"found": True, "value": "12.96 ± 1.12", "unit": "years",
                           "quote": "Age (years) 12.90 ± 1.30 12.96 ± 1.12",
                           "source_field_name": "Age", "location": "Table 1"})
    collector = Collector(_catalogue(retriever, backend))

    source = await collector.open_study(_REF)
    for _ in range(5):
        await collector.collect(_REVIEW, _REF, source=source)
    assert retriever.calls == 1


@pytest.mark.asyncio
async def test_a_caller_that_passes_no_source_still_works():
    """The old signature keeps its old behaviour: opened for this claim alone."""
    retriever = _CountingRetriever()
    backend = StubBackend({"found": False, "not_found_reason": "not there"})
    collector = Collector(_catalogue(retriever, backend))
    await collector.collect(_REVIEW, _REF)
    assert retriever.calls == 1


@pytest.mark.asyncio
async def test_a_paper_that_cannot_be_retrieved_fails_every_claim_the_same_way():
    """One failed retrieval, not one per claim — and the same outcome for each."""
    collector = Collector(_catalogue(_NoneRetriever(), StubBackend({})))
    source = await collector.open_study(_REF)
    assert source.retrieved is False
    outcomes = []
    for _ in range(3):
        result = await collector.collect(_REVIEW, _REF, source=source)
        outcomes.append(result.source_item.collection_outcome)
    assert set(outcomes) == {CollectionOutcome.SOURCE_ACCESS_FAILED}


@pytest.mark.asyncio
async def test_an_unresolvable_citation_is_decided_once_for_the_study():
    from react_review.steps.paper_verification.schemas import ReferenceEntry

    collector = Collector(_catalogue(_CountingRetriever(), StubBackend({})))
    source = await collector.open_study(ReferenceEntry(title=""))
    assert source.outcome is CollectionOutcome.UNRESOLVED_SOURCE
    result = await collector.collect(_REVIEW, ReferenceEntry(title=""), source=source)
    assert result.source_item.collection_outcome is CollectionOutcome.UNRESOLVED_SOURCE

