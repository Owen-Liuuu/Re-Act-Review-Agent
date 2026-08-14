"""Claim-level evidence adequacy is deterministic and precedes comparison."""
from __future__ import annotations

import hashlib

import pytest

from react_review.agents.collector import Collector
from react_review.audit.evidence_adequacy import EvidenceAdequacyEvaluator
from react_review.normalize.cohorts import CohortLabel, CohortRegistry
from react_review.schemas.adequacy import AdequacyStatus, AxisStatus
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.base import Tool, ToolStage
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import SourceValueResult
from react_review.tools.registry import ToolRegistry


REF = ReferenceEntry(title="Evidence adequacy fixture", doi="10.1000/adequacy")
COHORTS = {
    "t1dm": ["T1DM"],
    "control": ["Control"],
}


def _document(text: str, scope: DocumentScope = DocumentScope.ABSTRACT_ONLY):
    return PaperDocument(
        paper_id="fixture", reference=REF, full_text=text,
        document_scope=scope,
    )


def _review(
    field: str, value: str, *, group: str = "t1dm", raw: str = "",
    timepoint: str = "single", timepoint_label: str = "", unit: str = "",
) -> ReviewDataItem:
    return ReviewDataItem(
        review_data_id="A_01", study_id="fixture", group=group,
        field_type=field, raw_field_name=raw or field.replace("_", " "),
        value=value, unit=unit, timepoint=timepoint,
        timepoint_label=timepoint_label,
    )


def _source(
    field: str, value: str, quote: str, *, group: str = "t1dm",
    scope: DocumentScope = DocumentScope.ABSTRACT_ONLY,
    cohorts: list[str] | None = None, unit: str = "",
    assigned_arm_label: str = "",
) -> SourceEvidenceItem:
    return SourceEvidenceItem(
        review_data_id="A_01", study_id="fixture", group=group,
        field_type=field, source_value=value, source_quote=quote,
        source_unit=unit, document_scope=scope,
        assigned_arm_label=assigned_arm_label,
        cohorts_seen=cohorts or [
            "subjects with type 1 diabetes", "non-diabetic controls"
        ],
    )


def _assess(review, source, document, *, field_variants=()):
    return EvidenceAdequacyEvaluator.development().assess(
        review, source, document,
        field_variants=list(field_variants), cohort_variants=COHORTS,
    )


def test_metadata_only_is_insufficient_without_inspecting_text():
    quote = "Age was 52.8 ± 12 years."
    result = _assess(
        _review("age", "52.8 ± 12", raw="Age"),
        _source("age", "52.8 ± 12", quote,
                scope=DocumentScope.METADATA_ONLY),
        _document(quote, DocumentScope.METADATA_ONLY),
    )

    assert result.status is AdequacyStatus.INSUFFICIENT
    assert result.document_scope is DocumentScope.METADATA_ONLY
    assert "metadata_only" in result.reason_codes


def test_unknown_document_scope_cannot_become_sufficient_from_text_alone():
    quote = "Age was 52.8 ± 12 years in the T1DM group."
    result = _assess(
        _review("age", "52.8 ± 12", raw="Age"),
        _source(
            "age", "52.8 ± 12", quote, scope=DocumentScope.UNKNOWN,
            cohorts=["T1DM", "Control"],
        ),
        _document(quote, DocumentScope.UNKNOWN),
    )

    assert result.status is AdequacyStatus.UNKNOWN
    assert result.document_scope is DocumentScope.UNKNOWN
    assert result.reason_codes == ["document_scope_unknown"]
    assert all(
        result.axis_results[axis].status is AxisStatus.UNKNOWN
        for axis in result.required_axes
    )


def test_abstract_can_be_sufficient_when_value_field_and_target_are_bound():
    quote = (
        "15 subjects with type 1 diabetes (age 52.8 ± 12, 10 females, "
        "5 males, BMI 27.8 ± 5.2) and 15 non-diabetic controls underwent "
        "echocardiographic measurement."
    )
    result = _assess(
        _review("age", "52.8 ± 12", raw="Age"),
        _source("age", "52.8 ± 12", quote),
        _document("TITLE\n\nABSTRACT\n" + quote),
    )

    assert result.status is AdequacyStatus.SUFFICIENT, result.model_dump(
        mode="json"
    )
    assert result.axis_results["value"].status is AxisStatus.PASS
    assert result.axis_results["field"].status is AxisStatus.PASS
    assert result.axis_results["target"].status is AxisStatus.PASS


def test_same_sentence_age_and_bmi_bind_to_their_own_field_phrases():
    quote = (
        "The type I diabetic patients were aged 30.6 ± 10 years and had "
        "BMI 27.8 ± 5.2; healthy controls were recruited separately."
    )
    document = _document(quote)

    age = _assess(
        _review("age", "30.6 ± 10", raw="Age"),
        _source(
            "age", "30.6 ± 10", quote,
            cohorts=["type I diabetic patients", "healthy controls"],
        ), document,
    )
    bmi = _assess(
        _review("bmi", "27.8 ± 5.2", raw="BMI"),
        _source(
            "bmi", "27.8 ± 5.2", quote,
            cohorts=["type I diabetic patients", "healthy controls"],
        ), document,
    )

    assert age.status is AdequacyStatus.SUFFICIENT
    assert bmi.status is AdequacyStatus.SUFFICIENT
    assert age.axis_results["field"].matched_phrases == ["aged"]
    assert bmi.axis_results["field"].matched_phrases == ["BMI"]


def test_age_value_cannot_support_a_bmi_claim_even_with_a_direct_quote():
    quote = (
        "Seventy-six type I diabetic patients were aged 30.6 ± 10 years; "
        "36 healthy controls were enrolled."
    )
    result = _assess(
        _review("bmi", "23.31 ± 2.73", raw="BMI"),
        _source("bmi", "30.6 ± 10", quote),
        _document(quote),
    )

    assert result.status is AdequacyStatus.INSUFFICIENT
    assert result.axis_results["field"].status is AxisStatus.FAIL
    assert "field_mismatch" in result.reason_codes


def test_t1dm_value_cannot_be_reused_for_control_claim():
    quote = (
        "15 subjects with type 1 diabetes (age 52.8 ± 12 and BMI 27.8 ± 5.2) "
        "and 15 non-diabetic controls underwent measurement."
    )
    result = _assess(
        _review("age", "53 ± 9", group="control", raw="Age"),
        _source("age", "52.8 ± 12", quote, group="control"),
        _document(quote),
    )

    assert result.status is AdequacyStatus.INSUFFICIENT
    assert result.axis_results["target"].status is AxisStatus.FAIL
    assert "target_mismatch" in result.reason_codes


def test_same_value_twice_without_unique_attribution_is_unknown():
    quote = (
        "Treatment age was 30 ± 5 years and control age was 30 ± 5 years."
    )
    result = _assess(
        _review("age", "30 ± 5", group="control", raw="Age"),
        _source(
            "age", "30 ± 5", quote, group="control",
            cohorts=["Treatment", "Control"],
        ),
        _document(quote),
    )

    assert result.status is AdequacyStatus.UNKNOWN
    assert result.axis_results["value"].status is AxisStatus.UNKNOWN
    assert "value_attribution_ambiguous" in result.reason_codes


def test_rounded_abstract_value_cannot_verify_more_precise_claim():
    quote = (
        "Thirty-six type 1 diabetic patients aged 31 ± 8 years and 43 healthy "
        "controls were included."
    )
    result = _assess(
        _review("age", "30.8 ± 7.7", raw="Age"),
        _source("age", "31 ± 8", quote),
        _document(quote),
    )

    assert result.status is AdequacyStatus.INSUFFICIENT
    assert result.axis_results["value"].status is AxisStatus.FAIL
    assert "source_value_coarser_than_claim" in result.reason_codes


@pytest.mark.parametrize(("group", "arm", "value"), [
    ("t1dm", "Type 1 diabetes", "25.8 ± 3.9"),
    ("control", "Controls", "25.5 ± 4.2"),
])
def test_verified_a010_a013_unit_differences_remain_comparable(
    group, arm, value,
):
    quote = f"{arm}\tBody mass index (kg/m3)\t{value}"
    document = _document(
        "Type 1 diabetes\tControls\n" + quote,
        DocumentScope.FULL_TEXT,
    )
    review = _review(
        "bmi", value, group=group, raw="BMI", unit="kg/m2",
    )
    source = _source(
        "bmi", value, quote, group=group, scope=DocumentScope.FULL_TEXT,
        cohorts=["Type 1 diabetes", "Controls"], unit="kg/m3",
        assigned_arm_label=arm,
    )

    result = _assess(
        review, source, document, field_variants=["Body mass index"]
    )

    assert result.reason_codes == []
    assert result.status is AdequacyStatus.SUFFICIENT, result.model_dump(
        mode="json"
    )
    assert result.axis_results["value"].status is AxisStatus.PASS
    assert result.axis_results["field"].status is AxisStatus.PASS
    assert result.axis_results["target"].status is AxisStatus.PASS
    assert review.unit == "kg/m2"
    assert source.source_unit == "kg/m3"


def test_timepoint_is_required_only_when_the_claim_states_one():
    quote = "At week 12 the treatment response was 42%."
    explicit = _assess(
        _review("response", "42%", group="-", raw="response",
                timepoint="week_12", timepoint_label="week 12"),
        _source("response", "42%", quote, group="-", cohorts=[]),
        _document(quote), field_variants=["response"],
    )
    unstated = _assess(
        _review("response", "42%", group="-", raw="response"),
        _source("response", "42%", quote, group="-", cohorts=[]),
        _document(quote), field_variants=["response"],
    )

    assert "timepoint" in explicit.required_axes
    assert explicit.axis_results["timepoint"].status is AxisStatus.PASS
    assert "timepoint" not in unstated.required_axes
    assert unstated.axis_results["timepoint"].status is AxisStatus.NOT_REQUIRED


def test_saved_anchors_include_offsets_context_phrases_and_document_hash():
    quote = "The control group had BMI 27.4 ± 4.1 kg/m2."
    text = "Methods paragraph.\n\n" + quote + "\n\nDiscussion paragraph."
    result = _assess(
        _review("bmi", "27.4 ± 4.1", group="control", raw="BMI"),
        _source(
            "bmi", "27.4 ± 4.1", quote, group="control",
            cohorts=["Treatment", "Control"],
        ),
        _document(text),
    )

    assert result.document_sha256 == hashlib.sha256(text.encode()).hexdigest().upper()
    kinds = {anchor.kind for anchor in result.evidence_anchors}
    assert {"quote", "value", "field", "target"} <= kinds
    assert all(0 <= anchor.start < anchor.end <= len(text)
               for anchor in result.evidence_anchors)
    assert all(anchor.context and len(anchor.context) <= 800
               for anchor in result.evidence_anchors)


def test_legacy_source_item_omits_new_empty_fields():
    item = SourceEvidenceItem(study_id="legacy", field_type="age")
    body = item.model_dump(mode="json")

    assert item.document_scope is DocumentScope.UNKNOWN
    assert item.evidence_adequacy is None
    assert "document_scope" not in body
    assert "evidence_adequacy" not in body


def test_registered_policy_and_evaluator_hashes_are_resolved_together():
    evaluator = EvidenceAdequacyEvaluator.resolve()
    identity = evaluator.identity

    assert identity.policy_id == "evidence_adequacy_v1"
    assert len(identity.policy_sha256) == 64
    assert identity.evaluator_id == "evidence_adequacy"
    assert identity.evaluator_version == "1.0.0"
    assert identity.evaluator_hash.startswith("sha256:")
    assert identity.evaluator_status in {"registered", "unregistered"}
    assert identity.release_eligible is (identity.evaluator_status == "registered")


@pytest.mark.asyncio
async def test_collector_persists_scope_and_adequacy_with_anchors():
    quote = "The control group had BMI 27.4 ± 4.1 kg/m2."
    document = _document(quote, DocumentScope.FULL_TEXT)

    class Retriever(PaperRetriever):
        async def retrieve(self, reference):
            return document.model_copy(update={"reference": reference})

    class Extract(Tool):
        name = "extract_source_value"
        stage = ToolStage.EXTRACT

        async def run(self, payload):
            return SourceValueResult(
                found=True, value="27.4 ± 4.1", unit="kg/m2", quote=quote,
                value_origin="verbatim", cohorts_seen=["Treatment", "Control"],
            )

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(Retriever()))
    registry.register(Extract())
    collector = Collector(
        registry,
        cohorts=CohortRegistry(labels=[
            CohortLabel(key="treatment", display="Treatment"),
            CohortLabel(key="control", display="Control"),
        ]),
        adequacy_evaluator=EvidenceAdequacyEvaluator.development(),
        max_attempts=1,
    )
    review = _review("bmi", "27.4 ± 4.1", group="control", raw="BMI")

    result = await collector.collect(review, REF)

    source = result.source_item
    assert source.document_scope is DocumentScope.FULL_TEXT
    assert source.evidence_adequacy is not None
    assert source.evidence_adequacy.status is AdequacyStatus.SUFFICIENT
    assert source.evidence_adequacy.evidence_anchors
