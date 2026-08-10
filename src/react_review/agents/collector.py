"""Agent 1 — Evidence Collector: review-driven directed source extraction.

Given ONE review claim (study, group, field_type), fetch the source paper's full
text and extract that specific value with a verbatim quote, producing a
:class:`SourceEvidenceItem`. A bounded, reflection-driven retry loop handles
"not found" (retry, then escalate); the whole trajectory is recorded as a
ProcessingRecord (AgentRun).

Design note: the fetched document (up to ~60k chars) is held in Python, not
passed through an LLM trajectory — so this Collector orchestrates the tools
deterministically at the document level while the LLM does the directed
extraction inside ``extract_source_value``. (Dual-LLM cross-checking needs a
second backend / ``llm2`` and is deferred.)
"""
from __future__ import annotations

from pydantic import BaseModel

from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.dkb import KnowledgeBase, evidence_contradicts
from react_review.normalize.cohorts import CohortRegistry, parse_comparison
from react_review.orchestrator.reflection import ReflectionDecider, ReflectionSignals
from react_review.schemas.reason import ReasonRecord
from react_review.study_match import is_resolvable
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract_source import ExtractSourceValueInput, SourceValueResult
from react_review.tools.extraction_profile import DEFAULT_PROFILE
from react_review.tools.extraction_profile import DEFAULT_PROFILE
from react_review.tools.registry import ToolRegistry
from react_review.tools.search import ResolveReferenceInput


def _target_kind(review_item: ReviewDataItem) -> str:
    """Whether this claim is ABOUT an arm, or a value reported FOR one.

    Structural, not a vocabulary: the review's cell for an arm-identity column
    IS the review's own name for that arm — the same text the cohort was
    labelled with. When the two coincide, the field being audited is the arm's
    identity, and its source answer is the paper's name for that arm rather
    than any number reported about it. Nothing here knows what a drug is.
    """
    value = _norm_text(review_item.value)
    label = _norm_text(review_item.cohort_label)
    return "arm_identity" if value and value == label else "value"


def _norm_text(value: object) -> str:
    return " ".join(str(value or "").lower().split())


class CollectResult(BaseModel):
    source_item: SourceEvidenceItem
    record: AgentRun
    decision: ReflectionDecision


# Where a retriever records the document's location. Each tier uses its own key,
# so they are normalised to one field rather than left for the reader to guess.
_URI_KEYS = ("path", "oa_url", "pdf_url", "url", "pmc_id", "pmid", "openalex_id")


def _provenance(document) -> dict[str, str]:
    """Which document was actually read — path or URL, plus the retriever tier."""
    if document is None:
        return {}
    meta = getattr(document, "metadata", None) or {}
    uri = next((str(meta[k]) for k in _URI_KEYS if meta.get(k)), "")
    reference = getattr(document, "reference", None)
    return {
        "source_file": str(meta.get("path") or ""),
        "source_uri": uri,
        "source_paper_id": str(getattr(document, "paper_id", "") or ""),
        "source_doi": str(getattr(reference, "doi", "") or ""),
        "retriever_kind": str(meta.get("source") or ""),
    }


def _reasons_for(result: SourceValueResult, outcome: CollectionOutcome) -> list[ReasonRecord]:
    """Everything that explains this outcome, in words a reader can act on."""
    reasons: list[ReasonRecord] = []
    if result.error:
        reasons.append(ReasonRecord(code="extraction_error", source="exception",
                                    stage="collector", message=result.error))
    if result.not_found_reason and outcome is not CollectionOutcome.FOUND:
        reason_source = ("deterministic" if (
                             result.aggregation_status in {"rejected", "protocol_error"}
                             or result.evidence_check == "protocol_error")
                         else "llm")
        reasons.append(ReasonRecord(code=outcome.value, source=reason_source,
                                    stage="collector", message=result.not_found_reason))
    if result.aggregation_status == "derived":
        reasons.append(ReasonRecord(
            code="derived_source_value", source="deterministic", stage="collector",
            message=result.derivation,
            detail={"cohort_counts": [c.model_dump() for c in result.cohort_counts]}))
    if result.evidence_check == "protocol_error" and result.evidence_reason:
        reasons.append(ReasonRecord(
            code="source_quote_unanchored", source="deterministic",
            stage="collector", message=result.evidence_reason))
    if result.target_check == "reassigned" and result.target_reason:
        # The wrong-arm selection this contract exists to catch, caught. Kept as
        # a visible record rather than silently corrected: a reader must be able
        # to see that the model's own answer was not the one used.
        reasons.append(ReasonRecord(
            code="target_reassigned", source="deterministic", stage="collector",
            message=result.target_reason,
            detail={"assigned_arm": result.assigned_arm_label}))
    elif result.target_check not in {"ok", ""} and result.target_reason:
        reasons.append(ReasonRecord(
            code=f"target_{result.target_check}", source="deterministic",
            stage="collector", message=result.target_reason))
    if (result.source_components is not None
            and result.source_components.status == "incomplete"):
        # The interval was in the evidence and did not come out. Recorded so the
        # partial answer cannot be read as a complete one.
        reasons.append(ReasonRecord(
            code="incomplete_source_components", source="deterministic",
            stage="collector", message=result.source_components.reason,
            detail={"missing": list(result.source_components.missing)}))
    if result.cohort_check == "ambiguous" and result.cohort_reason:
        # The value is kept, but nobody has confirmed which arm it belongs to.
        reasons.append(ReasonRecord(code="cohort_ambiguous", source="deterministic",
                                    stage="collector", message=result.cohort_reason))
    if result.cohorts_seen:
        reasons.append(ReasonRecord(
            code="cohorts_seen", source="llm", stage="collector",
            message="the paper distinguishes: " + ", ".join(result.cohorts_seen)))
    return reasons


class Collector:
    """Collect the source value for review claims via fetch + directed extract."""

    def __init__(
        self,
        catalogue: ToolRegistry,
        *,
        knowledge: KnowledgeBase | None = None,
        cohorts: CohortRegistry | None = None,
        decider: ReflectionDecider | None = None,
        max_attempts: int = 3,
        extraction_profile: str = DEFAULT_PROFILE,
    ) -> None:
        self._fetch = catalogue.get("fetch_fulltext")
        self._extract = catalogue.get("extract_source_value")
        # Optional: resolve a citation with no printed DOI to a gated DOI (online).
        self._resolve = catalogue.get("resolve_reference") if "resolve_reference" in catalogue else None
        self._kb = knowledge
        self._cohorts = cohorts
        # Which prompt contract every request runs under. Carried explicitly so
        # a run's answers are attributable to the profile that produced them.
        self._extraction_profile = extraction_profile
        self._extraction_profile = extraction_profile
        self._max_attempts = max(1, max_attempts)
        self._decider = decider or ReflectionDecider(max_attempts=self._max_attempts)

    def _concept_for(self, field_type: str) -> str:
        if self._kb and field_type in self._kb.entries:
            return self._kb.entries[field_type].concept
        return ""

    def _concept_variants_for(self, field_type: str) -> list[str]:
        if self._kb and field_type in self._kb.entries:
            return self._kb.entries[field_type].all_names()
        return []

    def _cohort_variants(self) -> dict[str, list[str]]:
        """key → the review's own words for that cohort (for the guard)."""
        if self._cohorts is None:
            return {}
        return {c.key: [c.display, *c.raw_variants] for c in self._cohorts.labels}

    async def collect(
        self,
        review_item: ReviewDataItem,
        reference: ReferenceEntry,
        *,
        research_context: str = "",
    ) -> CollectResult:
        steps: list[StepRecord] = []

        # A claim whose cohort could not be placed has nothing specific to look
        # for. Sending it anyway would ask the paper for an unnamed arm and read
        # back whatever is nearest — so it stops here, as its own outcome rather
        # than as MISSING_SOURCE, which would imply the paper omitted the value.
        if getattr(review_item, "cohort_status", "resolved") in ("unknown", "ambiguous"):
            reason = next((str(r) for r in getattr(review_item, "reasons", [])
                           if r.code.startswith("cohort")),
                          f"the cohort {review_item.cohort_label!r} could not be placed")
            return self._result(
                review_item, SourceValueResult(found=False, not_found_reason=reason),
                steps, ReflectionDecision.ESCALATE, CollectionOutcome.UNKNOWN_COHORT)

        # The review's reference list yielded nothing for this study, so there is
        # no citation to look up. Searching on the study id would just match some
        # unrelated paper; say so instead.
        if not is_resolvable(reference):
            return self._result(
                review_item,
                SourceValueResult(found=False, not_found_reason=(
                    "the review's reference list contains no citation for this "
                    "study, so the source paper could not be identified")),
                steps, ReflectionDecision.ESCALATE,
                CollectionOutcome.UNRESOLVED_SOURCE)

        # 0. No printed DOI → resolve the citation to a GATED DOI online before
        # fetching. A low-confidence / no match is UNRESOLVED_SOURCE (we refuse
        # to fetch a wrong paper). References that already carry a DOI skip this.
        if self._resolve is not None and not (reference.doi or "").strip():
            rr = await self._resolve.run(ResolveReferenceInput(
                title=reference.title, authors=reference.authors,
                year=reference.year, journal=reference.journal))
            steps.append(StepRecord(
                index=len(steps), thought="resolve DOI from citation",
                tool="resolve_reference",
                observation={"status": rr.status, "doi": rr.doi,
                             "confidence": round(rr.confidence, 3), "source": rr.source},
            ))
            if rr.status == "resolved" and rr.doi:
                reference = reference.model_copy(update={"doi": rr.doi})
            else:
                return self._result(review_item, SourceValueResult(found=False), steps,
                                    ReflectionDecision.ESCALATE,
                                    CollectionOutcome.UNRESOLVED_SOURCE)

        # 1. Fetch the source paper (deterministic; document held in Python).
        fetched = await self._fetch.run(reference)
        provenance = _provenance(fetched.document)
        steps.append(StepRecord(
            index=len(steps), thought="fetch source full text", tool="fetch_fulltext",
            args={"doi": reference.doi or "", "title": reference.title[:60]},
            observation={"retrieved": fetched.retrieved, **provenance},
        ))
        if not fetched.retrieved or fetched.document is None:
            out = self._decider.decide(ReflectionSignals(retrieval_ok=False, attempt=0))
            return self._result(
                review_item,
                SourceValueResult(found=False, not_found_reason=out.reason),
                steps, out.decision, CollectionOutcome.SOURCE_ACCESS_FAILED,
                provenance=provenance)

        # 2. Directed extraction, with a bounded reflection-driven retry loop.
        concept = self._concept_for(review_item.field_type)
        result = SourceValueResult(found=False)
        decision = ReflectionDecision.ESCALATE
        reflection_reason = ""
        for attempt in range(self._max_attempts):
            result = await self._extract.run(ExtractSourceValueInput(
                document=fetched.document,
                field_type=review_item.field_type,
                group=review_item.group,
                concept=concept,
                concept_variants=self._concept_variants_for(review_item.field_type),
                raw_field_name=review_item.raw_field_name,
                unit_hint=review_item.unit,
                research_context=research_context,
                cohort_display=review_item.cohort_label,
                cohorts=self._cohort_variants(),
                target_kind=_target_kind(review_item),
                extraction_profile=self._extraction_profile,
                attempt=attempt,
                # A claim about two arms is carried as a pair, not as one name:
                # "A vs B" handed over as a single cohort string is what let the
                # extractor answer with whichever hazard ratio it met first.
                comparison=parse_comparison(review_item.group),
            ))
            steps.append(StepRecord(
                index=len(steps),
                thought=f"extract {review_item.field_type} for {review_item.group} "
                        f"(attempt {attempt + 1})",
                tool="extract_source_value",
                args={"field_type": review_item.field_type, "group": review_item.group},
                observation=result.model_dump(),
            ))
            # Reflection: a found value is accepted; otherwise retry then escalate.
            # Keep the decider's REASON — it was being generated and discarded,
            # so "retries exhausted" never reached anyone reading the report.
            outcome_of = self._decider.decide(
                ReflectionSignals(retrieval_ok=result.found, attempt=attempt))
            decision, reflection_reason = outcome_of.decision, outcome_of.reason
            # Deterministic rejection will not improve by repeating the same
            # extraction.  Preserve it and escalate without burning attempts.
            deterministic_rejection = (
                result.wrong_group_rejected or result.aggregation_status == "rejected"
                # An unresolvable target does not become resolvable by asking
                # the same question again; the arms the paper reports are what
                # they are. Retrying would only spend attempts to re-derive the
                # same refusal.
                or result.target_check in {"ambiguous", "not_reported",
                                           "direction_inverted", "unsupported",
                                           "inconsistent"})
            if deterministic_rejection:
                decision = ReflectionDecision.ESCALATE
            if result.found or deterministic_rejection or decision != ReflectionDecision.RETRY:
                break

        # Retrieved but the value was never located → potential fabrication.
        outcome = (CollectionOutcome.FOUND if result.found
                   else CollectionOutcome.MISSING_SOURCE)
        if not result.found and not result.not_found_reason and reflection_reason:
            result = result.model_copy(update={"not_found_reason": reflection_reason})

        # Back-check: for a CANDIDATE (LLM-guessed) concept, let the SOURCE
        # evidence validate the translation — if the extracted unit/value is a
        # different kind or out of range for the guessed field_type, the guess
        # was likely wrong (extraction refutes the parse-time hypothesis).
        mismatch_reason = ""
        if (result.found and self._kb is not None
                and getattr(review_item, "resolution_status", "resolved") == "candidate"):
            mismatch_reason = evidence_contradicts(
                review_item.field_type, kb=self._kb,
                unit=result.unit, value=result.value)
        return self._result(review_item, result, steps, decision, outcome,
                            mismatch_reason=mismatch_reason, provenance=provenance)

    def _result(
        self,
        review_item: ReviewDataItem,
        result: SourceValueResult,
        steps: list[StepRecord],
        decision: ReflectionDecision,
        outcome: CollectionOutcome = CollectionOutcome.FOUND,
        mismatch_reason: str = "",
        provenance: dict[str, str] | None = None,
    ) -> CollectResult:
        source_item = SourceEvidenceItem(
            study_id=review_item.study_id,
            group=review_item.group,
            timepoint=review_item.timepoint,
            field_type=review_item.field_type,
            # Carry the review cell forward so the join can tell two claims on
            # the same study/cohort/field apart instead of guessing.
            table_id=review_item.table_id,
            cell_ref=review_item.cell_ref,
            checklist_id=review_item.checklist_id,
            source_value=result.value,
            source_unit=result.unit,
            source_quote=result.quote,
            source_location_in_paper=result.location,
            value_origin=result.value_origin,
            derivation=result.derivation,
            source_components=result.source_components,
            population_scope=result.source_scope,
            cohort_counts=result.cohort_counts,
            aggregation_status=result.aggregation_status,
            aggregation_reason=result.aggregation_reason,
            evidence_check=result.evidence_check,
            evidence_reason=result.evidence_reason,
            collection_outcome=outcome,
            concept_mismatch=bool(mismatch_reason),
            concept_mismatch_reason=mismatch_reason,
            cohort_check=result.cohort_check,
            cohorts_seen=result.cohorts_seen,
            target_check=result.target_check,
            target_reason=result.target_reason,
            assigned_arm_label=result.assigned_arm_label,
            reasons=_reasons_for(result, outcome),
            **(provenance or {}),
        )
        record = AgentRun(
            agent="collector",
            task={"study_id": review_item.study_id, "group": review_item.group,
                  "field_type": review_item.field_type,
                  "checklist_id": review_item.checklist_id},
            steps=steps,
            status="finished",
            final=source_item.model_dump(mode="json"),
        )
        return CollectResult(source_item=source_item, record=record, decision=decision)
