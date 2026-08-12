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

import hashlib
import json

from pydantic import BaseModel

from react_review.contracts import ContractError
from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.dkb import KnowledgeBase, evidence_contradicts
from react_review.normalize.cohorts import CohortRegistry, parse_comparison
from react_review.orchestrator.reflection import ReflectionDecider, ReflectionSignals
from react_review.schemas.reason import ReasonRecord
from react_review.study_match import is_resolvable
from react_review.schemas.agent import AgentRun, StepRecord
from react_review.schemas.evidence import ReviewDataItem, SourceEvidenceItem
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.schemas.batch import ClaimBinding, ProjectionContract
from react_review.schemas.evidence import BatchProjectionProvenance
from react_review.tools.batch_group import (
    claim_kind,
    execution_id_for,
    group_claims,
    question_id_for,
)
from react_review.tools.batch_outcome import outcome_for
from react_review.tools.batch_parse import BatchReading
from react_review.tools.batch_project import project_claim
from react_review.tools.batch_prompt import aggregation_applies
from react_review.tools.batch_result import to_source_result
from react_review.tools.extract_batch import prompt_sha256
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    SourceValueResult,
    _paper_excerpt,
)
from react_review.tools.extraction_profile import (
    BATCH_PROFILE_NAME,
    DEFAULT_PROFILE,
    prompt_version,
)
from react_review.tools.registry import ToolRegistry
from react_review.tools.search import ResolveReferenceInput


#: Kept under its old name; the rule now lives with the grouping, so a claim
#: cannot be routed one way and grouped another.
_target_kind = claim_kind


def _document_sha256(text: str) -> str:
    """Which document was read. Line endings normalised, so a checkout
    cannot change the question a recording answers."""
    canonical = chr(10).join((text or "").splitlines()).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def _refuse_repeated_identities(claims) -> None:
    """Two claims may not share an identity.

    Everything downstream refers to a claim by that identity — the binding in an
    execution id, the reference on a row, the claim list on a reading. A
    duplicate would make those references ambiguous in an artifact nobody can
    re-run, so it is refused here rather than resolved by a rule nobody agreed.
    """
    seen: dict[str, int] = {}
    for position, claim in enumerate(claims):
        identity = _claim_id(claim)
        if identity in seen:
            raise ContractError(
                f"claims {seen[identity]} and {position} share the identity "
                f"{identity!r}. Every reference in a batched artifact names a "
                "claim by its identity, so two claims with one identity make "
                "every one of those references ambiguous")
        seen[identity] = position


def _claim_id(claim) -> str:
    """A stable name for one claim, preferring the one the review already gave.

    `review_data_id` is the review's own identity for the cell and is what
    production has. Falling back to the full locator keeps a hand-built or
    CSV-loaded claim identifiable; position in a list is never used, because a
    list that is grouped and regrouped does not preserve one.
    """
    declared = str(getattr(claim, "review_data_id", "") or "")
    if declared:
        return declared
    return "|".join(str(part) for part in (
        claim.study_id, claim.group, getattr(claim, "timepoint", "single"),
        claim.field_type, getattr(claim, "table_id", "") or "",
        getattr(claim, "cell_ref", None) or "",
        getattr(claim, "checklist_id", "") or ""))


class StudySource(BaseModel):
    """One paper, opened once.

    A study's claims used to be collected one at a time, each re-resolving the
    citation and re-fetching the same PDF: nine claims about one paper meant
    nine retrievals of it. Opening the study once and passing this around makes
    the cost proportional to papers rather than to cells — and, more quietly,
    guarantees every claim about a study is read from the SAME document.
    """

    reference: ReferenceEntry
    document: object | None = None
    retrieved: bool = False
    reason: str = ""                       # why there is no document, if there is none
    outcome: CollectionOutcome | None = None
    provenance: dict[str, str] = {}
    steps: list[StepRecord] = []

    model_config = {"arbitrary_types_allowed": True}


class CollectResult(BaseModel):
    source_item: SourceEvidenceItem
    record: AgentRun
    decision: ReflectionDecision


class CollectStudyResult(BaseModel):
    """One study's claims, and the readings they came out of.

    Two lists rather than one, because a batched run produces two KINDS of
    thing. The claim results are per claim, as they always were. The batch
    records are per reading, and there are fewer of them — that is the whole
    point. Copying a response onto every claim it answered would multiply it by
    the group size and still not show that the group shared it; referencing it
    by execution id does both.
    """

    claim_results: list[CollectResult] = []
    batch_records: list[object] = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def source_items(self) -> list[SourceEvidenceItem]:
        return [r.source_item for r in self.claim_results]

    @property
    def records(self) -> list[AgentRun]:
        return [r.record for r in self.claim_results]


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
        # The RUN CONTRACT, when there is one. It carries the route for each
        # kind of claim and the axes each field is held to, so the Collector
        # never has to be told the same thing twice in two shapes.
        contract=None,
        # Resolved ONCE by the harness that builds this Collector, never here:
        # readiness shells out to git, and a check that runs per claim is a
        # check somebody eventually removes for being slow.
        aggregation_runtime=None,
        batch_tool=None,
        knowledge_fingerprint: str = "",
        telemetry=None,
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
        self._contract = contract
        self._runtime = aggregation_runtime
        self._batch = batch_tool or (
            catalogue.get("extract_source_batch")
            if "extract_source_batch" in catalogue else None)
        # Recorded on a batch question. The concept and its variants are in
        # there too, so this adds attribution rather than identity.
        self._knowledge_fingerprint = knowledge_fingerprint
        # Counts retrievals, so "one paper, one fetch" is a measured claim.
        self._telemetry = telemetry
        self._max_attempts = max(1, max_attempts)
        self._decider = decider or ReflectionDecider(max_attempts=self._max_attempts)

    def route_for(self, review_item) -> str:
        """The extraction contract THIS claim runs under.

        A run may legitimately read values in batch and arm identities one at a
        time. What it may not do is leave that undeclared, so the route comes
        from the contract when there is one and from the single profile when
        there is not — never from a guess about what the claim looks like.
        """
        if self._contract is None:
            return self._extraction_profile
        return self._contract.route_for(claim_kind(review_item))

    def _axes_for(self, field_type: str) -> list[str]:
        if self._contract is None or not self._contract.scope_enabled:
            return []
        return self._contract.axes_for(field_type)

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

    async def open_study(self, reference: ReferenceEntry) -> StudySource:
        """Resolve the citation and fetch the paper — once per study.

        Everything here used to happen per claim. It depends only on the study,
        so doing it per claim was pure repetition, and it meant two claims about
        one paper could in principle be answered from two different retrievals.
        """
        steps: list[StepRecord] = []

        if not is_resolvable(reference):
            return StudySource(
                reference=reference, retrieved=False,
                outcome=CollectionOutcome.UNRESOLVED_SOURCE,
                reason=("the review's reference list contains no citation for "
                        "this study, so the source paper could not be identified"),
                steps=steps)

        # No printed DOI → resolve the citation to a GATED DOI online before
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
                return StudySource(
                    reference=reference, retrieved=False,
                    outcome=CollectionOutcome.UNRESOLVED_SOURCE, steps=steps)

        fetched = await self._fetch.run(reference)
        if self._telemetry is not None:
            self._telemetry.attempt("fetch_fulltext")
        provenance = _provenance(fetched.document)
        steps.append(StepRecord(
            index=len(steps), thought="fetch source full text", tool="fetch_fulltext",
            args={"doi": reference.doi or "", "title": reference.title[:60]},
            observation={"retrieved": fetched.retrieved, **provenance},
        ))
        if not fetched.retrieved or fetched.document is None:
            out = self._decider.decide(ReflectionSignals(retrieval_ok=False, attempt=0))
            return StudySource(reference=reference, retrieved=False,
                               outcome=CollectionOutcome.SOURCE_ACCESS_FAILED,
                               reason=out.reason, provenance=provenance, steps=steps)
        return StudySource(reference=reference, document=fetched.document,
                           retrieved=True, provenance=provenance, steps=steps)

    async def collect_study(
        self,
        claims: list[ReviewDataItem],
        reference: ReferenceEntry,
        *,
        research_context: str = "",
        source: StudySource | None = None,
    ) -> CollectStudyResult:
        """Every claim about one paper, each under the route its contract names.

        A mixed contract is the normal case, not an edge: values are worth
        batching because one reading answers several of them, and arm identities
        are not, because there is one per arm and asking "what is this arm
        called" alongside "what does it report" needs a different prompt. Both
        happen here, in one pass over one document, and each claim records which
        route actually read it — a run-level profile would describe half the
        run and leave the other half unattributable.

        Claims come back in the order they arrived. Nothing downstream should
        have to know that they were reordered to be grouped.
        """
        if source is None:
            source = await self.open_study(reference)

        _refuse_repeated_identities(claims)
        results: dict[int, CollectResult] = {}
        records: list[object] = []
        for group in group_claims(claims):
            route = (self._contract.route_for(group.kind) if self._contract
                     else self._extraction_profile)
            positions = group.positions
            if route == BATCH_PROFILE_NAME:
                produced, record = await self._collect_batched(
                    group, reference, source, research_context)
                if record is not None:
                    records.append(record)
            else:
                produced = [
                    await self.collect(claim, reference,
                                       research_context=research_context,
                                       source=source, route=route)
                    for claim in group.claims]
            for position, result in zip(positions, produced):
                results[position] = result
        return CollectStudyResult(
            claim_results=[results[i] for i in sorted(results)],
            batch_records=records)

    async def _collect_batched(self, group, reference, source, research_context):
        """One reading for a whole group, and one answer per claim out of it.

        Deliberately explicit rather than quietly falling through to the
        single-target path: a v5 group read one claim at a time would put half a
        run's answers under a profile the artifact does not name, and would make
        the cost of batching unmeasurable, since every fallback adds back the
        calls the batch was supposed to save.
        """
        if self._batch is None:
            raise ContractError(
                f"{group.describe()} is routed to {BATCH_PROFILE_NAME} and this "
                "Collector has no batch tool. A run that cannot honour its own "
                "contract must stop rather than read the claims some other way")

        if not source.retrieved or source.document is None:
            # No document, so no reading. Each claim gets the retrieval failure
            # it would have got one at a time — a paper nobody could fetch is
            # not a batch that failed.
            return [self._unretrieved(claim, source) for claim in group.claims], None

        field_type = group.key.field_type
        concept = self._concept_for(field_type)
        variants = self._concept_variants_for(field_type)
        target = concept or group.key.raw_field_name or field_type
        text = getattr(source.document, "full_text", "") or ""
        excerpt = _paper_excerpt(text, target=target,
                                 raw_label=group.key.raw_field_name,
                                 field_type=field_type, variants=variants)
        prompt = self._batch.build_prompt(
            target_shape=group.shape, field_type=field_type, concept=target,
            raw_label=group.key.raw_field_name or target,
            concept_variants=", ".join(variants) or target,
            unit_hint=group.key.unit_signature, paper_text=excerpt,
            research_context=research_context,
            timepoint_label=group.key.timepoint_label)

        question = question_id_for(
            group, concept=target, concept_variants=variants,
            research_context=research_context,
            document_sha256=_document_sha256(text),
            knowledge_fingerprint=self._knowledge_fingerprint,
            prompt_version=prompt_version(BATCH_PROFILE_NAME),
            prompt_sha256=prompt_sha256(prompt),
            aggregable=aggregation_applies(group.shape, field_type))
        record = await self._batch.read(question=question, prompt=prompt,
                                        document=excerpt)

        bindings = [self._binding_for(claim, group) for claim in group.claims]
        record.execution = execution_id_for(question, bindings,
                                            self._projection_contract())
        if self._telemetry is not None:
            # What batching actually did. A run that issued one prompt for four
            # claims and a run that issued four look identical in the backend's
            # totals, so the cost argument has to be counted where it happens.
            self._telemetry.record_batch(claims=len(group.claims),
                                         failed=record.reading is None)
        produced = [self._project_one(claim, group, record, source, binding)
                    for claim, binding in zip(group.claims, bindings)]
        return produced, record

    def _binding_for(self, claim, group) -> ClaimBinding:
        scope = getattr(claim, "population_scope", None)
        return ClaimBinding(
            claim_id=_claim_id(claim), target=str(claim.group or ""),
            requested_scope=(scope.describe() if scope is not None else ""),
            route=BATCH_PROFILE_NAME,
            required_axes=tuple(self._axes_for(claim.field_type)),
            timepoint_label=str(getattr(claim, "timepoint_label", "") or ""))

    def _projection_contract(self) -> ProjectionContract:
        """The rules this response will be READ under, as an identity.

        A recording is about words sent to a model; an answer is about those
        words read under a set of rules. Two runs sharing the recording and
        differing here produced different answers, and must not share
        provenance with each other.
        """
        contract, runtime = self._contract, self._runtime
        policy = runtime.policy if runtime is not None else None
        who = runtime.evaluator if runtime is not None else None
        return ProjectionContract(
            run_profile_sha256=(contract.sha256 if contract is not None else ""),
            population_contract_sha256=(
                contract.population_contract_sha256 if contract is not None else ""),
            cohort_fingerprint=self._cohort_fingerprint(),
            aggregation_policy_id=(policy.policy_id if policy else ""),
            aggregation_policy_sha256=(policy.sha256 if policy else ""),
            evaluator_id=(who.evaluator_id if who else ""),
            evaluator_version=(who.evaluator_version if who else ""),
            evaluator_hash=(who.evaluator_hash if who else ""))

    def _cohort_fingerprint(self) -> str:
        """The review-label mapping in force, which decides which arm is which."""
        labels = self._cohort_labels()
        if not labels:
            return ""
        body = json.dumps(labels, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
        return hashlib.sha256(body.encode("utf-8")).hexdigest().upper()[:32]

    def _cohort_labels(self) -> dict[str, str]:
        """Each cohort key mapped to the review's OWN words for that cohort."""
        return {key: (variants[0] if variants else key)
                for key, variants in self._cohort_variants().items()}

    def _project_one(self, claim, group, record, source, binding) -> CollectResult:
        """One claim's answer out of the shared reading, or why there is none."""
        steps = [StepRecord(
            index=0,
            thought=f"read {group.describe()} in one batch "
                    f"({len(record.attempts)} attempt(s))",
            tool="extract_source_batch",
            args={"execution": record.execution_id[:16],
                  "claims": len(group.claims)},
            observation={"summary": record.summary()})]

        if record.reading is None:
            reading = BatchReading(batch_error=record.detail or record.failure)
            projection = project_claim(reading, target_shape=group.shape,
                                       field_type=group.key.field_type)
        else:
            reading = record.reading
            scope = getattr(claim, "population_scope", None)
            projection = project_claim(
                reading, target_shape=group.shape,
                review_labels=self._cohort_labels(),
                cohort_key=str(claim.group or ""),
                comparison=parse_comparison(claim.group),
                requested_scope=scope,
                required_axes=list(binding.required_axes),
                timepoint_label=binding.timepoint_label,
                runtime=self._runtime, field_type=group.key.field_type)

        result = to_source_result(projection)
        outcome = outcome_for(projection, reading)
        if self._telemetry is not None:
            self._telemetry.record_projection(projection.status,
                                              projection.aggregation_status)
        entry = projection.entry
        provenance = BatchProjectionProvenance(
            batch_question_id=record.question.identity(),
            batch_execution_id=record.execution_id,
            claim_group_key=group.key.key(),
            claim_id=binding.claim_id,
            selected_entry_id=(entry.entry_id(record.question.identity())
                               if entry is not None else ""),
            projection_status=projection.status,
            projection_reason=projection.reason,
            route=BATCH_PROFILE_NAME,
            attempts=len(record.attempts),
            served_from_cache=record.served_from_cache)
        return self._result(claim, result, steps, ReflectionDecision.ESCALATE,
                            outcome, provenance=source.provenance,
                            batch_provenance=provenance)

    def _unretrieved(self, claim, source) -> CollectResult:
        outcome = source.outcome or CollectionOutcome.SOURCE_ACCESS_FAILED
        return self._result(
            claim, SourceValueResult(found=False, not_found_reason=source.reason),
            list(source.steps), ReflectionDecision.ESCALATE, outcome,
            provenance=source.provenance)

    async def collect(
        self,
        review_item: ReviewDataItem,
        reference: ReferenceEntry,
        *,
        research_context: str = "",
        source: StudySource | None = None,
        route: str = "",
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

        # The paper is opened once per study. Callers that do not pass one get
        # the old behaviour — opened here, for this claim alone.
        if source is None:
            source = await self.open_study(reference)
        steps.extend(source.steps)
        reference = source.reference

        provenance = source.provenance
        if not source.retrieved or source.document is None:
            outcome = source.outcome or CollectionOutcome.SOURCE_ACCESS_FAILED
            decision = (ReflectionDecision.ESCALATE
                        if outcome is CollectionOutcome.UNRESOLVED_SOURCE
                        else self._decider.decide(
                            ReflectionSignals(retrieval_ok=False, attempt=0)).decision)
            return self._result(
                review_item,
                SourceValueResult(found=False, not_found_reason=source.reason),
                steps, decision, outcome, provenance=provenance)

        # 2. Directed extraction, with a bounded reflection-driven retry loop.
        concept = self._concept_for(review_item.field_type)
        result = SourceValueResult(found=False)
        decision = ReflectionDecision.ESCALATE
        reflection_reason = ""
        for attempt in range(self._max_attempts):
            result = await self._extract.run(ExtractSourceValueInput(
                document=source.document,
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
                extraction_profile=(route or self.route_for(review_item)),
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
        batch_provenance=None,
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
            batch_provenance=batch_provenance,
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
