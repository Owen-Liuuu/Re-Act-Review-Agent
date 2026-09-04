"""Production collector: batch-split routing, field mapping, inter-group concurrency.

``collector.py`` is inside the ``evidence_adequacy`` hash boundary (1.0.0
and 1.1.0). Split routing, the ``source_field_mapping`` reason, and
group-level concurrency live here so v8 contracts that pin 1.0.0 still
load against that exact collector module as it was when 1.0.0 was
published — except 1.1.0 now also copies PMC tables onto the document
and classifies ``missing_source``. New behaviour that is not part of
that hash stays in this wrapper.

``source_field_mapping`` is human-readable only. It lets a reader check
综述叫法 → 论文叫法. Downstream must not parse the message (or scrape ``code``)
to recover a structured ``source_field_name`` — statistics, field alignment,
anything that branches on the paper's label. That needs evaluator 1.8.3 and a
real field on ``SourceEvidenceItem``. ``detail`` is left empty on purpose so
this cannot be used as a back-door structured slot.
"""
from __future__ import annotations

import asyncio
import re
import unicodedata

from react_review.agents.collector import (
    Collector,
    CollectResult,
    CollectStudyResult,
    _claim_id,
    _document_sha256,
    _refuse_repeated_identities,
)
from react_review.contracts import ContractError
from react_review.core.enums import CollectionOutcome, ReflectionDecision
from react_review.core.exceptions import PermanentProviderError
from react_review.schemas.batch import (
    ClaimBinding,
    ExcerptProvenance,
)
from react_review.schemas.reason import ReasonRecord
from react_review.tools.batch_group import (
    execution_id_for,
    group_claims,
    question_id_for,
)
from react_review.tools.batch_prompt import aggregation_applies
from react_review.tools.batch_split import build_batch_locate_prompt
from react_review.tools.extract_batch import prompt_sha256
from react_review.tools.extract_source import (
    SELECTION_METHOD_ID,
    SELECTION_VERSION,
    TABLE_SELECTION_METHOD_ID,
    TABLE_SELECTION_VERSION,
    SourceValueResult,
    select_excerpt,
)
from react_review.tools.extraction_profile import (
    BATCH_PROFILE_NAME,
    BATCH_SPLIT_PROFILE,
    is_batch_route,
    prompt_version,
)


def mapping_reason(review_name: str, paper_label: str) -> ReasonRecord:
    """A reader-facing 综述叫法 → 论文叫法 note. Not a structured field."""
    name = review_name or "(unresolved review field)"
    return ReasonRecord(
        code="source_field_mapping", source="llm", stage="collector",
        message=(
            f"the review column {name!r} was read from the paper's "
            f"own label {paper_label!r} (human check only; not a "
            "structured field)"))


def _fold_mention(text: str) -> str:
    """Lowercase, strip accents, collapse punctuation — for abstract precheck."""
    folded = unicodedata.normalize("NFKD", text or "").lower()
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", folded).strip()


def _concept_names(*, field_type: str = "", raw_field_name: str = "",
                   concept: str = "", variants=None) -> list[str]:
    names: list[str] = []
    for item in (field_type, raw_field_name, concept, *(variants or [])):
        text = str(item or "").strip()
        if text and text not in names:
            names.append(text)
    return names


def abstract_omits_concept(document, names: list[str]) -> str:
    """Why this abstract cannot answer the claim, or empty to proceed.

    Fail-closed: refuse only when every checked form is absent. An empty
    name list, an unknown/full-text scope, or any single hit all mean
    "ask the model as usual".
    """
    if document is None:
        return ""
    scope = getattr(document, "document_scope", None)
    scope_value = str(getattr(scope, "value", scope) or "")
    if scope_value != "abstract_only":
        return ""
    folded_text = _fold_mention(getattr(document, "full_text", "") or "")
    checked: list[tuple[str, str]] = []
    for name in names:
        folded = _fold_mention(name)
        if folded:
            checked.append((name, folded))
    if not checked:
        return ""
    if any(folded in folded_text for _, folded in checked):
        return ""
    listed = ", ".join(repr(name) for name, _ in checked)
    return (
        "the abstract does not mention this concept in any form "
        f"(checked: {listed})"
    )


def _unique_table_excerpt(document, *, group, field_type, concept, variants,
                          cohorts=None) -> str:
    """One unique matching table as TSV, or empty to keep the 20k excerpt."""
    tables = list(getattr(document, "tables", None) or [])
    if not tables:
        return ""
    from react_review.tools.source_table_lookup import (
        locate_source_table,
        render_table_prompt,
    )

    claim = group.claims[0] if group.claims else None
    hit = locate_source_table(
        tables,
        group=getattr(claim, "group", "") or "",
        cohort_display=getattr(claim, "cohort_label", "") or "",
        cohorts=cohorts or {},
        raw_field_name=group.key.raw_field_name,
        concept_variants=list(variants or []),
        field_type=field_type,
        column_header=getattr(claim, "column_header", "") or "",
        outcome=getattr(claim, "outcome", "") or "",
    )
    if not hit.unique or hit.table is None:
        return ""
    return render_table_prompt(hit.table) or ""


class SplitAwareCollector(Collector):
    """Collector that can honour ``batch_split_v1`` without editing collector.py."""

    async def collect_study(
        self,
        claims,
        reference,
        *,
        research_context: str = "",
        source=None,
    ) -> CollectStudyResult:
        """Every group of one paper, overlapping. Studies stay serial upstream
        unless ``--checkpoints none``.

        Concurrency is between groups, and between papers only when nobody is
        gating: HITL shows one study block at a time. The backend semaphore is
        the cap; nothing here adds another. Results are reassembled by
        ``positions``, never by gather order.
        """
        if source is None:
            source = await self.open_study(reference)
        _refuse_repeated_identities(claims)
        groups = list(group_claims(claims))
        gathered = await asyncio.gather(
            *[self._run_group(group, reference, source, research_context)
              for group in groups],
            return_exceptions=True)
        results: dict[int, CollectResult] = {}
        records: list[object] = []
        for group, outcome in zip(groups, gathered):
            if isinstance(outcome, PermanentProviderError):
                raise outcome
            if isinstance(outcome, BaseException):
                if isinstance(outcome, (KeyboardInterrupt, SystemExit,
                                        asyncio.CancelledError)):
                    raise outcome
                outcome = (group.positions,
                           [self._failed_group_claim(claim, source, outcome)
                            for claim in group.claims],
                           None)
            positions, produced, record = outcome
            for position, result in zip(positions, produced):
                results[position] = result
            if record is not None:
                records.append(record)
        return CollectStudyResult(
            claim_results=[results[i] for i in sorted(results)],
            batch_records=records)

    async def collect(
        self,
        review_item,
        reference,
        *,
        research_context: str = "",
        source=None,
        route: str = "",
    ):
        """Abstract precheck, then the frozen collector. Permanent errors raise."""
        if getattr(review_item, "cohort_status", "resolved") not in (
                "unknown", "ambiguous"):
            if source is None:
                source = await self.open_study(reference)
            document = getattr(source, "document", None)
            if getattr(source, "retrieved", False) and document is not None:
                field_type = str(getattr(review_item, "field_type", "") or "")
                reason = abstract_omits_concept(document, _concept_names(
                    field_type=field_type,
                    raw_field_name=str(
                        getattr(review_item, "raw_field_name", "") or ""),
                    concept=self._concept_for(field_type),
                    variants=self._concept_variants_for(field_type)))
                if reason:
                    return self._result(
                        review_item,
                        SourceValueResult(found=False, not_found_reason=reason),
                        list(getattr(source, "steps", []) or []),
                        ReflectionDecision.ESCALATE,
                        CollectionOutcome.MISSING_SOURCE,
                        provenance=getattr(source, "provenance", None) or {},
                        document=document)
        return await super().collect(
            review_item, reference, research_context=research_context,
            source=source, route=route)

    async def _run_group(self, group, reference, source, research_context):
        """One group, as a single task. Intra-group stays serial (F3)."""
        route = (self._contract.route_for(group.kind) if self._contract
                 else self._extraction_profile)
        try:
            if is_batch_route(route):
                produced, record = await self._collect_batched(
                    group, reference, source, research_context, route=route)
            else:
                produced = [
                    await self.collect(claim, reference,
                                       research_context=research_context,
                                       source=source, route=route)
                    for claim in group.claims]
                record = None
            return group.positions, produced, record
        except PermanentProviderError:
            raise
        except Exception as exc:
            produced = [self._failed_group_claim(claim, source, exc)
                        for claim in group.claims]
            return group.positions, produced, None

    def _failed_group_claim(self, claim, source, exc) -> CollectResult:
        """Keep the existing failure record; do not swallow the exception."""
        retrieved = bool(source is not None and source.retrieved)
        outcome = CollectionOutcome.EXTRACTION_FAILED if retrieved else (
            getattr(source, "outcome", None) or CollectionOutcome.SOURCE_ACCESS_FAILED)
        return self._result(
            claim,
            SourceValueResult(
                found=False, error=f"{type(exc).__name__}: {exc}"[:300],
                not_found_reason=(
                    getattr(exc, "not_found_reason", "")
                    or f"the extraction call failed: {exc}"[:300]),
                permanent_failure=isinstance(exc, PermanentProviderError)),
            list(getattr(source, "steps", []) or []),
            ReflectionDecision.ESCALATE, outcome,
            provenance=getattr(source, "provenance", None) or {},
            document=getattr(source, "document", None))

    async def _collect_batched(self, group, reference, source, research_context,
                               *, route: str = BATCH_PROFILE_NAME):
        """One reading for a whole group. ``batch_split_v1`` is still two calls,
        not N: locate the group, then transcribe every quote."""
        if self._batch is None:
            raise ContractError(
                f"{group.describe()} is routed to {route} and this "
                "Collector has no batch tool. A run that cannot honour its own "
                "contract must stop rather than read the claims some other way")
        if not source.retrieved or source.document is None:
            return [self._unretrieved(claim, source) for claim in group.claims], None

        field_type = group.key.field_type
        concept = self._concept_for(field_type)
        variants = self._concept_variants_for(field_type)
        skip_reason = abstract_omits_concept(source.document, _concept_names(
            field_type=field_type,
            raw_field_name=group.key.raw_field_name,
            concept=concept,
            variants=variants))
        if skip_reason:
            skipped = [
                self._result(
                    claim,
                    SourceValueResult(found=False, not_found_reason=skip_reason),
                    list(getattr(source, "steps", []) or []),
                    ReflectionDecision.ESCALATE,
                    CollectionOutcome.MISSING_SOURCE,
                    provenance=getattr(source, "provenance", None) or {},
                    document=source.document)
                for claim in group.claims]
            return skipped, None
        target = concept or group.key.raw_field_name or field_type
        text = getattr(source.document, "full_text", "") or ""
        table_text = _unique_table_excerpt(
            source.document, group=group, field_type=field_type,
            concept=concept, variants=variants,
            cohorts=self._cohort_variants())
        if table_text:
            excerpt, spans = table_text, [(0, len(table_text))]
            method_id, method_version = TABLE_SELECTION_METHOD_ID, TABLE_SELECTION_VERSION
        else:
            excerpt, spans = select_excerpt(text, target=target,
                                            raw_label=group.key.raw_field_name,
                                            field_type=field_type, variants=variants)
            method_id, method_version = SELECTION_METHOD_ID, SELECTION_VERSION
        split = route == BATCH_SPLIT_PROFILE
        if split:
            prompt = build_batch_locate_prompt(
                target_shape=group.shape, context=research_context,
                field_type=field_type, concept=target,
                raw_label=group.key.raw_field_name or target,
                concept_variants=", ".join(variants) or target,
                unit_hint=group.key.unit_signature, paper_text=excerpt,
                timepoint_label=group.key.timepoint_label)
        else:
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
            prompt_version=prompt_version(route),
            prompt_sha256=prompt_sha256(prompt),
            aggregable=aggregation_applies(group.shape, field_type))
        if split:
            record = await self._batch.read_split(
                question=question, locate_prompt=prompt, document=excerpt)
        else:
            record = await self._batch.read(question=question, prompt=prompt,
                                            document=excerpt)
        bindings = [self._binding_for(claim, group, route=route)
                    for claim in group.claims]
        record.execution = execution_id_for(question, bindings,
                                            self._projection_contract())
        record.excerpt = ExcerptProvenance(
            windowed=len(excerpt) != len(text), source_chars=len(text),
            excerpt_chars=len(excerpt), spans=spans,
            selection_method_id=method_id,
            selection_version=method_version)
        if self._telemetry is not None:
            self._telemetry.record_batch(claims=len(group.claims),
                                         failed=record.reading is None)
        produced = [self._project_one(claim, group, record, source, binding)
                    for claim, binding in zip(group.claims, bindings)]
        return produced, record

    def _binding_for(self, claim, group, *, route: str = BATCH_PROFILE_NAME) -> ClaimBinding:
        scope = getattr(claim, "population_scope", None)
        return ClaimBinding(
            claim_id=_claim_id(claim), target=str(claim.group or ""),
            requested_scope=(scope.describe() if scope is not None else ""),
            route=route,
            required_axes=tuple(self._axes_for(claim.field_type)),
            timepoint_label=str(getattr(claim, "timepoint_label", "") or ""))

    def _project_one(self, claim, group, record, source, binding) -> CollectResult:
        out = super()._project_one(claim, group, record, source, binding)
        item = out.source_item
        if item.batch_provenance is not None and binding.route:
            item = item.model_copy(update={
                "batch_provenance": item.batch_provenance.model_copy(
                    update={"route": binding.route}),
            })
            out = out.model_copy(update={
                "source_item": item,
                "record": out.record.model_copy(
                    update={"final": item.model_dump(mode="json")}),
            })
        name = self._paper_label_for(record, out.source_item)
        review_name = claim.raw_field_name or claim.field_type
        return self._attach_mapping(out, review_name, name)

    def _result(self, *args, **kwargs):
        out = super()._result(*args, **kwargs)
        result = args[1] if len(args) > 1 else kwargs.get("result")
        review_item = args[0] if args else kwargs.get("review_item")
        label = str(getattr(result, "source_field_name", "") or "").strip()
        if not label or review_item is None:
            return out
        review_name = (getattr(review_item, "raw_field_name", "")
                       or getattr(review_item, "field_type", ""))
        return self._attach_mapping(out, review_name, label)

    def _paper_label_for(self, record, source_item) -> str:
        names = getattr(record, "field_names", None) or {}
        if not names or record.reading is None:
            return ""
        selected = str(getattr(source_item.batch_provenance, "selected_entry_id", "")
                       or "")
        if not selected:
            return ""
        question_id = record.question.identity()
        for entry in record.reading.usable:
            if entry.entry_id(question_id) == selected:
                return str(names.get(entry.raw_index, "") or "")
        return ""

    def _attach_mapping(self, out: CollectResult, review_name: str,
                        paper_label: str) -> CollectResult:
        label = str(paper_label or "").strip()
        if not label:
            return out
        if any(r.code == "source_field_mapping" for r in out.source_item.reasons):
            return out
        item = out.source_item.model_copy(
            update={"reasons": list(out.source_item.reasons) + [
                mapping_reason(review_name, label)]})
        record = out.record.model_copy(
            update={"final": item.model_dump(mode="json")})
        return out.model_copy(update={"source_item": item, "record": record})
