"""Production collector: batch-split routing and human-readable field mapping.

``collector.py`` is inside the frozen ``evidence_adequacy_1.0.0`` hash boundary,
so this file exists: v8 contracts must still load against that exact collector
module. Split routing and the ``source_field_mapping`` reason live here.

``source_field_mapping`` is human-readable only. It lets a reader check
综述叫法 → 论文叫法. Downstream must not parse the message (or scrape ``code``)
to recover a structured ``source_field_name`` — statistics, field alignment,
anything that branches on the paper's label. That needs evaluator 1.8.3 and a
real field on ``SourceEvidenceItem``. ``detail`` is left empty on purpose so
this cannot be used as a back-door structured slot.
"""
from __future__ import annotations

from react_review.agents.collector import (
    Collector,
    CollectResult,
    CollectStudyResult,
    _claim_id,
    _document_sha256,
    _refuse_repeated_identities,
)
from react_review.contracts import ContractError
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
        if source is None:
            source = await self.open_study(reference)
        _refuse_repeated_identities(claims)
        results: dict[int, CollectResult] = {}
        records: list[object] = []
        for group in group_claims(claims):
            route = (self._contract.route_for(group.kind) if self._contract
                     else self._extraction_profile)
            positions = group.positions
            if is_batch_route(route):
                produced, record = await self._collect_batched(
                    group, reference, source, research_context, route=route)
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
        target = concept or group.key.raw_field_name or field_type
        text = getattr(source.document, "full_text", "") or ""
        excerpt, spans = select_excerpt(text, target=target,
                                        raw_label=group.key.raw_field_name,
                                        field_type=field_type, variants=variants)
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
            selection_method_id=SELECTION_METHOD_ID,
            selection_version=SELECTION_VERSION)
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
