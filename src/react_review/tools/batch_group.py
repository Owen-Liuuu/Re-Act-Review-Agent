"""Which claims may be asked together, and what asking them is called.

Grouping is the whole economy of the batch, and it is also where a batch can
quietly become wrong. Two claims belong in one reading when they put the SAME
question to the same paper — same field, same shape, same words, same timepoint.
They do not belong together merely because they concern the same study: a claim
about an arm and a claim about a comparison need different prompts, and a claim
whose review column is worded differently is a different question however alike
its ``field_type`` makes it look.

Three identities come out of this module and they are not interchangeable:

*The group key* decides which claims MAY be asked together. It is a planning
device: change it and the batches are drawn differently.

*The question id* says what the model was asked. It is what a cache entry is
about, and it deliberately excludes the claims: v5 asks for every arm and every
population the paper reports, so two consumers and three ask the same thing.

*The execution id* says what was answered. It carries the claims, their scopes
and the rules the response was read under, because the same recording projected
under different axes is a different set of answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from react_review.normalize.cohorts import parse_comparison
from react_review.schemas.batch import (
    ARM,
    COMPARISON,
    STUDY,
    BatchExecutionId,
    BatchQuestionId,
    ClaimBinding,
    ClaimGroupKey,
    ProjectionContract,
)

#: A review cell whose cohort is one of these is about the study, not an arm.
#: The same two strings the single-target path has always used for it.
STUDY_LEVEL_COHORTS = {"all", "-"}


def claim_kind(review_item) -> str:
    """Whether this claim asks what an arm IS, or what it reports.

    Structural, not a vocabulary: a review's cell for an arm-identity column IS
    the review's own name for that arm — the same text the cohort was labelled
    with. Nothing here knows what a drug is.
    """
    value = _norm(getattr(review_item, "value", ""))
    label = _norm(getattr(review_item, "cohort_label", ""))
    return "arm_identity" if value and value == label else "value"


def target_shape(review_item) -> str:
    """Study total, one arm, or a comparison between two.

    Read from the review's own cohort field, by the same rules the single-target
    path uses, so that batching a claim cannot change what the claim is about.
    """
    group = str(getattr(review_item, "group", "") or "")
    if parse_comparison(group) is not None:
        return COMPARISON
    return STUDY if group.strip().lower() in STUDY_LEVEL_COHORTS else ARM


def group_key_for(review_item) -> ClaimGroupKey:
    """The question this claim belongs to — everything except which cohort.

    The cohort is the dimension a batch spans, and the population is too: a
    batch is asked to bring back every population it can see and the claim picks
    afterwards. Everything else that changes the QUESTION is in the key.
    """
    return ClaimGroupKey(
        study_id=str(getattr(review_item, "study_id", "") or ""),
        field_type=str(getattr(review_item, "field_type", "") or ""),
        raw_field_name=str(getattr(review_item, "raw_field_name", "") or ""),
        column_header=str(getattr(review_item, "column_header", "") or ""),
        timepoint=str(getattr(review_item, "timepoint", "") or ""),
        timepoint_label=str(getattr(review_item, "timepoint_label", "") or ""),
        target_shape=target_shape(review_item),
        target_kind=claim_kind(review_item),
        unit_signature=str(getattr(review_item, "unit", "") or ""),
    )


@dataclass
class ClaimGroup:
    """One prompt's worth of claims, and where each came from.

    The positions travel WITH the claims. Recovering them afterwards by looking
    each claim up in the original list fails silently the moment two claims are
    equal — `list.index` returns the first match for both, so one result
    overwrites the other and a later read runs off the end.
    """

    key: ClaimGroupKey
    claims: list = field(default_factory=list)
    positions: list[int] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.key.target_kind

    @property
    def shape(self) -> str:
        return self.key.target_shape

    def describe(self) -> str:
        return f"{self.key.describe()} × {len(self.claims)}"


def group_claims(review_items) -> list[ClaimGroup]:
    """Group a study's claims, preserving arrival order within and between.

    Order is kept so that a preflight and a run enumerate the same batches in
    the same sequence, and so that anything reconstructing a run from its
    records reads it the way it happened.
    """
    groups: dict[str, ClaimGroup] = {}
    for position, item in enumerate(review_items):
        key = group_key_for(item)
        group = groups.setdefault(key.key(), ClaimGroup(key=key))
        group.claims.append(item)
        group.positions.append(position)
    return list(groups.values())


def question_id_for(group: ClaimGroup, *, concept: str = "",
                    concept_variants=(), research_context: str = "",
                    document_sha256: str = "", knowledge_fingerprint: str = "",
                    prompt_version: str = "", prompt_sha256: str = "",
                    aggregable: bool = False) -> BatchQuestionId:
    """What the model is asked. Every field here reaches the prompt."""
    return BatchQuestionId(
        study_id=group.key.study_id, target_shape=group.shape,
        field_type=group.key.field_type, raw_field_name=group.key.raw_field_name,
        concept=concept, concept_variants=tuple(concept_variants),
        unit_hint=group.key.unit_signature,
        timepoint_label=group.key.timepoint_label, aggregable=aggregable,
        research_context=research_context, document_sha256=document_sha256,
        knowledge_fingerprint=knowledge_fingerprint,
        prompt_version=prompt_version, prompt_sha256=prompt_sha256)


def execution_id_for(question: BatchQuestionId, bindings: list[ClaimBinding],
                     projection: ProjectionContract) -> BatchExecutionId:
    return BatchExecutionId(question=question, bindings=list(bindings),
                            projection=projection)


@dataclass
class BatchPlan:
    """What a run WOULD ask, worked out without asking anything.

    Batching was justified by a cost argument, and a cost argument that cannot
    be checked before it is paid is a hope. This makes the shape of the run
    inspectable: how many prompts, how many claims each answers, and how many
    batches contain exactly one claim — the last being the number that decides
    whether batching is buying anything at all.
    """

    groups: list[ClaimGroup] = field(default_factory=list)

    @property
    def batch_count(self) -> int:
        return len(self.groups)

    @property
    def claim_count(self) -> int:
        return sum(len(g.claims) for g in self.groups)

    @property
    def singletons(self) -> int:
        return sum(1 for g in self.groups if len(g.claims) == 1)

    @property
    def calls_saved(self) -> float:
        """Claims per prompt. One means the batch bought nothing."""
        return (self.claim_count / self.batch_count) if self.batch_count else 0.0

    def by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for group in self.groups:
            counts[group.kind] = counts.get(group.kind, 0) + len(group.claims)
        return counts

    def summary(self) -> str:
        return (f"{self.batch_count} batch(es) for {self.claim_count} claim(s) — "
                f"{self.calls_saved:.2f} claims per prompt, "
                f"{self.singletons} singleton(s); by kind {self.by_kind()}")


def plan_batches(review_items) -> BatchPlan:
    """Enumerate the batches a run would issue. Sends nothing."""
    return BatchPlan(groups=group_claims(review_items))


def _norm(value: object) -> str:
    return " ".join(str(value or "").lower().split())
