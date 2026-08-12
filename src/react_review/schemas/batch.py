"""What one batched reading of a paper contains, and how a claim finds itself in it.

Phase 7 asked one question per cell and threw away the rest of the answer: the
model enumerated three arms and exactly one was consumed. Batching keeps the
whole reading — but only if the pieces can be told apart, and the ways they must
be told apart are exactly the ways the audit has already been wrong.

*A target is not a row of the response.* The same arm legitimately appears more
than once: a trial reports 314 allocated to the combination arm and 313
analysed in it. Both are that arm. Identity therefore has two levels — WHICH
ARM (used to map the paper's arms onto the review's) and WHICH READING of it
(population, timepoint, effect definition) — and collapsing them is what would
make the MA004 evidence look like two different arms, or like a duplicate to be
thrown away.

*An entry's id is not its selection key.* ``EntryId`` says which piece of the
response this is; ``SelectionKey`` says what it is a reading of. Two entries
that share a SelectionKey and disagree about the value are a contradiction to be
surfaced, not a duplicate to be deleted — and certainly not grounds for
discarding the whole batch.

*A group key is not a request id.* The group key decides which claims may be
asked together; the request id records what was actually asked, including which
claims were in it. A key that stayed the same while its membership changed would
make two different questions look like one.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from react_review.normalize.cohorts import distinguishing_tokens
from react_review.normalize.population import PopulationScope

#: What kind of thing a batch is about. The prompt asks only for this shape:
#: making every batch enumerate arms AND comparisons AND populations grows
#: quadratically in the comparisons and spends more on output than batching
#: saves on calls.
STUDY = "study"
ARM = "arm"
COMPARISON = "comparison"
TARGET_SHAPES = (STUDY, ARM, COMPARISON)

UNSPECIFIED = ""


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")).hexdigest().upper()


class EvidenceAnchor(BaseModel):
    """Where in the document a piece of evidence sits.

    Offsets are into the flattened source text. They exist so that a population
    or timepoint phrase can be shown to belong to THIS value rather than merely
    to occur somewhere in the same paper — the difference between evidence and
    coincidence.
    """

    quote: str = ""
    start: int | None = None        # offset of the quote in the flattened text
    block: str = ""                 # table id / section label, when known

    @property
    def located(self) -> bool:
        return self.start is not None


class EntryIdentity(BaseModel):
    """The full composite identity of one reading.

    Two entries are the same reading only if ALL of this matches. The arm alone
    is not an identity: "the combination arm" names a target, and "the
    combination arm, allocated, at randomisation" names a reading of it.
    """

    target_shape: str = ARM                 # study | arm | comparison
    target_kind: str = "value"              # value | arm_identity
    arm_label: str = ""                     # the paper's own words, for an arm
    comparison_pair: tuple[str, str] | None = None
    timepoint: str = UNSPECIFIED            # the review's canonical timepoint
    timepoint_phrase: str = ""              # the paper's own words for it
    population: PopulationScope | None = None
    effect_definition: str = ""             # the paper's own words, when stated

    def target(self) -> tuple:
        """WHICH ARM — the level at which the paper is mapped to the review.

        Deliberately excludes population, timepoint and effect: those are
        readings OF this target, and folding them in here is what would make one
        arm look like several and break the one-to-one assignment.

        Labels are compared on their distinguishing WORDS, because a paper names
        the same arm differently in different places — "nivolumab-plus-
        ipilimumab group" in the results, "Nivolumab plus Ipilimumab" in a table
        header. Comparing the strings would split one arm in two and lose
        exactly the case this identity exists to hold together. It stays strict
        where it must: "nivolumab group" keeps one word where the combination
        keeps three, so the two never collapse.
        """
        labels = self.comparison_pair or (self.arm_label,)
        return (self.target_shape, self.target_kind,
                tuple(tuple(sorted(distinguishing_tokens(label)))
                      for label in labels))

    def selection(self) -> tuple:
        """WHICH READING — what a claim selects among candidates for one target."""
        population = self.population or PopulationScope()
        return (self.target(), population.basis, population.analysis_set,
                self.timepoint, self.effect_definition)


class BatchEntry(BaseModel):
    """One reading the paper reports, with the evidence that makes it one."""

    identity: EntryIdentity
    value: str | None = None
    unit: str = ""
    quote: str = ""
    components: dict[str, float] = Field(default_factory=dict)
    population_anchor: EvidenceAnchor | None = None
    timepoint_anchor: EvidenceAnchor | None = None
    effect_anchor: EvidenceAnchor | None = None
    raw_index: int = -1                     # position in the model's response
    rejected_reason: str = ""               # set when this entry alone was refused

    @property
    def usable(self) -> bool:
        return not self.rejected_reason and self.value is not None

    def entry_id(self, response_digest: str) -> str:
        """Which PIECE of which response this is — never what it is about."""
        return _digest({"response": response_digest, "index": self.raw_index,
                        "quote": self.quote})

    def selection_key(self) -> str:
        return _digest(list(self.identity.selection()))


class BatchCohortCount(BaseModel):
    """One arm's count, offered as a COMPONENT of a total rather than as a reading.

    Kept apart from :class:`BatchEntry` deliberately. A component is not an
    answer to anything on its own — it exists only to be summed, and only if the
    partition evidence holds. Letting it travel as an ordinary reading would put
    a per-arm number one selection away from being returned as a whole-study
    total, which is the confusion this separation exists to make impossible.

    It carries no population of its own: it belongs to an
    :class:`AggregationSet`, and the set is what has a population. A component
    that could name its own population could be put in a set that contradicts
    it, which is the mixing this structure exists to make unrepresentable.
    """

    arm_label: str = ""
    count: int = 0
    quote: str = ""
    #: Position in the model's response, so a rejected component can be named.
    source_index: int = -1


class PartitionWitness(BaseModel):
    """The claim that a set's arms cover its population once each, with proof.

    Both flags default to ``False``: an assessment the model did not make is not
    a partition it established. But the flags are the weakest part of this — they
    are the model's opinion about the paper. What makes them checkable is the
    rest: a locatable passage, and how many arms (or which ones) that passage
    says there are. Given those, deterministic code can confirm that the
    components in hand really are the set the paper described, instead of
    trusting ``complete: true`` about a list it cannot see.
    """

    complete: bool = False
    mutually_exclusive: bool = False
    quote: str = ""
    reason: str = ""
    #: How many arms the paper says the population was divided into ("three
    #: groups"). The check that turns "complete" into something verifiable.
    declared_arm_count: int | None = None
    #: The paper's own names for them, when it lists them.
    declared_arm_labels: list[str] = Field(default_factory=list)
    #: Set by the parser when ``quote`` was found in the document.
    anchored: bool = False


class AggregationSet(BaseModel):
    """One population, at one timepoint, and the arms that divide it.

    A paper reports several of these — who was randomised, who was treated, who
    was analysed — and they are different sets of people whose counts must never
    meet. Making each its own set means mixing them is not a mistake the code has
    to catch: there is no place to put a mixture.
    """

    #: The model's own label for this population. Kept verbatim for the record,
    #: but never used for matching: the classified scope below is what decides,
    #: because a second population vocabulary would eventually disagree with the
    #: frozen one and nothing would say which was right.
    population_type: str = ""
    population_phrase: str = ""
    population_quote: str = ""
    timepoint_phrase: str = ""
    timepoint_quote: str = ""
    cohort_counts: list[BatchCohortCount] = Field(default_factory=list)
    partition: PartitionWitness | None = None
    #: Classified from the phrase by the frozen population contract, once the
    #: phrase has been shown to be anchored.
    population: PopulationScope | None = None
    source_index: int = -1

    def scope_key(self) -> tuple[str, str, str]:
        """What a claim matches against: BOTH population axes, and the timepoint."""
        scope = self.population or PopulationScope()
        return (scope.basis, scope.analysis_set, self.timepoint_phrase)

    def describe(self) -> str:
        scope = self.population or PopulationScope()
        return (f"{scope.describe()}"
                + (f" at {self.timepoint_phrase}" if self.timepoint_phrase else ""))


class RejectedAggregationSet(BaseModel):
    """A set that could not be read, and as much of it as could be.

    Dropping a broken set silently would be wrong twice over. If it described a
    population the claim does not want, refusing everything punishes a claim the
    response answered perfectly well. If it described the population the claim
    DOES want, letting another set answer instead would be worse: nobody can
    show the good set was the only candidate. Which of those applies depends on
    what the broken set was about, so whatever was legible about it is kept.
    """

    source_index: int = -1
    population: PopulationScope | None = None
    population_phrase: str = ""
    timepoint_phrase: str = ""
    errors: list[str] = Field(default_factory=list)

    @property
    def population_known(self) -> bool:
        return self.population is not None and self.population.stated

    def describe(self) -> str:
        where = (self.population.describe() if self.population_known
                 else "an unidentified population")
        return f"set {self.source_index} ({where})"


class BatchAggregationEvidence(BaseModel):
    """Everything a batch offers toward computing a total it could not read."""

    sets: list[AggregationSet] = Field(default_factory=list)

    @property
    def present(self) -> bool:
        return bool(self.sets)


class ClaimGroupKey(BaseModel):
    """Which claims may be asked in one reading.

    Everything that changes the QUESTION is in here. The cohort is not: that is
    the dimension a batch spans. Nor is the population: a batch is asked to
    bring back every population it can see, and the claim picks afterwards.
    """

    study_id: str
    field_type: str
    #: The review's OWN column label. It reaches the prompt, so two claims whose
    #: columns are worded differently are different questions however alike
    #: their field_type makes them look.
    raw_field_name: str = ""
    #: Display only. Deliberately NOT in the question identity: it never reaches
    #: the prompt, and letting it split a group would ask one paper the same
    #: thing twice because a header was punctuated differently.
    column_header: str = ""
    timepoint: str = UNSPECIFIED
    timepoint_label: str = ""               # the review's own words
    target_shape: str = ARM
    target_kind: str = "value"
    unit_signature: str = ""                # unit / effect family, when declared

    #: Carried for a reader, never for the key. See ``column_header``.
    _DISPLAY_ONLY = ("column_header",)

    def key(self) -> str:
        body = self.model_dump(mode="json")
        for name in self._DISPLAY_ONLY:
            body.pop(name, None)
        return _digest(body)

    def describe(self) -> str:
        return (f"{self.study_id}/{self.field_type}"
                f"@{self.timepoint or 'unspecified'}"
                f"[{self.target_shape}/{self.target_kind}]")


class BatchReadingRecord(BaseModel):
    """One reading of one paper, as it is written down.

    The persistent form of a batch. Kept ONCE per reading and referenced by
    every claim it answered, because copying it onto each claim would multiply
    it by the group size and still not establish that the group shared it.

    Deliberately a schema of its own rather than the working object: what a run
    holds in memory may change freely, and what an artifact promises may not.
    """

    question_id: str = ""
    execution_id: str = ""
    study_id: str = ""
    field_type: str = ""
    target_shape: str = ARM
    #: Which claims this reading answered, so a reference can be resolved in
    #: both directions.
    claim_ids: list[str] = Field(default_factory=list)
    attempts: int = 0
    served_from_cache: bool = False
    failure: str = ""
    detail: str = ""
    #: Entries the parser refused, one reason each. A reading that produced
    #: usable answers AND refusals is a fact worth keeping: it says the response
    #: was partly unusable without costing the claims that survived it.
    parse_errors: list[str] = Field(default_factory=list)
    usable_readings: int = 0
    rejected_readings: int = 0
    #: The DECODED JSON the cache stores — not the model's literal bytes, which
    #: the cache has never held. Calling it a raw response would describe a
    #: provenance nobody has.
    model_payload: dict | None = None


class BatchQuestionId(BaseModel):
    """What the MODEL was asked — and nothing about who consumes the answer.

    Every field here reaches the prompt. Nothing else may: the v5 contract asks
    for EVERY arm and EVERY population the paper reports, so a group covering
    two claims and a group covering three put the identical question to the
    identical document. Folding the members in would record the same prompt
    twice and call a change of downstream consumer a change of question.

    The concept and its variants come from the knowledge base, so its
    fingerprint is in here too — otherwise a new ontology rewrites the prompt
    while the identity insists nothing moved.
    """

    study_id: str = ""
    target_shape: str = ARM
    field_type: str = ""
    raw_field_name: str = ""              # the review's own column label
    concept: str = ""                     # the KB's canonical name
    concept_variants: tuple[str, ...] = ()
    unit_hint: str = ""
    timepoint_label: str = ""             # the review's own words for when
    aggregable: bool = False              # whether components were asked for
    research_context: str = ""
    document_sha256: str = ""
    knowledge_fingerprint: str = ""
    prompt_version: str = ""
    prompt_sha256: str = ""

    def identity(self) -> str:
        return _digest(self.model_dump(mode="json"))


class ClaimBinding(BaseModel):
    """One claim's share of a batch, kept as a unit.

    Target, scope, route and axes travel together. Recorded as parallel lists
    they would say which targets and which scopes appeared, and not which went
    with which — and "the ITT total for arm A" and "the allocated total for arm
    B" are indistinguishable from their mirror image once the pairing is gone.
    """

    claim_id: str
    target: str = ""                      # the review's cohort key, or a pair
    requested_scope: str = ""             # "" | basis | basis/analysis_set
    route: str = ""                       # the extraction profile that read it
    required_axes: tuple[str, ...] = ()   # the axes THIS claim was held to
    timepoint_label: str = ""

    def key(self) -> tuple:
        return (self.claim_id, self.target, self.requested_scope, self.route,
                tuple(self.required_axes), self.timepoint_label)


class ProjectionContract(BaseModel):
    """The rules by which one response is turned into answers.

    A batched response is read twice: once by the model, once by the projection.
    The second reading depends on things the first never saw — which axes the
    run requires, how populations are classified, how the review's labels map to
    the paper's, which aggregation policy and which evaluator. Leave them out
    and one execution id can describe two different sets of answers.
    """

    run_profile_sha256: str = ""
    population_contract_sha256: str = ""
    cohort_fingerprint: str = ""          # the review-label mapping in force
    aggregation_policy_id: str = ""
    aggregation_policy_sha256: str = ""
    evaluator_id: str = ""
    evaluator_version: str = ""
    evaluator_hash: str = ""

    def identity(self) -> str:
        return _digest(self.model_dump(mode="json"))


class BatchExecutionId(BaseModel):
    """What was asked, who it was asked for, and how the answer was read.

    Distinct from :class:`BatchQuestionId` on purpose. The question identifies a
    cache entry — a recording may be reused whenever the same words were sent to
    the same model. This identifies an ANSWER: the same recording, read under
    different axes or a different evaluator, is a different set of answers and
    must not share provenance with them.
    """

    question: BatchQuestionId
    bindings: list[ClaimBinding] = Field(default_factory=list)
    projection: ProjectionContract = Field(default_factory=ProjectionContract)

    def identity(self) -> str:
        return _digest({
            "question": self.question.identity(),
            "bindings": sorted(list(b.key()) for b in self.bindings),
            "projection": self.projection.identity(),
        })

    def claim_ids(self) -> list[str]:
        return sorted(b.claim_id for b in self.bindings)
