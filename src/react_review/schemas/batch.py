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
    column_header: str = ""                 # the review's own words for the field
    timepoint: str = UNSPECIFIED
    timepoint_label: str = ""               # the review's own words
    target_shape: str = ARM
    target_kind: str = "value"
    unit_signature: str = ""                # unit / effect family, when declared

    def key(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def describe(self) -> str:
        return (f"{self.study_id}/{self.field_type}"
                f"@{self.timepoint or 'unspecified'}"
                f"[{self.target_shape}/{self.target_kind}]")


class BatchRequestId(BaseModel):
    """What was ACTUALLY asked — group, members, contract and document.

    A group key with A and B one day and A, B and C the next is the same key and
    a different question. This records the difference, so a recorded response is
    never replayed for a request it does not answer.
    """

    group: ClaimGroupKey
    claim_targets: list[str] = Field(default_factory=list)
    requested_scopes: list[str] = Field(default_factory=list)
    extraction_profile: str = ""
    research_context: str = ""
    document_sha256: str = ""

    def identity(self) -> str:
        return _digest({
            "group": self.group.key(),
            "targets": sorted(self.claim_targets),
            "scopes": sorted(self.requested_scopes),
            "profile": self.extraction_profile,
            "context": self.research_context,
            "document": self.document_sha256,
        })
