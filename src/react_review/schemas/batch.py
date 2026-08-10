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
        """
        return (self.target_shape, self.target_kind,
                self.comparison_pair or (self.arm_label,))

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
