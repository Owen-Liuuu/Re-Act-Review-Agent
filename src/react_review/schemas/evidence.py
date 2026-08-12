"""Review-side and source-side evidence rows (the two extraction ground-truths).

Field names mirror the benchmark CSV columns so the eval harness can load them
directly. ``group`` uses the canonical vocabulary values (``t1dm`` / ``control``
/ ``all`` / ``-`` for study-level rows); ``field_type`` is the canonical concept
key. ``value`` / ``source_value`` are kept as the verbatim strings (e.g.
``"6.60 ± 0.71"``) — the syntax normaliser extracts the primary number at
compare time, so the raw spread is preserved for the report.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_serializer

from react_review.core.enums import CollectionOutcome
from react_review.normalize.population import PopulationScope
from react_review.schemas.reason import ReasonRecord

Value = str | int | float | None


class CohortCount(BaseModel):
    """One explicitly printed arm count and its supporting source words."""

    label: str
    count: int
    quote: str


class AggregationProvenance(BaseModel):
    """Everything a derived total rests on, and which rules permitted it.

    A computed number is only as good as the account of how it was computed, and
    that account has to survive as far as the reader. Four separate passages are
    load-bearing and none substitutes for another: the components say what was
    added, the partition says why adding them is the whole, the population says
    whom they count, and the timepoint says when. The policy id and hash say
    which version of the conditions was applied, so a total derived today can be
    re-judged against the rules that were in force rather than today's.
    """

    policy_id: str = ""
    policy_sha256: str = ""
    # WHICH CODE applied those rules. A policy hash alone says what the
    # conditions were, not what enforced them, and every wrong total in this
    # phase came from rules that read correctly and code that did not keep them.
    evaluator_id: str = ""
    evaluator_version: str = ""
    evaluator_hash: str = ""
    git_commit: str = ""                    # full 40 characters, or empty
    git_commit_matches_evaluator: bool = False
    evaluator_status: str = ""              # registered | unregistered | unavailable
    release_eligible: bool = False
    aggregation_set: str = ""               # which population/timepoint won
    required_axes: list[str] = Field(default_factory=list)
    population_quote: str = ""
    timepoint_quote: str = ""
    partition_quote: str = ""
    component_quotes: list[str] = Field(default_factory=list)
    #: Sets that could not be read. Present even when a printed total was
    #: released instead, because a broken part of the response is a fact about
    #: the response rather than about the claim that survived it.
    errors: list[str] = Field(default_factory=list)
    #: Broken sets that described other people, and so cost this claim nothing.
    unrelated_rejections: list[str] = Field(default_factory=list)

    @property
    def anchors(self) -> list[str]:
        return [q for q in (self.population_quote, self.timepoint_quote,
                            self.partition_quote, *self.component_quotes) if q]


class BatchProjectionProvenance(BaseModel):
    """Which reading this claim came out of, and how it was picked.

    A batched run answers several claims from one act of reading. Without this,
    an artifact can show four answers and no way to establish that they came
    from ONE response rather than four — which is the entire cost and the entire
    consistency argument for batching, unevidenced.

    The response itself is not here. It is kept once, keyed by
    ``batch_execution_id``; copying it onto every claim would multiply it by the
    group size and still not show that the group shared it.
    """

    batch_question_id: str = ""
    batch_execution_id: str = ""
    claim_group_key: str = ""
    claim_id: str = ""
    #: Which piece of the response was released, when one was. Empty for a
    #: derived total, which is not any single entry.
    selected_entry_id: str = ""
    projection_status: str = ""
    projection_reason: str = ""
    #: The profile that ACTUALLY read this claim. A mixed contract routes value
    #: claims and arm-identity claims differently, and a reader must be able to
    #: ask which one produced this row rather than infer it from a run-level
    #: profile that only describes half the run.
    route: str = ""
    attempts: int = 0
    served_from_cache: bool = False


class SourceNumericComponents(BaseModel):
    """A source value's parts, as the extraction reported them.

    The comparator used to see only the verbatim string, so a response that gave
    the point estimate and dropped the interval was indistinguishable from a
    source that prints no interval. Each part is recorded separately, together
    with whether it was found in the supporting quote, so an incomplete
    extraction stays visible instead of earning a complete-match verdict.
    """

    point_estimate: float | None = None
    ci_level: float | None = None          # 95, 99.5 — the level, not the bounds
    ci_lower: float | None = None
    ci_upper: float | None = None
    # component name -> whether it is anchored in ``source_quote``
    anchored: dict[str, bool] = Field(default_factory=dict)
    # Components the paper prints that the extraction did not return.
    missing: list[str] = Field(default_factory=list)
    status: str = "ok"          # ok | incomplete | protocol_error
    reason: str = ""

    @property
    def complete_interval(self) -> bool:
        return None not in (self.ci_level, self.ci_lower, self.ci_upper)


class ReviewDataItem(BaseModel):
    """One value the review reports (a cell of its data-extraction table).

    Mirrors ``review_ground_truth.csv``.
    """

    review_data_id: str = ""
    study_id: str
    group: str = "-"
    timepoint: str = "single"
    field_type: str
    raw_field_name: str = ""
    value: Value = None
    unit: str = ""
    source_location: str = ""
    # DKB resolution: "resolved" (authoritative) | "candidate" (provisional, tentative)
    # | "unresolved" (field_type unknown — kept, but not comparable / needs review).
    resolution_status: str = "resolved"
    # --- provenance back to the captured table (all optional: CSV-loaded items
    # and hand-built test items keep working unchanged) ---
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None    # (row, column) in the captured table
    column_header: str = ""                    # the header path, verbatim
    cohort_label: str = ""                     # the review's OWN word for the cohort
    # What population the REVIEW says this cell counts, and where that came
    # from (a declared evaluation contract, or the review's own column words).
    population_scope: PopulationScope | None = None
    population_scope_source: str = ""          # contract | column_header | ""
    # How that label was placed: resolved | alias | combined | ambiguous |
    # unknown | not_applicable (a study-level field has no cohort dimension).
    # "unknown"/"ambiguous" must reach a human — never be treated as a cohort.
    cohort_status: str = "resolved"
    timepoint_label: str = ""                  # the review's OWN word, "" if none
    origin: str = "review_table"               # review_table | checklist
    # Stable identity for a checklist-authored concrete claim. Empty for normal
    # table cells and for presence/gap checks, which never enter value matching.
    checklist_id: str = ""
    # Joins this row to the run-level FieldResolutionRecord that explains how
    # ``raw_field_name`` became ``field_type``.  Empty for placeholders and
    # hand-built/CSV fixtures that never went through the Resolver.
    resolution_key: str = ""
    reasons: list[ReasonRecord] = Field(default_factory=list)


class SourceEvidenceItem(BaseModel):
    """One value located in a source paper, keyed to the review claim it answers.

    Mirrors the source side of ``audit_template.csv``.
    """

    study_id: str
    group: str = "-"
    timepoint: str = "single"
    field_type: str
    # Which review cell this evidence answers. Carried so two claims that share
    # a study/cohort/field can still be told apart when they are paired.
    table_id: str = ""
    cell_ref: tuple[int, int] | None = None
    checklist_id: str = ""
    source_value: Value = None
    source_unit: str = ""
    source_quote: str = ""
    source_location_in_paper: str = ""
    value_origin: str = ""       # verbatim | derived_sum | unresolved
    derivation: str = ""
    # Structured parts of the source value (Phase 7B). ``None`` means the
    # extraction contract that produces them did not run for this item.
    source_components: SourceNumericComponents | None = None
    # WHICH POPULATION this value counts, read from its own supporting quote.
    # 313 analysed and 314 allocated are different quantities, not a 0.3%
    # transcription difference, and nothing upstream of this could say so.
    population_scope: PopulationScope | None = None
    cohort_counts: list[CohortCount] = Field(default_factory=list)
    aggregation_status: str = "not_applicable"  # not_applicable | derived | rejected | protocol_error
    aggregation_reason: str = ""
    evidence_check: str = "ok"   # ok | protocol_error
    evidence_reason: str = ""
    # WHERE this was read from. ``source_location_in_paper`` says "Table 2" — of
    # WHICH document was never recorded, so a reader could not go and check.
    # A local run has a file; an online one has a URL; both always have a kind.
    source_file: str = ""          # absolute path, when read from disk
    source_uri: str = ""           # URL / PMC id / DOI — the online equivalent
    source_paper_id: str = ""
    source_doi: str = ""
    retriever_kind: str = ""       # local_pdf | pmc | unpaywall | openalex_pdf | …
    collection_outcome: CollectionOutcome = CollectionOutcome.FOUND
    # Back-check: source evidence (unit/value) contradicts a CANDIDATE translation
    # → the auto-classified field_type is likely wrong. Set only for candidates.
    concept_mismatch: bool = False
    concept_mismatch_reason: str = ""
    # ok | wrong_cohort | ambiguous — whether the paper's own cohort label could
    # be confirmed against the one asked for. "ambiguous" must reach a human.
    cohort_check: str = "ok"
    # How the value was tied to the requested arm or comparison pair.
    # ok | reassigned | ambiguous | not_reported | direction_inverted |
    # inconsistent | unsupported | protocol_error.
    target_check: str = "ok"
    target_reason: str = ""
    assigned_arm_label: str = ""    # the paper's own name for the assigned arm
    cohorts_seen: list[str] = Field(default_factory=list)
    reasons: list[ReasonRecord] = Field(default_factory=list)
    # --- batch-path provenance -------------------------------------------
    # Both are None on every path that predates the batch, and both are DROPPED
    # from the serialised form when None (see `model_dump` below). A legacy
    # artifact gaining `"batch_provenance": null` is a changed artifact, and the
    # replay comparisons that guard this project compare bytes.
    batch_provenance: BatchProjectionProvenance | None = None
    aggregation_provenance: AggregationProvenance | None = None

    #: Fields that exist only for the batch path. Present when a batch produced
    #: this row; absent — not null — otherwise.
    _BATCH_ONLY = ("batch_provenance", "aggregation_provenance")

    @model_serializer(mode="wrap")
    def _omit_unused_batch_fields(self, handler):
        """Legacy rows serialise exactly as they did before these fields existed.

        A wrap serialiser rather than a `model_dump` override, because Pydantic
        serialises NESTED models through its own core machinery and never calls
        an overridden method on the child. The override looked right on this row
        and did nothing inside an EvidencePackage — which is the one that
        matters, since that is what a replay compares.

        Not `exclude_none`: `source_components` and `population_scope` are null
        on purpose in every artifact ever written, and dropping them would
        change exactly the bytes this exists to protect.
        """
        body = handler(self)
        for name in self._BATCH_ONLY:
            if body.get(name) is None:
                body.pop(name, None)
        return body


class IncludedStudy(BaseModel):
    """A cited source paper. Mirrors ``included_studies.csv``."""

    study_id: str
    review_citation: str = ""
    country: str = ""
    reported_N: int | None = None
    measurement_tool: str = ""
    modality: str = ""
    overall_quality: str = ""
    doi: str = ""
    review_ref_number: str = ""
    source_pdf: str = ""
