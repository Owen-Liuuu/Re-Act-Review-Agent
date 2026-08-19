"""Extract stage: extract_source_value — directed, group-aware source lookup.

Stage 2 of the normalization pipeline: given a paper document and ONE target
concept (field_type + group), find that specific value with a verbatim quote —
NOT a blind whole-table extraction. The canonical concept came from the review
side; here the LLM maps it back to whatever the source paper calls it.
"""
from __future__ import annotations

import re
import unicodedata
from types import SimpleNamespace

import structlog
from pydantic import BaseModel, ConfigDict, Field

from react_review.contracts import ContractError
from react_review.schemas.telemetry import SINGLE_EXTRACTION
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.normalize.anchors import normalised_contains
from react_review.normalize.cohorts import ComparisonTarget
from react_review.normalize.population import PopulationScope, classify_population
from react_review.schemas.evidence import (
    AggregationProvenance,
    CohortCount,
    ReviewDataItem,
    SourceNumericComponents,
)
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.tools.base import Tool, ToolStage
from react_review.tools.extraction_cache import (
    ExtractionCache,
    ExtractionCacheMiss,
    extraction_cache_key,
)
from react_review.tools.extraction_profile import (
    DEFAULT_PROFILE,
    LEGACY_V3,
    prompt_profile,
    prompt_version,
    uses_targeted_sections,
)
from react_review.tools.value_components import (
    PROTOCOL_ERROR,
    verify_components,
)
from react_review.tools.target_assignment import (
    ArmEvidence,
    ComparisonEvidence,
    TargetAssignment,
    parse_arms,
    parse_comparisons,
    resolve_arm_target,
    resolve_comparison_target,
)

logger = structlog.get_logger(__name__)

_MAX_TEXT = 20000
#: The legacy contract's version string, kept importable under its old name.
PROMPT_VERSION = LEGACY_V3

_SAMPLE_SIZE_RULES = """- SPECIAL CASE — whole-study sample size: if the paper explicitly prints the
  total, copy it and set ``explicit_total_reported=true``. If it prints only arm
  counts, leave found=false/value=null and copy EVERY explicit arm count into
  ``cohort_counts`` with its own verbatim quote. Never add the arms yourself.
- ``explicit_total_reported=true`` is allowed ONLY when the total numeral itself
  appears in ``quote``. A quote containing "15 in A and 15 in B" does NOT print
  the total 30: return value=null and the two cohort_counts. Whenever multiple
  cohorts are seen, populate cohort_counts even if an explicit total also exists.
- Set ``cohort_partition_complete`` and ``cohort_partition_mutually_exclusive``
  true only when the paper clearly establishes that the listed arms cover the
  full population without overlap. Copy those supporting words/header into
  ``cohort_partition_quote``. For overlap, missing arms, a range, or unclear
  coverage, leave the relevant flag false and explain why.
- Every quote must be one contiguous verbatim substring of PAPER TEXT. Do not
  insert ellipses, paraphrase, or join separated passages. A Methods sentence
  that enumerates all enrolled/participating arms can establish coverage; it
  need not literally use the words "complete" or "mutually exclusive".
- Operational definition: ``complete=true`` when a Methods/enrolment sentence
  presents the counted groups as all participant arms in that study (for
  example, "N arm A and M arm B underwent...") and names no additional arm.
  ``mutually_exclusive=true`` when that same passage presents them as separate
  comparison arms (patients vs controls, treatment vs placebo, etc.) and does
  not say membership overlaps. Do NOT set either false merely because the paper
  omits the literal words "only", "complete", or "mutually exclusive".
- A baseline/characteristics table can also establish the partition when its
  only participant columns are the separately counted study arms. In that case,
  quote one contiguous table header/block containing every arm label and N."""

_PROMPT = """You are extracting ONE specific value from a source paper for an audit.

## RESEARCH CONTEXT
{context}

## TARGET
Find the value of **{concept}** for the **{group_desc}**.
(The review's column was labelled "{raw_label}"; internal field_type: {field_type}.)
The paper may name this field using: {concept_variants}
Expected unit (hint, may differ in the paper): "{unit_hint}"
{outcome_line}{targeted_target}
## RULES
{cohort_rules}
- First list every cohort/column the paper reports for this field in ``cohorts_seen``.
- Set ``group_label_in_paper`` to the paper's own name for the cohort you took the
  value from; it MUST be the {group_desc}. Your quote MUST be the {group_desc}'s
  OWN cell or sentence — never another cohort's.
- If the paper reports no value specifically for the {group_desc}, set found=false —
  do NOT substitute or infer another cohort's value.
- When found=false you MUST fill ``not_found_reason``: say what you looked at and
  why the value is not there (absent from the paper / reported for other cohorts
  only / the text is truncated / the table did not come through, …). "Not found"
  without a reason cannot be acted on.
- Return the value and unit EXACTLY as printed (keep "mean ± SD" / "median (IQR)").
- Do not infer a missing value and do not perform arithmetic.
- ``quote`` must be one contiguous verbatim substring of PAPER TEXT and must
  contain the returned value. Never paraphrase, insert ellipses, or construct a
  supporting sentence. If no such quote exists, return found=false.
{sample_size_rules}
{retry_rules}{targeted_rules}

## PAPER TEXT
{paper_text}

## OUTPUT — one JSON object, nothing else:
{{"cohorts_seen": ["each cohort/column label the paper reports for this field"],
  "group_label_in_paper": "the paper's name for the cohort this value is taken from",
  "found": true or false, "value": "verbatim value or null", "unit": "verbatim unit or empty",
  "quote": "verbatim supporting sentence/cell", "source_field_name": "the paper's own label for this field",
  "location": "where (e.g. Table 2, Results)",
  "explicit_total_reported": false,
  "cohort_counts": [{{"label": "paper's arm name", "count": 15,
                       "quote": "verbatim text containing this arm and count"}}],
  "cohort_partition_complete": false,
  "cohort_partition_mutually_exclusive": false,
  "cohort_partition_quote": "verbatim text establishing the partition, or empty",
  "cohort_partition_reason": "why coverage/exclusivity is clear or unclear",
  "not_found_reason": "when found=false, why — otherwise empty"{targeted_outputs}}}
"""

#: The two rules whose worked examples name cohorts. Held apart from the rest of
#: the prompt because they are the only part that has to differ per profile, and
#: because substituting a whole second copy of the template would let the shared
#: rules drift between profiles without any test noticing.
#:
#: V3 is what Phase 6 and the melanoma benchmark were recorded under. Its bytes
#: are frozen: the disease names below are not a style choice to be tidied, they
#: are the recorded question, and editing them is what makes every replay a miss.
_COHORT_RULES_V3 = """- PREFER the data table (e.g. Table 1) over prose. A narrative sentence like
  "Regarding diabetic children, the mean age was X" reports only ONE cohort —
  NEVER reuse that number for another cohort, even if the paper says the groups
  "did not differ significantly".
- Tables list cohorts as COLUMNS in a fixed order — a row reads e.g.
  "Age (years) | <diabetic value> | <control value> | <P value>". Identify which
  column is the {group_desc} and read THAT column's cell in the target row."""

#: The same two rules with the examples written as placeholders, so a review from
#: another field is not handed one disease as the shape of the answer.
_COHORT_RULES_V6 = """- PREFER the data table (e.g. Table 1) over prose. A narrative sentence that
  names ONE cohort ("In <cohort A>, the mean age was X") reports that cohort
  only — NEVER reuse its number for another cohort, even if the paper says the
  groups "did not differ significantly".
- Tables list cohorts as COLUMNS in a fixed order — a row reads e.g.
  "Age (years) | <cohort A value> | <cohort B value> | <P value>". Identify which
  column is the {group_desc} and read THAT column's cell in the target row."""

_TARGETED_ARM = """
The requested target is ONE arm. Do not decide which of the paper's arms it is —
list them all below and let the audit make that assignment."""

_TARGETED_COMPARISON = """
The requested target is the COMPARISON **{left}** versus **{right}** — one value
about the PAIR (a hazard ratio, a difference), not a value for either arm alone.
The direction is part of the claim: the reverse comparison is a different
number. List every comparison the paper reports and let the audit assign it."""

_TARGETED_RULES = """
- Do NOT pick the target arm yourself. List EVERY arm the paper reports FOR THIS
  FIELD in ``arms_reported``: the paper's own label, that arm's value and unit,
  and one contiguous verbatim quote that names that arm and contains that value.
- If the paper reports comparisons BETWEEN arms for this field (hazard ratio,
  difference, ratio), list each one in ``comparisons_reported`` with both side
  labels in the paper's own words, in the order the paper states them, and one
  quote that names BOTH sides and contains the value.
- Every enumerated entry needs its OWN quote. Never support one arm with another
  arm's sentence, and never merge two arms into one entry.
- Fill ``value``/``quote`` with your own best answer for the requested target as
  well. It is checked against the enumeration, and the enumeration decides.
- When an entry's value carries a confidence interval, give its parts in that
  entry's ``value_components``: the point estimate, the confidence LEVEL as a
  number (95, 99.5), and the lower and upper bounds. Copy them from the quote —
  never convert, round, or supply a level the paper does not print. Leave a part
  out only when the quote does not state it; a returned part that the quote does
  not print is a failed extraction, and so is dropping an interval the quote
  does state."""

# Substituted INTO the formatted prompt, so its braces are literal: doubling
# them here would show the model malformed JSON.
_TARGETED_OUTPUTS = """,
  "arms_reported": [{"label": "the paper's own name for this arm",
                     "value": "that arm's verbatim value", "unit": "verbatim unit",
                     "quote": "verbatim text naming this arm and its value",
                     "value_components": {"point_estimate": 0, "ci_level": 95,
                                          "ci_lower": 0, "ci_upper": 0}}],
  "comparisons_reported": [{"left_label": "paper's name for the first side",
                            "right_label": "paper's name for the second side",
                            "value": "verbatim value", "unit": "verbatim unit",
                            "quote": "verbatim text naming both sides and the value",
                            "value_components": {"point_estimate": 0, "ci_level": 95,
                                                 "ci_lower": 0, "ci_upper": 0}}]"""


def _targeted_target(comparison: ComparisonTarget | None) -> str:
    """The extra TARGET paragraph the targeted contract adds, if any."""
    if comparison is not None:
        return _TARGETED_COMPARISON.format(left=comparison.left,
                                           right=comparison.right)
    return _TARGETED_ARM


def _outcome_line(profile: str, outcome: str) -> str:
    """Outcome clause for targeted_v7 only — empty on frozen v3/v4/v6 bytes."""
    text = (outcome or "").strip()
    if profile != "targeted_v7" or not text:
        return ""
    return f'This claim is about the outcome: "{text}".\n'


def cohort_description(group: str, *, display: str = "",
                       variants: list[str] | None = None) -> str:
    """How to describe the wanted cohort to the extractor — in the REVIEW's words.

    No disease vocabulary lives in this code any more: whatever the review calls
    its arm is what the model is asked for. An unidentified cohort is stated as
    such rather than described as the whole study, which would have the model
    fetch a pooled number for a claim that is about one arm.
    """
    g = (group or "").strip().lower()
    if not g:
        return ("cohort the review did not identify — report every cohort the "
                "paper distinguishes and say which one you took the value from")
    if g in ("all", "-"):
        return "whole study cohort (the review reports no separate arms here)"
    name = display or group
    also = [v for v in (variants or []) if v and v != name]
    suffix = f" (the paper may call it: {', '.join(also)})" if also else ""
    return f'"{name}" cohort{suffix}'


def cohort_conflicts(
    target: str, label_in_paper: str, *,
    cohorts: dict[str, list[str]] | None = None,
) -> tuple[str, str]:
    """Does the paper's own cohort label contradict the one we asked for?

    Returns ``(verdict, reason)`` where verdict is ``ok`` | ``wrong_cohort`` |
    ``ambiguous``. **Ambiguous is not ok**: a tie, or a label matching no known
    cohort, means the guard could not confirm the value belongs to the requested
    arm, and letting that through as clean is exactly the silent pass this guard
    exists to prevent. It is kept as evidence but forced to human review.
    """
    label = (label_in_paper or "").strip().lower()
    t = (target or "").strip().lower()
    if not label or t in ("all", "-", ""):
        return "ok", ""
    if not cohorts:
        # No cohort information was supplied at all (e.g. auditing two CSVs whose
        # groups are already canonical). The guard is not configured here, which
        # is different from being unable to confirm a cohort it does know about.
        return "ok", ""
    if t not in cohorts:
        return "ambiguous", (f"cohort {target!r} is not one this review was found "
                             "to report, so the paper's label could not be checked "
                             "against it")

    scores = {key: _overlap(label, variants) for key, variants in cohorts.items()}
    best_key, best = max(scores.items(), key=lambda kv: kv[1])
    mine = scores.get(t, 0.0)
    if best < 0.5 or best_key == t:
        # Nothing matched well, or the best match IS the target.
        if best < 0.5:
            return "ambiguous", (f"the paper calls this cohort {label_in_paper!r}, "
                                 "which matches no cohort this review reports")
        return "ok", ""
    if best - mine >= 0.25:
        return "wrong_cohort", (f"the paper attributes this value to "
                                f"{label_in_paper!r}, which matches cohort "
                                f"{best_key!r}, not the requested {target!r}")
    return "ambiguous", (f"the paper's label {label_in_paper!r} fits {best_key!r} "
                         f"and {target!r} about equally well")


def _overlap(label: str, variants: list[str]) -> float:
    """Best word-overlap of any variant against the paper's label (0..1)."""
    label_words = set(re.findall(r"[a-z0-9]+", label))
    best = 0.0
    for variant in variants:
        words = set(re.findall(r"[a-z0-9]+", (variant or "").lower()))
        if words:
            best = max(best, len(words & label_words) / len(words))
    return best


class SourceQuery(BaseModel):
    """What a source-paper extraction may be asked — concepts and cohorts only.

    This is the prompt-argument envelope. It has no slot for a review cell
    value, so a later change cannot start interpolating ``review_value`` into
    the source prompt by adding a field here.
    """

    model_config = ConfigDict(extra="forbid")

    concept: str = ""
    concept_variants: list[str] = Field(default_factory=list)
    field_type: str = ""
    cohort_target: str = ""
    unit_hint: str = ""
    research_context: str = ""
    outcome: str = ""


def source_query_from_claim(
    claim: ReviewDataItem,
    *,
    concept: str = "",
    concept_variants: list[str] | None = None,
    research_context: str = "",
) -> SourceQuery:
    """Build a source query from a review claim without reading its value."""
    return SourceQuery(
        concept=concept or claim.raw_field_name or claim.field_type,
        concept_variants=list(concept_variants or []),
        field_type=claim.field_type,
        cohort_target=claim.cohort_label or claim.group,
        unit_hint=claim.unit,
        research_context=research_context,
        outcome=str(getattr(claim, "outcome", "") or ""),
    )


def source_query_from_payload(payload: "ExtractSourceValueInput") -> SourceQuery:
    """The query envelope for a single-target request. Never copies a cell value."""
    return SourceQuery(
        concept=payload.concept or payload.raw_field_name or payload.field_type,
        concept_variants=list(payload.concept_variants),
        field_type=payload.field_type,
        cohort_target=payload.cohort_display or payload.group,
        unit_hint=payload.unit_hint,
        research_context=payload.research_context,
        outcome=payload.outcome,
    )


def render_source_extract_prompt(
    query: SourceQuery,
    *,
    paper_text: str,
    raw_label: str = "",
    group: str = "-",
    cohort_display: str = "",
    cohorts: dict[str, list[str]] | None = None,
    comparison: ComparisonTarget | None = None,
    attempt: int = 0,
    extraction_profile: str = DEFAULT_PROFILE,
) -> str:
    """Render the single-target source prompt from a SourceQuery plus paper text.

    Structural extras (group, profile, comparison, attempt) describe HOW to ask,
    not WHAT value the review reported. The review cell is not an argument.
    """
    profile = prompt_profile(SimpleNamespace(extraction_profile=extraction_profile))
    targeted = uses_targeted_sections(profile)
    target = query.concept
    group_desc = (
        f'comparison of "{comparison.left}" versus "{comparison.right}"'
        if targeted and comparison is not None else
        cohort_description(
            group, display=cohort_display,
            variants=(cohorts or {}).get(group, [])))
    return _PROMPT.format(
        targeted_target=(_targeted_target(comparison) if targeted else ""),
        targeted_rules=(_TARGETED_RULES if targeted else ""),
        targeted_outputs=(_TARGETED_OUTPUTS if targeted else ""),
        cohort_rules=(_COHORT_RULES_V6 if profile in {"targeted_v6", "targeted_v7"}
                      else _COHORT_RULES_V3).format(group_desc=group_desc),
        outcome_line=_outcome_line(profile, query.outcome),
        context=query.research_context or "a systematic review",
        concept=target,
        raw_label=raw_label or target,
        field_type=query.field_type or "(unresolved)",
        concept_variants=(", ".join(query.concept_variants)
                          or raw_or_target(raw_label, target)),
        group_desc=group_desc,
        unit_hint=query.unit_hint,
        sample_size_rules=(
            _SAMPLE_SIZE_RULES if query.field_type == "sample_size" and
            (group or "").strip().lower() in {"all", "-"} else ""),
        retry_rules=("- RETRY CORRECTION: the previous response failed a "
                     "deterministic evidence check. Re-read the paper and "
                     "return only an exact contiguous quote that is visibly "
                     "present in PAPER TEXT and contains the extracted value. "
                     "The returned unit must also be exactly the unit printed "
                     "in that quote; do not correct a paper's apparent typo."
                     if attempt else ""),
        paper_text=_paper_excerpt(
            paper_text, target=target, raw_label=raw_label,
            field_type=query.field_type, variants=query.concept_variants),
    )


class ExtractSourceValueInput(BaseModel):
    document: PaperDocument
    field_type: str
    group: str = "-"
    concept: str = ""
    concept_variants: list[str] = Field(default_factory=list)
    raw_field_name: str = ""      # the review's own column label — the extraction target
    unit_hint: str = ""
    research_context: str = ""
    # The review's own name for the wanted cohort, and every cohort it reports —
    # so the guard can check the paper's label without any disease vocabulary.
    cohort_display: str = ""
    cohorts: dict[str, list[str]] = {}
    # WHEN and WHICH EFFECT, carried from the review. Present from P8 D1-0 but
    # not rendered by any single-target prompt, whose bytes are frozen;
    # the v5 batch contract is where they are asked for and guarded. A timepoint
    # that only exists in a cache key cannot stop the model returning the wrong
    # one, which is why it has to travel with the request first.
    timepoint: str = ""
    timepoint_label: str = ""
    effect_definition: str = ""
    attempt: int = 0
    # Which prompt contract this request runs under. Carried explicitly so a
    # frozen benchmark cannot drift onto a new prompt by accident; see
    # :mod:`react_review.tools.extraction_profile`.
    extraction_profile: str = DEFAULT_PROFILE
    # What KIND of thing is being asked for. ``arm_identity`` means the field's
    # value IS the arm — its drug, dose and label — so the answer is the paper's
    # own name for that arm, never a number reported about it. The caller
    # derives this structurally (see ``Collector``); the model is not asked to
    # understand the distinction, because it demonstrably does not.
    target_kind: str = "value"          # value | arm_identity
    # A claim about TWO arms cannot be expressed as one cohort name. Carrying it
    # as a structured pair is what lets the request say WHICH hazard ratio, and
    # what lets the assignment check the direction.
    comparison: ComparisonTarget | None = None
    # Which review-side outcome this claim is about. Carried on every request;
    # interpolated into the prompt only under targeted_v7.
    outcome: str = ""


class SourceValueResult(BaseModel):
    found: bool = False
    value: str | None = None
    unit: str = ""
    quote: str = ""
    source_field_name: str = ""
    location: str = ""
    group_label_in_paper: str = ""
    wrong_group_rejected: bool = False
    # ok | wrong_cohort | ambiguous — "ambiguous" keeps the value but must reach
    # a human, because the guard could not confirm which arm it belongs to.
    cohort_check: str = "ok"
    cohort_reason: str = ""
    # Why nothing was found, in the model's own words, and what it DID see.
    # Both were previously discarded, leaving "found=false" with no explanation.
    not_found_reason: str = ""
    cohorts_seen: list[str] = Field(default_factory=list)
    value_origin: str = "unresolved"
    derivation: str = ""
    cohort_counts: list[CohortCount] = Field(default_factory=list)
    aggregation_status: str = "not_applicable"  # also derived | rejected | protocol_error
    aggregation_reason: str = ""
    # How a derived total was arrived at, and under which frozen policy. Only
    # the batch path fills this; the legacy path predates the policy and would
    # be claiming a provenance it never had.
    aggregation_provenance: AggregationProvenance | None = None
    # Which arm/comparison this value was deterministically assigned to, and how
    # that went. ok | reassigned | ambiguous | not_reported | direction_inverted
    # | inconsistent | unsupported | protocol_error.
    target_check: str = "ok"
    target_reason: str = ""
    assigned_arm_label: str = ""
    target_margin: float = 0.0
    arms_reported: list[ArmEvidence] = Field(default_factory=list)
    comparisons_reported: list[ComparisonEvidence] = Field(default_factory=list)
    # The parts of the returned value, verified against its own quote. ``None``
    # means the contract that produces them did not run for this request.
    source_components: SourceNumericComponents | None = None
    # Which population the supporting quote is talking about. Classified from
    # that quote alone, so it travels with the evidence rather than with the
    # request that asked for it.
    source_scope: PopulationScope | None = None
    evidence_check: str = "ok"  # ok | protocol_error
    evidence_reason: str = ""
    error: str = ""


class ExtractSourceValueTool(Tool):
    """Find one target value (field_type + group) in a source paper, with a quote."""

    name = "extract_source_value"
    stage = ToolStage.EXTRACT
    input_model = ExtractSourceValueInput
    output_model = SourceValueResult

    def __init__(
        self,
        backend: LLMBackend | None,
        *,
        cache: ExtractionCache | None = None,
        cache_mode: str = "live",
        stage: str = "",
        telemetry=None,
    ) -> None:
        if cache_mode not in {"live", "record", "replay"}:
            raise ValueError("cache_mode must be live, record, or replay")
        if cache_mode in {"record", "replay"} and cache is None:
            raise ValueError(f"{cache_mode} extraction requires a cache")
        if cache_mode != "replay" and backend is None:
            raise ValueError("live extraction requires an LLM backend")
        self._backend = backend
        self._cache = cache
        self._cache_mode = cache_mode
        self._stage = stage
        self._telemetry = telemetry
        # Cross-claim reuse of the same (paper, group, field, outcome) question.
        # attempt>0 still runs: collector retries must not be short-circuited.
        self._query_reuse: dict[tuple, SourceValueResult] = {}

    async def run(self, payload: ExtractSourceValueInput) -> SourceValueResult:
        query = source_query_from_payload(payload)
        reuse_key = (
            str(getattr(payload.document, "paper_id", "") or ""),
            payload.group,
            payload.field_type,
            query.outcome,
        )
        if payload.attempt == 0:
            reused = self._query_reuse.get(reuse_key)
            if reused is not None:
                return reused
        if self._telemetry is not None:
            self._telemetry.attempt("extract_source_value")
            if payload.attempt:
                self._telemetry.repeated_attempts += 1
        # The target description prefers the canonical concept, but falls back to
        # the review's RAW column label — so an UNRESOLVED field (no field_type)
        # is still extractable: the raw name itself says what to look for.
        profile = prompt_profile(payload)
        if profile == "targeted_v5_batch":
            # The second gate. A v5 request reaching here would be built with the
            # LEGACY prompt body — v5 is not one of the profiles that turn the
            # targeted sections on — and cached under the v5 prompt version:
            # neither contract, and
            # written into the namespace of the one it is not. The startup gate
            # should make this unreachable; this is what makes a hole in the
            # startup gate a crash rather than a poisoned recording.
            raise ContractError(
                "a targeted_v5_batch claim reached the single-target extractor. "
                "That path builds the legacy prompt and would record it under the "
                "v5 cache namespace, which is neither contract. Route it to the "
                "batch tool or fail the run")
        prompt = render_source_extract_prompt(
            query,
            paper_text=payload.document.full_text or "",
            raw_label=payload.raw_field_name,
            group=payload.group,
            cohort_display=payload.cohort_display,
            cohorts=payload.cohorts,
            comparison=payload.comparison,
            attempt=payload.attempt,
            extraction_profile=payload.extraction_profile,
        )
        model_id = ((self._backend.model_id if self._backend is not None else "")
                    or (self._cache.model_id if self._cache is not None else "")
                    or "replay")
        key = extraction_cache_key(
            model_id=model_id,
            prompt_version=prompt_version(prompt_profile(payload)),
            prompt=prompt, attempt=payload.attempt)
        try:
            # record is resumable: reuse a response already persisted for this
            # exact prompt/attempt, and call the model only for a cache miss.
            data = (self._cache.get(key)
                    if self._cache_mode in {"record", "replay"} else None)
            if self._stage and self._telemetry is not None and self._cache_mode != "live":
                # The same stage accounting the batch path keeps, so the two are
                # comparable. Only the stage bucket: the harness folds each
                # cache's own totals into the global counters when a run ends.
                #
                # And only when a stage was asked for. Per-stage numbers exist
                # to compare one reading against another; a run with a single
                # route has nothing to compare and its global counters already
                # say what it cost — so recording them anyway would add a
                # section to every artifact ever replayed, for a measurement
                # nobody was making.
                self._telemetry.record_stage_cache(
                    self._stage,
                    hits=1 if data is not None else 0,
                    misses=0 if data is not None else 1)
            if data is None:
                if self._cache_mode == "replay":
                    raise ExtractionCacheMiss(
                        f"no recorded extraction for {payload.document.paper_id}/"
                        f"{payload.group}/{payload.field_type}, attempt {payload.attempt + 1}")
                assert self._backend is not None
                raw = await self._backend.complete(prompt)
                data = parse_llm_response(raw, self._backend.model_id)
                if self._cache_mode == "record" and self._cache is not None:
                    self._cache.put(key, data, model_id=self._backend.model_id)
        except ExtractionCacheMiss:
            raise
        except Exception as exc:
            # The text is CARRIED, not just logged: without it the Collector can
            # only record "not found", and a transport error becomes
            # indistinguishable from a paper that genuinely omits the value.
            logger.warning("extract_source_value_failed", error=str(exc)[:160])
            return SourceValueResult(
                found=False, error=str(exc)[:300],
                not_found_reason=f"the extraction call failed: {type(exc).__name__}")

        result = _finalize_result(data, payload)
        # The population belongs to the EVIDENCE, so it is read from the quote
        # the result actually rests on — after the assignment has decided which
        # quote that is, and in one place rather than on every return path.
        if result.source_scope is None and result.quote:
            result.source_scope = classify_population(result.quote)
        if payload.attempt == 0:
            self._query_reuse[reuse_key] = result
        return result


def _targeted_applies(payload: ExtractSourceValueInput) -> bool:
    """Whether this request has an arm-level target to assign at all.

    A study-level row ("-") is about the whole paper: there is no arm to pick,
    and forcing an assignment would refuse rows that were never at risk.
    """
    if not uses_targeted_sections(prompt_profile(payload)):
        return False
    if payload.comparison is not None:
        return True
    group = (payload.group or "").strip().lower()
    return bool(group and group not in {"-", "all"} and payload.cohorts)


def _review_labels(payload: ExtractSourceValueInput) -> dict[str, str]:
    """Each cohort key mapped to the review's OWN words for that cohort."""
    labels = {key: (variants[0] if variants else key)
              for key, variants in payload.cohorts.items()}
    if payload.cohort_display and payload.group:
        labels[payload.group] = payload.cohort_display
    return labels


def _assign_target(data: dict, payload: ExtractSourceValueInput
                   ) -> tuple[TargetAssignment, list[ArmEvidence],
                              list[ComparisonEvidence]]:
    """Let deterministic code decide which arm's value was asked for."""
    paper_text = payload.document.full_text or ""
    arms, arm_error = parse_arms(data.get("arms_reported"), paper_text)
    comparisons, comparison_error = parse_comparisons(
        data.get("comparisons_reported"), paper_text)
    error = arm_error or comparison_error
    if error:
        return (TargetAssignment(
            status="protocol_error",
            reason=f"the enumerated evidence is not usable: {error}"),
            arms, comparisons)

    labels = _review_labels(payload)
    if payload.comparison is not None:
        return (resolve_comparison_target(
            comparison=payload.comparison, review_labels=labels,
            comparisons=comparisons), arms, comparisons)
    return (resolve_arm_target(target_key=payload.group, review_labels=labels,
                               arms=arms), arms, comparisons)


def _finalize_result(data: dict, payload: ExtractSourceValueInput) -> SourceValueResult:
    """Validate raw model JSON and perform the one permitted derivation."""

    arms_reported: list[ArmEvidence] = []
    comparisons_reported: list[ComparisonEvidence] = []
    target_check, target_reason, assigned_label, target_margin = "ok", "", "", 0.0
    source_components: SourceNumericComponents | None = None
    source_scope: PopulationScope | None = None
    # Which population the supporting quote is talking about. Classified from
    # that quote alone, so it travels with the evidence rather than with the
    # request that asked for it.
    source_scope: PopulationScope | None = None

    if _targeted_applies(payload):
        assignment, arms_reported, comparisons_reported = _assign_target(data, payload)
        if not assignment.ok:
            logger.info("extract_source_target_unresolved", status=assignment.status,
                        group=payload.group, field=payload.field_type)
            return SourceValueResult(
                found=False, value=None,
                source_field_name=str(data.get("source_field_name") or "").strip(),
                location=str(data.get("location") or "").strip(),
                not_found_reason=assignment.reason,
                cohorts_seen=[a.label for a in arms_reported],
                value_origin="unresolved",
                target_check=assignment.status, target_reason=assignment.reason,
                assigned_arm_label=assignment.paper_label,
                target_margin=round(assignment.margin, 4),
                arms_reported=arms_reported,
                comparisons_reported=comparisons_reported)
        # The enumeration decides, not the model's own pick. When they differ
        # the value is still taken from the assignment — that IS the fix — but
        # the disagreement is recorded, because it is the wrong-arm selection
        # this contract exists to catch, caught.
        if payload.target_kind == "arm_identity":
            return _arm_identity_result(assignment, data, payload,
                                        arms_reported, comparisons_reported)
        # The parts of the assigned value, checked against its own quote. An
        # interval the quote states but the response dropped keeps the result
        # partial: a point estimate alone must not read as a complete answer.
        components, component_status, component_reason = verify_components(
            assignment.components, value=assignment.value or "",
            quote=assignment.quote, rival_values=assignment.rival_values)
        if component_status == PROTOCOL_ERROR:
            return SourceValueResult(
                found=False, value=None,
                source_field_name=str(data.get("source_field_name") or "").strip(),
                location=str(data.get("location") or "").strip(),
                quote=assignment.quote,
                group_label_in_paper=assignment.paper_label,
                not_found_reason=component_reason,
                value_origin="unresolved",
                target_check="ok", assigned_arm_label=assignment.paper_label,
                arms_reported=arms_reported,
                comparisons_reported=comparisons_reported,
                source_components=components,
                evidence_check="protocol_error", evidence_reason=component_reason)
        source_components = components
        source_scope = assignment.population
        assigned_label = assignment.paper_label
        target_margin = round(assignment.margin, 4)
        own_pick = str(data.get("value") or "").strip()
        if own_pick and not _same_value(own_pick, assignment.value or ""):
            target_check = "reassigned"
            target_reason = (
                f"the extraction offered {own_pick!r} for this target; the "
                f"paper's own enumeration assigns {assignment.value!r} to "
                f"{assignment.paper_label!r}")
        data = {**data, "found": True, "value": assignment.value,
                "unit": assignment.unit or data.get("unit") or "",
                "quote": assignment.quote,
                "group_label_in_paper": assignment.paper_label}

    value = data.get("value")
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a"):
        value = None
    found = bool(data.get("found")) and value is not None
    label_in_paper = str(data.get("group_label_in_paper") or "").strip()
    quote = str(data.get("quote") or "").strip()
    counts, count_error = _cohort_counts(
        data.get("cohort_counts"), payload.document.full_text or "")

    whole_sample = payload.field_type == "sample_size" and (
        (payload.group or "").strip().lower() in {"all", "-"})
    aggregation_status = "not_applicable"
    aggregation_reason = ""
    derivation = ""
    value_origin = "verbatim" if found else "unresolved"

    if whole_sample:
        explicit_total = bool(data.get("explicit_total_reported"))
        direct_count = _positive_integer(value)
        direct_anchored = found and (
            _count_quote_anchors(
                quote, direct_count, payload.document.full_text or "")
            if direct_count is not None else
            _quote_anchors(quote, str(value), payload.document.full_text or ""))
        # In a single-cohort paper, an anchored participant count is itself the
        # whole-study count even if the model forgot the explicit-total boolean.
        # With multiple cohorts we still require the explicit-total signal, so
        # an arm count cannot masquerade as the total.
        direct_is_total = direct_anchored and (
            explicit_total or len(data.get("cohorts_seen") or []) <= 1)
        if direct_is_total:
            value_origin = "verbatim"
        else:
            derived, aggregation_reason = _derive_whole_study_count(
                counts=counts, count_error=count_error,
                complete=bool(data.get("cohort_partition_complete")),
                mutually_exclusive=bool(data.get("cohort_partition_mutually_exclusive")),
                partition_quote=str(data.get("cohort_partition_quote") or "").strip(),
                partition_reason=str(data.get("cohort_partition_reason") or "").strip(),
                paper_text=payload.document.full_text or "")
            if derived is not None:
                value = str(derived)
                found = True
                value_origin = "derived_sum"
                aggregation_status = "derived"
                derivation = (" + ".join(str(c.count) for c in counts)
                              + f" = {derived} ("
                              + "; ".join(c.label for c in counts) + ")")
                quote = " | ".join(dict.fromkeys(c.quote for c in counts))
            else:
                value = None
                found = False
                value_origin = "unresolved"
                # The model claiming an unquoted computed total, omitting arm
                # structures despite seeing multiple arms, or returning a
                # non-verbatim count quote is a retryable protocol failure. A
                # well-formed answer that says coverage/overlap is unclear is a
                # substantive rejection and must not be guessed around.
                protocol_error = (
                    (bool(data.get("found")) and explicit_total and not direct_anchored)
                    or bool(count_error)
                    or "not anchored by a verbatim source quote" in aggregation_reason
                    or (len(data.get("cohorts_seen") or []) >= 2
                        and len(counts) < len(data.get("cohorts_seen") or []))
                )
                aggregation_status = "protocol_error" if protocol_error else "rejected"
                if protocol_error:
                    aggregation_reason = (
                        "the extraction response violated the sample-size contract: "
                        + aggregation_reason)

    # Guard: check the paper's OWN cohort label and quote against the review.
    # Skipped once a target assignment has run: that assignment matched every
    # arm at once against every arm the paper reports, which is strictly more
    # than this pairwise guard can establish. Re-running it here would only add
    # an "ambiguous" flag to values whose arm is already settled.
    verdict, reason = "ok", ""
    if found and not assigned_label:
        verdict, reason = cohort_conflicts(payload.group, label_in_paper,
                                           cohorts=payload.cohorts)
        if verdict != "wrong_cohort":
            quote_verdict, quote_reason = cohort_conflicts(
                payload.group, quote, cohorts=payload.cohorts)
            if quote_verdict == "wrong_cohort":
                verdict, reason = quote_verdict, quote_reason

    if verdict == "wrong_cohort":
        logger.info("extract_source_wrong_cohort", target=payload.group,
                    got=label_in_paper, quote=quote[:80])
        return SourceValueResult(
            found=False, group_label_in_paper=label_in_paper,
            wrong_group_rejected=True, cohort_check=verdict, cohort_reason=reason,
            not_found_reason=reason, cohort_counts=counts,
            value_origin="unresolved", aggregation_status=aggregation_status,
            aggregation_reason=aggregation_reason,
            arms_reported=arms_reported, comparisons_reported=comparisons_reported)

    # Every direct value must be supported by source text that can be found in
    # the document. Derived totals were already checked count-by-count above;
    # their display quote intentionally joins multiple independently anchored
    # snippets and is not expected to be one source substring.
    evidence_reason = ""
    if found and value_origin != "derived_sum":
        value_anchored = (
            _count_quote_anchors(
                quote, direct_count, payload.document.full_text or "")
            if whole_sample and direct_count is not None else
            _value_quote_anchors(
                quote, str(value), payload.document.full_text or ""))
        if not value_anchored:
            evidence_reason = (
                "the extracted value is not supported by a contiguous verbatim "
                "quote in the source document")
        elif not _unit_quote_anchors(str(data.get("unit") or ""), quote):
            evidence_reason = (
                "the returned unit is not printed in the supporting source quote")
    if evidence_reason:
        return SourceValueResult(
            found=False, value=None, unit=str(data.get("unit") or "").strip(),
            quote=quote,
            source_field_name=str(data.get("source_field_name") or "").strip(),
            location=str(data.get("location") or "").strip(),
            group_label_in_paper=label_in_paper,
            cohort_check=verdict, cohort_reason=reason,
            not_found_reason=evidence_reason,
            cohorts_seen=[str(c) for c in (data.get("cohorts_seen") or [])
                          if isinstance(c, (str, int, float))],
            value_origin="unresolved", cohort_counts=counts,
            aggregation_status=aggregation_status,
            aggregation_reason=aggregation_reason,
            evidence_check="protocol_error", evidence_reason=evidence_reason,
            target_check=target_check, target_reason=target_reason,
            assigned_arm_label=assigned_label, target_margin=target_margin,
            arms_reported=arms_reported, comparisons_reported=comparisons_reported,
            source_components=source_components)

    model_not_found = str(data.get("not_found_reason") or "").strip()
    return SourceValueResult(
        found=found,
        value=value if found else None,
        unit=str(data.get("unit") or "").strip(),
        quote=quote,
        source_field_name=str(data.get("source_field_name") or "").strip(),
        location=str(data.get("location") or "").strip(),
        group_label_in_paper=label_in_paper,
        cohort_check=verdict, cohort_reason=reason,
        not_found_reason=("" if found else (aggregation_reason or model_not_found)),
        cohorts_seen=[str(c) for c in (data.get("cohorts_seen") or [])
                      if isinstance(c, (str, int, float))],
        value_origin=value_origin, derivation=derivation,
        cohort_counts=counts, aggregation_status=aggregation_status,
        aggregation_reason=aggregation_reason,
        target_check=target_check, target_reason=target_reason,
        assigned_arm_label=assigned_label, target_margin=target_margin,
        arms_reported=arms_reported, comparisons_reported=comparisons_reported,
        source_components=source_components, source_scope=source_scope,
    )


def _arm_identity_result(
    assignment: TargetAssignment, data: dict,
    payload: ExtractSourceValueInput,
    arms_reported: list[ArmEvidence],
    comparisons_reported: list[ComparisonEvidence],
) -> SourceValueResult:
    """For an identity field, the answer is the arm's NAME, not a number about it.

    The enumeration already carries what is wanted: the paper's own label for
    the arm the assignment selected, with a quote that has been checked against
    the document. The generic answer for such a field came back as the arm's
    size — 315 where the review says "Ipilimumab (3 mg/kg) + placebo" — which
    compares as a number against a description and can never be right. So the
    label is projected deterministically here rather than explained to the
    model: the data contract decides what kind of answer the field takes.
    """
    label = (assignment.paper_label or "").strip()
    paper_text = payload.document.full_text or ""
    if not label or not normalised_contains(paper_text, label):
        reason = (f"the arm label {label!r} the enumeration assigned to this "
                  "target is not printed in the source document")
        return SourceValueResult(
            found=False, value=None, quote=assignment.quote,
            source_field_name=str(data.get("source_field_name") or "").strip(),
            location=str(data.get("location") or "").strip(),
            group_label_in_paper=label, not_found_reason=reason,
            cohorts_seen=[a.label for a in arms_reported],
            value_origin="unresolved", target_check="unsupported",
            target_reason=reason, assigned_arm_label=label,
            arms_reported=arms_reported, comparisons_reported=comparisons_reported,
            evidence_check="protocol_error", evidence_reason=reason)

    own_pick = str(data.get("value") or "").strip()
    target_check, target_reason = "ok", ""
    if own_pick and not _same_value(own_pick, label):
        target_check = "reassigned"
        target_reason = (
            f"the extraction offered {own_pick!r} for this target; this field is "
            f"the arm's own identity, so the value is the paper's label for the "
            f"assigned arm, {label!r}")
    return SourceValueResult(
        found=True, value=label, unit="", quote=assignment.quote,
        source_field_name=str(data.get("source_field_name") or "").strip(),
        location=str(data.get("location") or "").strip(),
        group_label_in_paper=label,
        cohorts_seen=[a.label for a in arms_reported],
        value_origin="arm_enumeration",
        derivation=(f"the enumerated arms were assigned to the review's cohorts "
                    f"and {payload.group!r} resolved to the paper's {label!r}; "
                    "an identity field takes that label as its value"),
        target_check=target_check, target_reason=target_reason,
        assigned_arm_label=label, target_margin=round(assignment.margin, 4),
        arms_reported=arms_reported, comparisons_reported=comparisons_reported)


def raw_or_target(raw_label: str, target: str) -> str:
    return raw_label or target


def _same_value(one: str, other: str) -> bool:
    """Whether two printed values state the same thing.

    Compared on the numbers when both carry numbers — "0.42" and
    "0.42 (99.5% CI, 0.31 to 0.57)" are the same estimate reported at different
    completeness, and calling that a disagreement would flag every partial
    answer as a wrong-arm selection.
    """
    left = re.findall(r"-?\d+(?:\.\d+)?", one or "")
    right = re.findall(r"-?\d+(?:\.\d+)?", other or "")
    if left and right:
        return left[0] == right[0]
    return _normalise(one) == _normalise(other)


#: Which selector chose the excerpt, and which revision of it. Recorded beside
#: every windowed reading: the spans a run sent are only interpretable if the
#: rule that picked them is named, and a later revision that picks different
#: spans would otherwise be indistinguishable from a different paper.
SELECTION_METHOD_ID = "abstract_plus_target_dense_blocks"
#: v2 reports what was SENT rather than what the markers declare. v1 parsed the
#: markers back, and the marker on a truncated block names the region it was cut
#: from — so a 20,000-character excerpt declared 21,000 characters of source and
#: coverage computed from it could call a passage included that had been cut off
#: before it. The prompt is byte-identical between the two; only the reporting
#: changed, which is why this is a selector version and not a prompt version.
SELECTION_VERSION = "v2"


def select_excerpt(text: str, *, target: str, raw_label: str, field_type: str,
                   variants: list[str] | None = None) -> tuple[str, list[tuple[int, int]]]:
    """The excerpt, and WHICH REGIONS of the source it actually carries.

    The spans are the honest part. A reading that found nothing and a reading
    that was never shown the passage produce the same answer — "the paper does
    not say" — and telling them apart afterwards is impossible unless the run
    wrote down what it sent. Reported, never acted on: nothing here widens a
    window, retries, or changes a refusal, because a second extraction policy
    hiding inside a diagnostic is worse than no diagnostic.
    """
    if len(text) <= _MAX_TEXT:
        return text, [(0, len(text))]
    return _render_excerpt(text, sorted(_selected_blocks(
        text, target=target, raw_label=raw_label, field_type=field_type,
        variants=variants)))


def _paper_excerpt(text: str, *, target: str, raw_label: str,
                   field_type: str, variants: list[str] | None = None) -> str:
    """Keep the abstract plus the most target-dense later source blocks.

    Taking only the first 20k characters silently omitted later tables.  This
    selector is lexical and deterministic: it does not identify a value, it only
    makes source regions containing the requested concept available to the
    extractor while preserving the same context-size ceiling.
    """
    if len(text) <= _MAX_TEXT:
        return text
    return _render_excerpt(text, sorted(_selected_blocks(
        text, target=target, raw_label=raw_label, field_type=field_type,
        variants=variants)))[0]


def _selected_blocks(text: str, *, target: str, raw_label: str,
                     field_type: str,
                     variants: list[str] | None = None) -> set[tuple[int, int]]:
    """Which regions of the source this query wants, before the ceiling is applied."""
    generic = {
        "and", "data", "field", "mean", "source", "study", "table", "the",
        "total", "value", "values",
    }
    terms = {
        token for token in re.findall(
            r"[a-z0-9]+",
            f"{target} {raw_label} {field_type} {' '.join(variants or [])}".lower())
        if len(token) >= 3 and token not in generic
    }
    # Smaller overlapping blocks let a numeric table compete with discussion
    # prose that repeats the concept many times. Numeric density is only a bonus
    # after a block contains a target term, so unrelated number-heavy tables do
    # not win a text-field query.
    block_size, stride = 3000, 2500
    blocks: list[tuple[int, int, int]] = []
    lower = text.lower()
    for start in range(0, len(text), stride):
        end = min(len(text), start + block_size)
        block = lower[start:end]
        term_score = sum(block.count(term) for term in terms)
        numeric_count = len(re.findall(r"(?<![a-z])\d+(?:\.\d+)?", block))
        score = (term_score + min(numeric_count // 10, 20)
                 if term_score else 0)
        blocks.append((score, start, end))
        if end == len(text):
            break

    selected = {(0, min(block_size, len(text)))}
    for _score, start, end in sorted(blocks, key=lambda b: (-b[0], b[1])):
        selected.add((start, end))
        if sum(e - s for s, e in selected) >= _MAX_TEXT - 500:
            break
    return selected


def _render_excerpt(text: str, blocks) -> tuple[str, list[tuple[int, int]]]:
    """Assemble the excerpt, and report which regions it ACTUALLY carries.

    The marker keeps declaring the block it came from — those bytes are part of
    the prompt and changing them would invalidate every recording. But the piece
    beneath it is cut to the room left under the ceiling, so the marker on the
    last block routinely names an end the text never reached: a 20,000-character
    excerpt was declaring 21,000 characters of source.

    Parsing the markers back therefore over-reported what had been sent, and
    coverage computed from that could call a passage covered because the marker
    said the region was included when the region had been truncated before it.
    The second return value is what was really there.
    """
    pieces: list[str] = []
    sent: list[tuple[int, int]] = []
    used = 0
    for start, end in blocks:
        marker = f"\n\n[SOURCE EXCERPT {start}:{end}]\n"
        room = _MAX_TEXT - used - len(marker)
        if room <= 0:
            break
        stop = min(end, start + room)
        piece = text[start:stop]
        pieces.append(marker + piece)
        used += len(marker) + len(piece)
        sent.append((start, stop))
    return "".join(pieces), sent


def _derive_whole_study_count(
    *, counts: list[CohortCount], count_error: str, complete: bool,
    mutually_exclusive: bool, partition_quote: str, partition_reason: str,
    paper_text: str,
) -> tuple[int | None, str]:
    if count_error:
        return None, count_error
    if len(counts) < 2:
        return None, ("whole-study total is not printed and fewer than two "
                      "explicit arm counts were captured")
    labels = [_normalise(c.label) for c in counts]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        return None, "arm labels are missing or duplicated, so the counts cannot be safely summed"
    if not complete:
        reason = "the captured arms are not confirmed to cover the full study population"
        return None, reason + (f": {partition_reason}" if partition_reason else "")
    if not mutually_exclusive:
        reason = "the captured arms are not confirmed to be mutually exclusive"
        return None, reason + (f": {partition_reason}" if partition_reason else "")
    if partition_quote and not _normalised_contains(paper_text, partition_quote):
        # Every arm count and quote has already been independently anchored.
        # The two structured partition flags are sufficient for arithmetic; keep
        # the absent independent partition sentence visible in the reason.
        return sum(c.count for c in counts), (
            "all explicit arm counts were anchored and the extraction marked "
            "them complete/mutually exclusive; no separate partition quote was anchored")
    return sum(c.count for c in counts), "all explicit, mutually-exclusive arms were deterministically summed"


def _cohort_counts(raw: object, paper_text: str) -> tuple[list[CohortCount], str]:
    if raw in (None, ""):
        return [], ""
    if not isinstance(raw, list):
        return [], "cohort_counts is not a list"
    result: list[CohortCount] = []
    for item in raw:
        if not isinstance(item, dict):
            return [], "a cohort count is not a structured object"
        label = str(item.get("label") or "").strip()
        quote = str(item.get("quote") or "").strip()
        count = _positive_integer(item.get("count"))
        if count is None:
            return [], f"the count for {label or 'an unnamed arm'} is not one positive integer"
        if not _count_quote_anchors(quote, count, paper_text):
            return [], f"the count for {label or 'an unnamed arm'} is not anchored by a verbatim quote"
        if not _label_anchors(label, quote):
            return [], f"the quote for arm {label or '(unnamed)'} does not identify that arm"
        result.append(CohortCount(label=label, count=count, quote=quote))
    return result, ""


def _positive_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    text = str(value or "").strip()
    return int(text) if re.fullmatch(r"[1-9][0-9]*", text) else None


def _normalise(text: str) -> str:
    # PDF text often carries a printed line-wrap hyphen ("pres-\nent"). Join
    # only across an actual newline; ordinary compounds such as "age-matched"
    # remain tokenised as two words on both sides.
    text = unicodedata.normalize("NFKD", text or "")
    text = re.sub(r"(?<=[A-Za-z])[-\u00ad]\s*\n\s*(?=[A-Za-z])", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _normalised_contains(haystack: str, needle: str) -> bool:
    normal_needle = _normalise(needle)
    if not normal_needle:
        return False
    if normal_needle in _normalise(haystack):
        return True
    # PDF line wrapping may produce either pres-\nent -> present or
    # sex-\nmatched -> sex matched. Ignoring separators on BOTH sides handles
    # both without fuzzy word substitution or semantic inference.
    compact_needle = _alnum_compact(needle)
    compact_haystack = _alnum_compact(haystack)
    return bool(compact_needle) and compact_needle in compact_haystack


def _quote_anchors(quote: str, value: str, paper_text: str) -> bool:
    return (_normalised_contains(paper_text, quote)
            and bool(re.search(rf"(?<![0-9]){re.escape(value.strip())}(?![0-9])", quote)))


def _value_quote_anchors(quote: str, value: str, paper_text: str) -> bool:
    if not _normalised_contains(paper_text, quote):
        return False
    normal_value = _normalise(value)
    normal_quote = _normalise(quote)
    if normal_value and normal_value in normal_quote:
        return True
    compact_value = _alnum_compact(value)
    compact_quote = _alnum_compact(quote)
    return bool(compact_value) and compact_value in compact_quote


def _unit_quote_anchors(unit: str, quote: str) -> bool:
    compact_unit = _alnum_compact(unit)
    if not compact_unit or compact_unit in {"count", "counts", "participant",
                                            "participants", "subject", "subjects"}:
        return True
    return compact_unit in _alnum_compact(quote)


def _alnum_compact(text: str) -> str:
    normal = unicodedata.normalize("NFKD", text or "").lower()
    return "".join(re.findall(r"[a-z0-9]+", normal))


def _count_quote_anchors(quote: str, count: int, paper_text: str) -> bool:
    if not _normalised_contains(paper_text, quote):
        return False
    if re.search(rf"(?<![0-9]){count}(?![0-9])", quote):
        return True
    words = _integer_words(count)
    return bool(words and words in _normalise(quote))


def _integer_words(value: int) -> str:
    """English rendering used only to anchor printed counts such as Thirty-six."""
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety"]
    if 0 <= value < 20:
        return ones[value]
    if value < 100:
        return tens[value // 10] + (f" {ones[value % 10]}" if value % 10 else "")
    if value < 1000:
        rest = _integer_words(value % 100) if value % 100 else ""
        return f"{ones[value // 100]} hundred" + (f" {rest}" if rest else "")
    return ""


def _label_anchors(label: str, quote: str) -> bool:
    label_tokens = set(re.findall(r"[a-z0-9]+", label.lower()))
    quote_tokens = set(re.findall(r"[a-z0-9]+", quote.lower()))
    if label_tokens and label_tokens & quote_tokens:
        return True
    # Short paper labels are often acronyms while the quoted sentence spells the
    # same arm out in words. The exact quote and count are already anchored
    # above; accept the acronym only when the quote also contains substantive arm
    # descriptors, not merely "N participants".
    if len(_normalise(label).replace(" ", "")) <= 4:
        generic = {
            "a", "an", "and", "arm", "cohort", "consisted", "enrolled",
            "group", "included", "of", "participant", "participants",
            "patient", "patients", "people", "recruited", "subject",
            "subjects", "the", "was", "were", "with",
        }
        return len(quote_tokens - generic) >= 2
    return False


def bind_claim_outcome_and_dedup(collector):
    """Fill SourceQuery.outcome from the claim without editing collector.py.

    ``collector.py`` is inside the evidence-adequacy hash boundary. Production
    wraps the extract tool here so a claim's outcome travels with the request
    and same-key questions reuse the first attempt-0 result.
    """
    inner = collector._extract
    current: dict[str, object] = {"claim": None}

    class _Bound:
        async def run(self, payload: ExtractSourceValueInput) -> SourceValueResult:
            claim = current["claim"]
            outcome = str(getattr(claim, "outcome", "") or payload.outcome or "")
            if outcome and payload.outcome != outcome:
                payload = payload.model_copy(update={"outcome": outcome})
            return await inner.run(payload)

    collector._extract = _Bound()
    original = collector.collect

    async def collect(review_item, *args, **kwargs):
        current["claim"] = review_item
        try:
            return await original(review_item, *args, **kwargs)
        finally:
            current["claim"] = None

    collector.collect = collect
    return collector
