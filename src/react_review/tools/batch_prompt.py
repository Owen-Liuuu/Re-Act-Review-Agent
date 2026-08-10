"""The v5 batch contract: ask for every reading, once.

Phase 7 asked one question per cell. The model answered by enumerating all the
arms anyway, and the audit consumed exactly one of them — so three claims about
one table cost three readings of it, and the three answers could disagree with
each other for no reason except that they were three separate acts of reading.

v5 asks once and keeps everything:

*Every reading, not the best one.* The model is told not to choose. Choosing is
what the deterministic assignment does afterwards, from evidence it can check.

*The same arm may appear more than once.* 314 allocated to a group and 313
analysed in it are two readings of one arm. A contract that treats a repeated
arm label as a duplicate cannot express the very case this phase exists to fix.

*By shape.* An arm batch asks only for arms, a comparison batch only for
comparisons, a study batch only for whole-study values. Asking every batch for
arms AND comparisons AND populations grows with the square of the arms and
spends more on output tokens than the batching saves in calls.

*Evidence, not assertion.* A population or timepoint the paper does not print
next to the value is not that value's population or timepoint. The model may
say where it read it; deterministic code then checks that the words are really
there and really adjacent (see ``tools/evidence_binding``).
"""
from __future__ import annotations

from react_review.schemas.batch import ARM, COMPARISON, STUDY

BATCH_PROFILE = "targeted_v5_batch"

_HEADER = """You are reading ONE field out of a source paper for an audit.

## RESEARCH CONTEXT
{context}

## FIELD
**{concept}** — the review's column was labelled "{raw_label}"
(internal field_type: {field_type}). The paper may call it: {concept_variants}
Expected unit (a hint; the paper may print another): "{unit_hint}"
{timepoint_line}
## WHAT TO RETURN
List EVERY reading of this field that the paper reports. Do NOT choose one.
Choosing is done afterwards, from the evidence you supply, by code that checks
it — so a reading you leave out cannot be recovered, and a reading you invent
will be caught and will invalidate the rest.
"""

_ARM_BODY = """
A reading is ONE ARM, at ONE POPULATION, at ONE TIMEPOINT.

The same arm appears more than once whenever the paper reports it more than
once. "314 were assigned to the combination group" and "Combination (N = 313)"
in an analysis table are TWO readings of ONE arm — list both, each with its own
evidence. Never merge them, never pick between them, never drop one because it
looks like a duplicate.

Return them in ``readings``. Do not return comparisons between arms here.
"""

_COMPARISON_BODY = """
A reading is ONE COMPARISON between two arms, at one population and timepoint —
a hazard ratio, a difference, a ratio. The direction is part of it: A versus B
and B versus A are different numbers, and swapping them is not a presentation
choice.

Return them in ``readings``. Do not return single-arm values here.
"""

_STUDY_BODY = """
A reading is ONE WHOLE-STUDY value — a total, a design statement, a figure the
paper reports about the study rather than about an arm. If the paper gives such
a total for more than one population (all randomised, all treated, all
analysed), list each as its own reading.

Return them in ``readings``. Do not return per-arm values here; if the paper
prints only per-arm counts and no total, say so with an empty list rather than
adding the arms up. Arithmetic is not reading.
"""

#: Only for whole-study COUNTS. People partition; means, rates and hazard ratios
#: do not, so no other field may offer components to be summed.
_STUDY_COUNT_BODY = """
A reading is ONE WHOLE-STUDY count — how many people the paper reports for the
study as a whole, at one population (all randomised, all treated, all analysed).
If it prints such a total for more than one population, list each separately.

Return those in ``readings``.

## SEPARATELY: THE ARMS, AS COMPONENTS

Whether or not the paper prints a total, also report every per-arm count it
states, in ``cohort_counts`` — NOT in ``readings``. These are not answers; they
are components that code may add up, and only if you can show the paper says
they cover everyone once each.

**You must not add them up.** Do not report a sum, a subtotal, a remainder, or
a number you worked out. If you return a total, it must be one the paper prints,
and ``quote`` must be the passage that prints it. A computed number presented as
a quoted one is the single worst thing you can return here, and it is checked.

For each component give ``arm_label``, ``count`` (a whole number), ``quote``
(one contiguous verbatim passage naming that arm and printing that number), and
``population_phrase`` (the paper's own words for which people it counts). All
components must be the same population: do not put a count of those randomised
and a count of those analysed in the same list.

Then assess the set in ``partition``:
- ``complete``: true only if the paper shows these arms are ALL the groups that
  population was divided into. Not "it looks like all of them".
- ``mutually_exclusive``: true only if the paper shows no person is in two of
  them.
- ``quote``: the contiguous verbatim passage that shows it — the randomisation
  sentence, the flow-diagram caption, the table header. **A true without a
  locatable quote counts as a false**, so if you have no passage, say false.
- ``reason``: what in that passage establishes it.

If you are not sure, answer false. An honest false costs one refused claim; a
hopeful true silently corrupts a number nobody will re-check.
"""

_RULES = """
## RULES FOR EVERY READING
- ``value`` and ``unit`` exactly as printed. Keep "mean ± SD", "median (IQR)",
  and any interval as the paper writes it. Do not convert, round or normalise.
- ``quote`` is ONE contiguous verbatim passage of PAPER TEXT containing that
  value and naming what it is a reading of. Not a paraphrase, not two pieces
  joined, not a sentence you assembled. If no such passage exists, leave the
  reading out.
- ``population_phrase``: the paper's OWN words for which people this counts
  ("underwent randomization", "analysis population", "safety population",
  "received at least one dose"), taken from beside the value itself. Leave it
  empty if the paper does not say there; an empty field is an honest answer and
  a guessed one is not, and a phrase carried over from another part of the
  paper describes a different set of people than the one you are counting.
- ``timepoint_phrase`` and ``timepoint_quote``: the same rule, for WHEN
  ("at baseline", "median follow-up 12 months", "at 5 years").
- ``effect_definition``: the paper's own words for what the number measures
  ("hazard ratio for death or disease progression"), when it names one.
- ``value_components``: for a value with a confidence interval, its parts —
  point estimate, the LEVEL as a number (95, 99.5), lower and upper bound —
  copied from the quote. Omit a part the quote does not print.
- Do not compute anything. Do not fill a gap with a number from elsewhere in
  the paper. A reading you cannot support with one passage does not belong in
  the list.

## PAPER TEXT
{paper_text}

## OUTPUT — one JSON object, nothing else:
{{"readings": [{{{reading_fields}
    "value": "verbatim value", "unit": "verbatim unit or empty",
    "quote": "one contiguous verbatim passage containing this value",
    "population_phrase": "the paper's words for which people, or empty",
    "timepoint_phrase": "the paper's words for when, or empty",
    "timepoint_quote": "passage carrying those words, if not in quote",
    "effect_definition": "the paper's words for what this measures, or empty",
    "value_components": {{"point_estimate": 0, "ci_level": 95,
                          "ci_lower": 0, "ci_upper": 0}}}}],{aggregation_fields}
  "nothing_reported_reason": "when readings is empty, what you looked at and why it is not there"}}
"""

_AGGREGATION_FIELDS = """
  "cohort_counts": [{"arm_label": "the paper's own name for this arm",
    "count": 0, "quote": "one contiguous verbatim passage naming the arm and printing the number",
    "population_phrase": "the paper's words for which people this counts"}],
  "partition": {"complete": false, "mutually_exclusive": false,
    "quote": "the contiguous verbatim passage that shows it, or empty",
    "reason": "what in that passage establishes it"},"""

_READING_FIELDS = {
    ARM: '\n    "arm_label": "the paper\'s own name for this arm",',
    COMPARISON: ('\n    "left_label": "the paper\'s own name for the first side",'
                 '\n    "right_label": "the paper\'s own name for the second side",'),
    STUDY: '\n    "scope_label": "what this total covers, in the paper\'s words",',
}

_BODIES = {ARM: _ARM_BODY, COMPARISON: _COMPARISON_BODY, STUDY: _STUDY_BODY}

#: Which (shape, field) pairs may offer components for a deterministic sum. Kept
#: as data so that widening it is a visible decision; see
#: configs/aggregation/safe_sum_v1.json, which decides whether the sum happens.
AGGREGABLE = {(STUDY, "sample_size")}


def aggregation_applies(target_shape: str, field_type: str) -> bool:
    return (target_shape, field_type) in AGGREGABLE


def build_batch_prompt(
    *, target_shape: str, context: str, concept: str, raw_label: str,
    field_type: str, concept_variants: str, unit_hint: str, paper_text: str,
    timepoint_label: str = "",
) -> str:
    """The prompt for one batched reading of one field, in one shape."""
    if target_shape not in _BODIES:
        raise ValueError(f"unknown target shape {target_shape!r}")
    aggregable = aggregation_applies(target_shape, field_type)
    timepoint_line = (
        f"The review reports this at: **{timepoint_label}** — a reading at "
        "another timepoint is a different number, so give the paper's own words "
        "for the timepoint of each reading.\n"
        if timepoint_label else "")
    return (_HEADER.format(context=context or "a systematic review",
                           concept=concept, raw_label=raw_label,
                           field_type=field_type or "(unresolved)",
                           concept_variants=concept_variants or concept,
                           unit_hint=unit_hint, timepoint_line=timepoint_line)
            + (_STUDY_COUNT_BODY if aggregable else _BODIES[target_shape])
            + _RULES.format(
                paper_text=paper_text,
                reading_fields=_READING_FIELDS[target_shape],
                aggregation_fields=_AGGREGATION_FIELDS if aggregable else ""))
