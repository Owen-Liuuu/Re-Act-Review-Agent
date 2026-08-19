"""Locate then transcribe — two batch calls, not N.

``targeted_v5_batch`` asks for values and quotes in one shot. This pair splits
that into:

1. **locate** (judge): every reading's place in the paper, with an anchor
   quote and grouping fields, and no values.
2. **transcribe**: every value / unit / ``source_field_name``, from those
   quotes only.

Still two round-trips for the whole group. Splitting per claim would give back
every round-trip batching exists to save.

The load-bearing check is not optional: a transcribed value must appear in
**that reading's own quote**. Sibling quotes in the same transcribe prompt do
not count. Batch transcribe sees every quote at once, so the model can copy
claim B's number onto claim A — the SRMA failure where one age filled a BMI
cell, and one ``31±8`` filled four cells. Checking "the value appears in some
quote" would miss it.

``legacy_v3`` / ``targeted_v4`` / ``targeted_v6`` / ``targeted_v5_batch`` stay
frozen. These prompts are new contracts.
"""
from __future__ import annotations

from typing import Any

from react_review.normalize.anchors import normalised_contains, value_supported_by_quote
from react_review.tools.batch_parse import parse_batch
from react_review.tools.batch_prompt import (
    _BODIES,
    _HEADER,
    _READING_FIELDS,
    aggregation_applies,
)

BATCH_LOCATE_VERSION = "extract-source-batch-locate-v1"
BATCH_TRANSCRIBE_VERSION = "extract-source-batch-transcribe-v1"

_LOCATE_RULES = """
## RULES FOR EVERY READING
- Do NOT return a value, a unit, or value_components. Locating is not reading
  the number. A number you write here is discarded.
- ``quote`` is ONE contiguous verbatim passage of PAPER TEXT that names what
  this reading is of (the arm, both sides of a comparison, or the study total)
  and that contains the number you would later transcribe. Not a paraphrase,
  not two pieces joined. If no such passage exists, leave the reading out.
- ``population_phrase``: the paper's OWN words for which people this counts,
  taken from beside the number. Leave it empty if the paper does not say.
- ``timepoint_phrase`` and ``timepoint_quote``: the same rule, for WHEN.
- ``effect_definition``: the paper's own words for what the number measures,
  when it names one.
- Do not compute anything. Do not fill a gap from elsewhere in the paper.

## PAPER TEXT
{paper_text}

## OUTPUT — one JSON object, nothing else:
{{"readings": [{{{reading_fields}
    "quote": "one contiguous verbatim passage naming this reading and containing its number",
    "population_phrase": "the paper's words for which people, or empty",
    "timepoint_phrase": "the paper's words for when, or empty",
    "timepoint_quote": "passage carrying those words, if not in quote",
    "effect_definition": "the paper's words for what this measures, or empty"}}],
  "nothing_reported_reason": "when readings is empty, what you looked at and why it is not there"}}
"""

_TRANSCRIBE = """You are transcribing numbers from passages already located in a source paper.

## FIELD
**{concept}** — the review's column was labelled "{raw_label}"
(internal field_type: {field_type}).
Expected unit (a hint; the paper may print another): "{unit_hint}"

## PASSAGES
Each block is ONE reading. Transcribe ONLY from that block. A number that
appears in another block is a different reading — copying it here is the
error this step exists to catch.

{passages}

## RULES
- ``index`` must be the number in brackets above.
- ``value`` and ``unit`` exactly as printed in THAT passage. Keep
  "mean ± SD", "median (IQR)", and any interval as the paper writes it.
- ``source_field_name``: the paper's own label for this field, taken from
  THAT passage (a table header, a sentence subject). Empty if the passage
  does not name it.
- ``value_components``: for a value with a confidence interval, its parts
  copied from THAT passage. Omit a part the passage does not print.
- Do not compute. Do not read a number out of a neighbouring passage.

## OUTPUT — one JSON object, nothing else:
{{"readings": [{{"index": 0, "value": "verbatim value", "unit": "verbatim unit or empty",
                 "source_field_name": "the paper's own label, or empty",
                 "value_components": {{"point_estimate": 0, "ci_level": 95,
                                       "ci_lower": 0, "ci_upper": 0}}}}]}}
"""


def build_batch_locate_prompt(
    *, target_shape: str, context: str, concept: str, raw_label: str,
    field_type: str, concept_variants: str, unit_hint: str, paper_text: str,
    timepoint_label: str = "",
) -> str:
    """Ask where each reading lives. Values are not part of this question."""
    if target_shape not in _BODIES:
        raise ValueError(f"unknown target shape {target_shape!r}")
    # Locate never asks for aggregation components: those are numbers, and
    # numbers belong to transcribe. An aggregable study claim still locates
    # the printed total (if any) as a reading.
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
            + _BODIES[target_shape]
            + _LOCATE_RULES.format(
                paper_text=paper_text,
                reading_fields=_READING_FIELDS[target_shape]))


def format_passages(located: list[dict[str, Any]]) -> str:
    """The numbered blocks the transcribe prompt is built from."""
    blocks: list[str] = []
    for index, item in enumerate(located):
        who = (item.get("arm_label") or item.get("scope_label")
               or " / ".join(filter(None, [item.get("left_label"),
                                           item.get("right_label")]))
               or f"reading {index}")
        quote = str(item.get("quote") or "").strip()
        blocks.append(f"[{index}] {who}\n\"\"\"\n{quote}\n\"\"\"")
    return "\n\n".join(blocks) if blocks else "(no passages were located)"


def build_batch_transcribe_prompt(
    *, concept: str, raw_label: str, field_type: str, passages: str,
    unit_hint: str = "",
) -> str:
    """Ask for the numbers in already-located passages. One call, every quote."""
    return _TRANSCRIBE.format(
        concept=concept, raw_label=raw_label,
        field_type=field_type or "(unresolved)",
        unit_hint=unit_hint, passages=passages)


def parse_locate(raw: object, document: str) -> tuple[list[dict[str, Any]], str]:
    """Keep every located passage that is actually in the document.

    Values, if the model wrote any, are stripped: locate is not allowed to
    be the source of a number.
    """
    if not isinstance(raw, dict):
        return [], "the locate response is not a JSON object"
    readings = raw.get("readings")
    if readings is None:
        return [], "the locate response carries no `readings` list"
    if not isinstance(readings, list):
        return [], "`readings` is not a list"
    kept: list[dict[str, Any]] = []
    for item in readings:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        if not quote or not normalised_contains(document, quote):
            continue
        cleaned = dict(item)
        cleaned["quote"] = quote
        cleaned.pop("value", None)
        cleaned.pop("unit", None)
        cleaned.pop("value_components", None)
        kept.append(cleaned)
    return kept, str(raw.get("nothing_reported_reason") or "").strip()


def own_quote_supports_value(
    value: str, own_quote: str, sibling_quotes: list[str],
) -> tuple[bool, str]:
    """Whether THIS reading's quote supports the value — siblings do not count.

    The second argument is the load-bearing one. A value that only appears in
    another claim's quote is the cross-row copy this design exists to refuse.
    """
    if value_supported_by_quote(own_quote, value):
        return True, ""
    if any(value_supported_by_quote(other, value) for other in sibling_quotes
           if other != own_quote):
        return False, (
            f"the value {value!r} appears only in another reading's quote, "
            "not in this reading's own quote")
    return False, f"the quote does not contain the value {value!r}"


def merge_located_and_transcribed(
    located: list[dict[str, Any]], transcribed: object, document: str, *,
    target_shape: str,
) -> tuple[Any, dict[int, str]]:
    """Stitch transcribed values onto located quotes, then parse as a v5 batch.

    Returns ``(BatchReading, source_field_names_by_index)``. Field names travel
    beside the reading rather than on :class:`BatchEntry`: ``schemas/batch.py``
    is inside the frozen evaluator boundary.
    """
    field_names: dict[int, str] = {}
    if not isinstance(transcribed, dict):
        reading = parse_batch({"readings": []}, document, target_shape=target_shape)
        reading.batch_error = "the transcribe response is not a JSON object"
        return reading, field_names
    rows = transcribed.get("readings")
    if not isinstance(rows, list):
        reading = parse_batch({"readings": []}, document, target_shape=target_shape)
        reading.batch_error = "the transcribe response carries no `readings` list"
        return reading, field_names

    by_index: dict[int, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        by_index[index] = item

    siblings = [str(loc.get("quote") or "") for loc in located]
    merged: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, loc in enumerate(located):
        own_quote = str(loc.get("quote") or "")
        row = by_index.get(index)
        if row is None:
            rejected.append({"index": index,
                             "reason": "no transcribed value was returned "
                                       "for this located passage"})
            continue
        value = row.get("value")
        value = None if value is None else str(value).strip() or None
        if value is None:
            rejected.append({"index": index,
                             "reason": "the transcribed reading carries no value"})
            continue
        ok, reason = own_quote_supports_value(
            value, own_quote, [q for i, q in enumerate(siblings) if i != index])
        if not ok:
            rejected.append({"index": index, "reason": reason})
            continue
        entry = dict(loc)
        entry["value"] = value
        entry["unit"] = str(row.get("unit") or "").strip()
        if row.get("value_components") is not None:
            entry["value_components"] = row.get("value_components")
        name = str(row.get("source_field_name") or "").strip()
        if name:
            # Keyed by the merged-list index, which parse_batch stores as
            # ``raw_index``. Located index would drift once a sibling is dropped.
            field_names[len(merged)] = name
        merged.append(entry)

    reading = parse_batch({"readings": merged}, document, target_shape=target_shape,
                          aggregable=aggregation_applies(target_shape, ""))
    reading.rejected = list(rejected) + list(reading.rejected)
    return reading, field_names
