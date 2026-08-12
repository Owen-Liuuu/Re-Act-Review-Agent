"""Did the window contain the evidence? — a question only an answer key can ask.

A run cannot ask it. "The paper does not report it" and "the passage was never
sent" produce the same sentence from the extractor, and deciding between them
needs to know where the answer lives, which is precisely what the run does not
have. So the production side records only what it SENT
(:class:`ExcerptProvenance`), and the judging happens here, offline, against
witnesses a human wrote down.

Four outcomes, because three of them were being reported as one:

``covered``                the witness is in the extracted text AND inside a
                           window the run sent
``window_missed``          it is in the extracted text and was NOT sent — the
                           only outcome that indicts the selector
``fulltext_unlocatable``   the key quotes it, and the extracted text does not
                           contain it. Not a selector fault: PDF extraction
                           breaks words across lines, drops table structure and
                           reorders columns
``non_text_unassessable``  the evidence is a figure, a scanned page or an image
                           of a table. Nothing lexical can assess it, and
                           counting it as missing would blame the window for a
                           limit of the extractor

The distinction is not academic. Larkin's own key quotes "A total of 945
patients underwent randomization"; the extracted text reads "underwent ran-
domization" across a line break. A literal check calls that missing evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: How a quote and a document are made comparable before either is searched.
#: Named and versioned because it decides outcomes: a later revision that folds
#: one more artifact would move witnesses between classes, and a coverage number
#: computed under a different rule is a different measurement.
LOCATOR_VERSION = "lexical_v1"

COVERED = "covered"
WINDOW_MISSED = "window_missed"
FULLTEXT_UNLOCATABLE = "fulltext_unlocatable"
NON_TEXT_UNASSESSABLE = "non_text_unassessable"

#: Modalities nothing lexical can assess. A witness declares its own, so a
#: figure is never silently counted as text the window failed to include.
NON_TEXT_MODALITIES = frozenset({"figure", "scan", "image", "graph"})

_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class _Flattened:
    """Normalised text, and where each character came from in the original."""

    text: str
    offsets: list[int]


def _flatten(text: str) -> _Flattened:
    """Casefold and drop every space and hyphen, keeping each character's offset.

    Hyphens and whitespace go together because a hyphen at a line break is
    ambiguous and the ambiguity cannot be resolved from the text. Larkin breaks
    "ran- domization", where the hyphen IS the break; three lines later, in the
    same sentence, it breaks "nivolumab-plus- ipilimumab", where the hyphen
    belongs to the word. A rule that drops line-break hyphens loses the second;
    one that keeps them loses the first; and trying both readings of the whole
    document loses any quote that spans one of each — which the partition
    sentence does, so it was reported missing from a paper that states it.

    Removing both from BOTH sides makes the two cases the same case. The cost is
    that word boundaries stop being enforced, so this is not a general-purpose
    search: it locates a quote a human copied out of the paper, where a false
    match would need tens of characters to coincide.

    Each surviving character keeps the offset it had in the source, so a match
    here is reported as a position THERE. The spans a run recorded are source
    offsets, and comparing them against normalised ones would be comparing two
    different coordinate systems.
    """
    out: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        if char.isspace() or char == "-":
            continue
        out.append(char.lower())
        offsets.append(index)
    return _Flattened("".join(out), offsets)


def _needle(quote: str) -> str:
    """The same normalisation, applied to the side that is searched FOR."""
    return _flatten(quote or "").text


@dataclass(frozen=True)
class WitnessOutcome:
    """One witness, classified — and where it was found, when it was."""

    witness_id: str
    witness_type: str
    outcome: str
    source_start: int | None = None
    source_end: int | None = None

    @property
    def assessable(self) -> bool:
        return self.outcome in (COVERED, WINDOW_MISSED)


def locate(text: str, quote: str) -> tuple[int, int] | None:
    """Where a quote sits in the SOURCE text, or nothing.

    Nothing is a real answer here, not a failure to try harder: a fuzzier
    matcher would turn "the key's quote is not in what we extracted" — a fact
    worth reporting — into a coordinate somebody would then treat as evidence.
    """
    needle = _needle(quote)
    if not needle:
        return None
    flat = _flatten(text)
    at = flat.text.find(needle)
    if at < 0:
        return None
    return flat.offsets[at], flat.offsets[at + len(needle) - 1] + 1


def classify(text: str, *, quote: str, modality: str, spans, witness_id: str = "",
             witness_type: str = "") -> WitnessOutcome:
    """One witness against one run's windows."""
    if (modality or "text").lower() in NON_TEXT_MODALITIES:
        return WitnessOutcome(witness_id, witness_type, NON_TEXT_UNASSESSABLE)
    found = locate(text, quote)
    if found is None:
        return WitnessOutcome(witness_id, witness_type, FULLTEXT_UNLOCATABLE)
    start, end = found
    inside = any(int(a) <= start and end <= int(b) for a, b in spans or ())
    return WitnessOutcome(witness_id, witness_type,
                          COVERED if inside else WINDOW_MISSED, start, end)


@dataclass(frozen=True)
class CoverageTally:
    """The four counts, kept apart.

    One `missing` number would answer a question nobody asked. A window that
    dropped a passage, a quote the extractor mangled and a figure are three
    different problems with three different owners, and only the first is an
    argument about the selector.
    """

    windowed_batches: int = 0
    gold_text_assessable_batches: int = 0
    gold_covered_batches: int = 0
    gold_missing_batches: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"windowed_batches": self.windowed_batches,
                "gold_text_assessable_batches": self.gold_text_assessable_batches,
                "gold_covered_batches": self.gold_covered_batches,
                "gold_missing_batches": self.gold_missing_batches}


def tally(batches) -> CoverageTally:
    """Count batches, not witnesses.

    A batch is the unit that was sent, so it is the unit a window can be wrong
    about. Counting witnesses would let one batch with five quotes outvote five
    batches with one, and the selector made a single decision in each case.

    A batch is covered only when every assessable witness in it is: the point of
    batching is that one reading answers several claims, so a window holding
    four of five passages has failed the claim resting on the fifth.
    """
    windowed = assessable = covered = missing = 0
    for outcomes, was_windowed in batches:
        if was_windowed:
            windowed += 1
        judged = [o for o in outcomes if o.assessable]
        if not judged:
            continue
        assessable += 1
        if all(o.outcome == COVERED for o in judged):
            covered += 1
        else:
            missing += 1
    return CoverageTally(windowed, assessable, covered, missing)


# --- joining a run's readings to the key ------------------------------------

def load_gold(path) -> dict:
    """The witnesses, as published. Read-only: this file is an answer key."""
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def assess(batch_readings, gold: dict, text_for) -> tuple[CoverageTally, list[dict]]:
    """Classify every batch this run made that the key has something to say about.

    Joined on (study, field, shape) — the identity of what was READ, not of the
    claims that consumed it. A batch answering three claims and a batch
    answering four put the same question to the same document, and the window
    was the same decision either way.

    Batches the key does not cover are skipped rather than counted as covered:
    a key silent about a batch has not judged it, and folding silence into the
    numerator is how a coverage figure comes to mean nothing.
    """
    by_key = {(b.get("study_id", ""), b.get("field_type", ""),
               b.get("target_shape", "")): b for b in gold.get("batches") or ()}
    counted: list[tuple[list[WitnessOutcome], bool]] = []
    detail: list[dict] = []
    for reading in batch_readings:
        entry = by_key.get((reading.study_id, reading.field_type,
                            reading.target_shape))
        provenance = getattr(reading, "excerpt_provenance", None)
        if entry is None or provenance is None:
            continue
        text = text_for(reading.study_id)
        if text is None:
            continue
        outcomes = [
            classify(text, quote=w.get("source_quote", ""),
                     modality=w.get("modality", "text"),
                     spans=provenance.spans, witness_id=w.get("witness_id", ""),
                     witness_type=w.get("witness_type", ""))
            for w in entry.get("witnesses") or ()]
        counted.append((outcomes, bool(provenance.windowed)))
        detail.append({
            "batch_id": entry.get("batch_id", ""),
            "execution_id": reading.execution_id,
            "windowed": bool(provenance.windowed),
            "source_chars": provenance.source_chars,
            "excerpt_chars": provenance.excerpt_chars,
            "selection_method_id": provenance.selection_method_id,
            "selection_version": provenance.selection_version,
            "locator_version": LOCATOR_VERSION,
            "witnesses": [{"witness_id": o.witness_id,
                           "witness_type": o.witness_type,
                           "outcome": o.outcome} for o in outcomes],
        })
    return tally(counted), detail
