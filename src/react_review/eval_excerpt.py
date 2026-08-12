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


def locate_all(text: str, quote: str) -> list[tuple[int, int]]:
    """EVERY place the quote occurs, in source coordinates.

    Every, not the first. Papers repeat themselves — an abstract states a
    hazard ratio and the results state it again — and a window that kept the
    results but not the abstract is a window that showed the evidence. Taking
    only the first occurrence would call that a miss and blame the selector for
    a passage it did include.
    """
    needle = _needle(quote)
    if not needle:
        return []
    flat = _flatten(text)
    found: list[tuple[int, int]] = []
    at = flat.text.find(needle)
    while at >= 0:
        found.append((flat.offsets[at], flat.offsets[at + len(needle) - 1] + 1))
        at = flat.text.find(needle, at + 1)
    return found


def locate(text: str, quote: str) -> tuple[int, int] | None:
    """The first place the quote occurs, or nothing.

    Nothing is a real answer here, not a failure to try harder: a fuzzier
    matcher would turn "the key's quote is not in what we extracted" — a fact
    worth reporting — into a coordinate somebody would then treat as evidence.
    """
    found = locate_all(text, quote)
    return found[0] if found else None


def classify(text: str, *, quote: str, modality: str, spans, witness_id: str = "",
             witness_type: str = "") -> WitnessOutcome:
    """One witness against one run's windows."""
    if (modality or "text").lower() in NON_TEXT_MODALITIES:
        return WitnessOutcome(witness_id, witness_type, NON_TEXT_UNASSESSABLE)
    found = locate_all(text, quote)
    if not found:
        return WitnessOutcome(witness_id, witness_type, FULLTEXT_UNLOCATABLE)
    windows = [(int(a), int(b)) for a, b in spans or ()]
    for start, end in found:
        if any(a <= start and end <= b for a, b in windows):
            return WitnessOutcome(witness_id, witness_type, COVERED, start, end)
    start, end = found[0]
    return WitnessOutcome(witness_id, witness_type, WINDOW_MISSED, start, end)


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


def _claim_key(ids) -> tuple[str, ...]:
    """The identity a run and a key can be joined on.

    The claims a reading answered, normalised. Not (study, field, shape): that
    triple is not an identity — one study reports one field at several
    timepoints, under several column labels, in several units, and every one of
    those is a separate batch that would collide into a single gold entry and be
    judged against the wrong witnesses.

    `claim_ids` and `audit_ids` are the two ends of the same fact, and the
    benchmark already carries both.
    """
    return tuple(sorted(str(i).strip() for i in (ids or ()) if str(i).strip()))


@dataclass(frozen=True)
class CoverageReport:
    """Either four counts, or a reason there are none.

    Never zeros standing in for a reason. All-zero counts read as "no window
    problem", which is the most dangerous thing this could say when what
    actually happened is that no batch could be judged at all.
    """

    assessable: bool
    reason: str = ""
    tally: CoverageTally | None = None
    batches: tuple = ()
    #: Readings the key says nothing about. Reported rather than skipped: a key
    #: that silently covers half a run yields a coverage figure computed over a
    #: sample nobody chose.
    unjudged_run_batches: tuple = ()

    def as_dict(self) -> dict:
        if not self.assessable:
            return {"assessable": False, "reason": self.reason,
                    "unjudged_run_batches": list(self.unjudged_run_batches)}
        return {"assessable": True, **self.tally.as_dict(),
                "unjudged_run_batches": list(self.unjudged_run_batches),
                "batches": list(self.batches)}


class GoldError(ValueError):
    """An excerpt key that cannot be used as given."""


def index_gold(gold: dict) -> dict[tuple[str, ...], dict]:
    """Gold entries by the claims they judge, refusing an ambiguous key."""
    index: dict[tuple[str, ...], dict] = {}
    for entry in gold.get("batches") or ():
        key = _claim_key(entry.get("audit_ids"))
        if not key:
            raise GoldError(
                f"gold batch {entry.get('batch_id', '?')!r} names no audit_ids, "
                "so nothing in a run can be matched to it")
        if key in index:
            raise GoldError(
                f"gold batches {index[key].get('batch_id', '?')!r} and "
                f"{entry.get('batch_id', '?')!r} both claim {list(key)}. One set "
                "of claims has one reading, so one of these would silently judge "
                "the other's witnesses")
        index[key] = entry
    return index


def assess(batch_readings, gold: dict, text_for, *, sha_for=None) -> CoverageReport:
    """Classify every batch this run made that the key has something to say about.

    Joined on the claims, then CHECKED against study, field and shape: the claim
    set is the identity, and a disagreement about what those claims are means the
    key describes a different reading than the one that happened.

    The document is bound by hash, not by name. Offsets are only meaningful
    against the exact text they were computed from, and a PDF extractor that
    changes its output between versions would otherwise move every witness while
    every number still looked plausible.
    """
    index = index_gold(gold)
    expected_sha = str(gold.get("document_sha256") or "")
    counted: list[tuple[list[WitnessOutcome], bool]] = []
    detail: list[dict] = []
    unjudged: list[str] = []

    def refuse(reason: str) -> "CoverageReport":
        return CoverageReport(False, reason,
                              unjudged_run_batches=tuple(unjudged))

    for reading in batch_readings:
        key = _claim_key(getattr(reading, "claim_ids", ()))
        entry = index.get(key)
        provenance = getattr(reading, "excerpt_provenance", None)
        if entry is None:
            unjudged.append(reading.execution_id
                            or f"{reading.study_id}/{reading.field_type}")
            continue
        batch_id = entry.get("batch_id", "?")
        if provenance is None:
            return refuse(f"batch {batch_id} recorded no excerpt provenance, so "
                          "what it sent is unknown")
        for field, expected in (("study_id", entry.get("study_id")),
                                ("field_type", entry.get("field_type")),
                                ("target_shape", entry.get("target_shape"))):
            actual = getattr(reading, field, "")
            if expected and actual != expected:
                return refuse(
                    f"gold batch {batch_id} judges claims {list(key)} as "
                    f"{field}={expected!r}, and the run read them as {actual!r}. "
                    "The key describes a different reading")

        text = text_for(reading.study_id)
        if not text:
            return refuse(f"no extracted text for {reading.study_id}, so no "
                          "witness can be located and nothing about the window "
                          "can be said")
        if expected_sha and sha_for is not None:
            actual_sha = str(sha_for(reading.study_id) or "")
            if actual_sha != expected_sha:
                return refuse(
                    f"the gold's offsets were computed from document "
                    f"{expected_sha[:16]} and {reading.study_id} extracted to "
                    f"{actual_sha[:16] or '(nothing)'}. Comparing them would "
                    "judge one document's spans against another's witnesses")

        outcomes = [
            classify(text, quote=w.get("source_quote", ""),
                     modality=w.get("modality", "text"),
                     spans=provenance.spans, witness_id=w.get("witness_id", ""),
                     witness_type=w.get("witness_type", ""))
            for w in entry.get("witnesses") or ()]
        counted.append((outcomes, bool(provenance.windowed)))
        detail.append({
            "batch_id": batch_id,
            "execution_id": reading.execution_id,
            "claim_ids": list(key),
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

    if not counted:
        return refuse("no batch this run made appears in the excerpt key")
    return CoverageReport(True, tally=tally(counted), batches=tuple(detail),
                          unjudged_run_batches=tuple(unjudged))


# --- what a run, or a run that has not happened yet, would be judged on ------

def coverage_for_run(results, studies, benchmark, gold_path) -> "CoverageReport | None":
    """The four counts for a finished run, or nothing when there is no key.

    Lives here rather than in the runner because a test cannot import that
    module: it replaces `sys.stdout` at import time, which destroys pytest's
    capture for the whole session. That is reason enough on its own, but the
    join between a run's readings and an answer key is also not a property of
    one script.
    """
    from pathlib import Path

    from react_review.agents.collector import _document_sha256
    from react_review.retrieval.local_pdf import _pdf_text

    readings = list(getattr(results, "batch_readings", []) or [])
    if gold_path is None or not Path(gold_path).is_file() or not readings:
        return None

    paths = {s.study_id: (Path(benchmark) / s.source_pdf) for s in studies
             if getattr(s, "source_pdf", "")}
    texts: dict[str, str] = {}

    def text_for(study_id: str):
        if study_id not in texts:
            path = paths.get(study_id)
            # The same extraction the run used. Any other one produces offsets
            # into a document nothing reads, silently compared against the run's
            # own spans.
            texts[study_id] = _pdf_text(path) if path and path.is_file() else ""
        return texts[study_id] or None

    def sha_for(study_id: str):
        text = text_for(study_id)
        return _document_sha256(text) if text else ""

    return assess(readings, load_gold(gold_path), text_for, sha_for=sha_for)


def benchmark_reviews(rows, targets):
    """The claims the accuracy harness builds, from the same two files.

    Shared so that a dry run and a real run group the same claims. A second copy
    would drift, and the whole point of computing what WOULD be sent is that it
    is what would be sent.
    """
    from react_review.eval_accuracy import _target_scope
    from react_review.schemas.evidence import ReviewDataItem

    built = []
    for row in rows:
        target = (targets or {}).get(row.get("audit_id", ""))
        extra = {} if target is None else {
            "raw_field_name": target.raw_field_name,
            "cohort_label": target.cohort_label,
            "timepoint": target.timepoint or "single",
            "population_scope": _target_scope(target),
            "population_scope_source": getattr(target, "population_scope_source", ""),
        }
        built.append(ReviewDataItem(
            review_data_id=row.get("audit_id") or "", study_id=row["study_id"],
            group=(row.get("group") or "-"), field_type=row["field_type"],
            value=(row.get("review_value") or None), unit=(row.get("unit") or ""),
            column_header=(row.get("column_header") or ""), **extra))
    return built


class _Unreachable:
    """A tool endpoint a dry run must never touch, so it says so if reached."""

    model_id = "dry-run"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        raise AssertionError(
            "a dry run computes what WOULD be sent and never sends it")

    async def retrieve(self, reference):
        raise AssertionError("a dry run reads its papers from disk")


def dry_run_collector(contract, knowledge):
    """A Collector wired exactly as production wires it, minus any way out.

    The selector's terms come from the knowledge base's concept and variants,
    from the target contract's raw field name and from the review's own column
    label. Approximating any of them — an early version used
    `field_type.replace("_", " ")` — makes the resulting coverage figure a
    property of the approximation rather than of the run.
    """
    from react_review.production import build_collector
    from react_review.tools.extract import FetchFullTextTool
    from react_review.tools.extract_batch import ExtractSourceBatchTool
    from react_review.tools.extract_source import ExtractSourceValueTool
    from react_review.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Unreachable()))
    registry.register(ExtractSourceValueTool(_Unreachable()))
    registry.register(ExtractSourceBatchTool(_Unreachable()))
    return build_collector(registry, contract=contract, knowledge=knowledge)


def planned_batches(collector, claims, route):
    """Which batches a run over these claims would make, in order."""
    from react_review.tools.batch_group import group_claims

    return group_claims([c for c in claims if collector.route_for(c) == route])
