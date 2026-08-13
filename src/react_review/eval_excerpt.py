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
    #: Batches held OUT of the denominator because a witness the key quotes is
    #: not in the extracted text at all. Reported so they cannot vanish: a batch
    #: that leaves every count silently is a batch nobody knows was skipped.
    gold_unlocatable_batches: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"windowed_batches": self.windowed_batches,
                "gold_text_assessable_batches": self.gold_text_assessable_batches,
                "gold_covered_batches": self.gold_covered_batches,
                "gold_missing_batches": self.gold_missing_batches,
                "gold_unlocatable_batches": self.gold_unlocatable_batches}


def tally(batches) -> CoverageTally:
    """Count batches, not witnesses.

    A batch is the unit that was sent, so it is the unit a window can be wrong
    about. Counting witnesses would let one batch with five quotes outvote five
    batches with one, and the selector made a single decision in each case.

    A batch is covered only when EVERY text witness in it was located and sent.
    The point of batching is that one reading answers several claims, so a window
    holding four of five passages has failed the claim resting on the fifth.

    And a batch with a witness the extracted text does not contain is not
    covered — it is not assessable. Dropping the unlocatable witness and judging
    the rest was the earlier behaviour, and it let a batch whose partition
    sentence had been mangled by the PDF extractor be reported as fully covered
    on the strength of the witnesses that survived. What the window did with a
    passage nobody can locate is not knowable, and "covered" is a claim.
    """
    windowed = assessable = covered = missing = unlocatable = 0
    for outcomes, was_windowed in batches:
        if was_windowed:
            windowed += 1
        text = [o for o in outcomes if o.outcome != NON_TEXT_UNASSESSABLE]
        if not text:
            # Nothing lexical to judge. Neither a pass nor a failure.
            continue
        if any(o.outcome == FULLTEXT_UNLOCATABLE for o in text):
            unlocatable += 1
            continue
        assessable += 1
        if any(o.outcome == WINDOW_MISSED for o in text):
            missing += 1
        else:
            covered += 1
    return CoverageTally(windowed, assessable, covered, missing, unlocatable)


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
    #: Readings the key says nothing about, and batches the key expects that the
    #: run never made. Either one means the two are describing different runs,
    #: so both are reported and both make the result unassessable.
    unjudged_run_batches: tuple = ()
    missing_from_run: tuple = ()

    def as_dict(self) -> dict:
        body = {"assessable": self.assessable,
                "unjudged_run_batches": list(self.unjudged_run_batches),
                "missing_from_run": list(self.missing_from_run)}
        if not self.assessable:
            return {**body, "reason": self.reason}
        return {**body, **self.tally.as_dict(), "batches": list(self.batches)}


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
    seen_keys: set[tuple[str, ...]] = set()

    def refuse(reason: str) -> "CoverageReport":
        absent = tuple(sorted(
            index[k].get("batch_id", "/".join(k)) for k in index
            if k not in seen_keys))
        return CoverageReport(False, reason,
                              unjudged_run_batches=tuple(unjudged),
                              missing_from_run=absent)

    for reading in batch_readings:
        key = _claim_key(getattr(reading, "claim_ids", ()))
        entry = index.get(key)
        provenance = getattr(reading, "excerpt_provenance", None)
        if entry is None:
            unjudged.append(reading.execution_id
                            or f"{reading.study_id}/{reading.field_type}")
            continue
        seen_keys.add(key)
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

    # Both directions, or the number is computed over a sample nobody chose.
    # A run that made six of the key's seven batches would otherwise report
    # "6/6 covered" — true of what it did, and silent about what it did not.
    absent = sorted(index[k].get("batch_id", "/".join(k)) for k in index
                    if k not in seen_keys)
    if absent or unjudged:
        return refuse(
            f"the run and the key describe different sets of readings: "
            f"{len(absent)} batch(es) the key expects were never made "
            f"({', '.join(absent[:3]) or 'none'}), and {len(unjudged)} the run "
            f"made are not in the key ({', '.join(unjudged[:3]) or 'none'})")
    return CoverageReport(True, tally=tally(counted), batches=tuple(detail),
                          unjudged_run_batches=(), missing_from_run=())


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
    if gold_path is None or not Path(gold_path).is_file():
        # No key: this benchmark has not judged its windows, and reporting
        # anything would be reporting a measurement nobody made.
        return None
    if not readings:
        # A key exists and the run produced no reading at all. Silence here
        # removed the coverage section from exactly the reports that most need
        # one — the runs where everything failed.
        return CoverageReport(
            False, "the run produced no batched reading, so there is nothing to "
            "judge against a key that expects some",
            missing_from_run=tuple(sorted(
                b.get("batch_id", "") for b in load_gold(gold_path).get("batches") or ())))

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
    """The claims the accuracy harness builds — the SAME function it calls.

    Not a copy that agrees today. The whole point of computing what would be
    sent is that it is what would be sent, and a second builder drifts the first
    time a field is added to the target contract.
    """
    from react_review.eval_accuracy import review_items_for_rows

    return review_items_for_rows(rows, targets)


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


def benchmark_cohorts(profile, rows):
    """The review's own arm labels, keyed exactly as the answer key groups them.

    Shared for the same reason the claim builder is: the cohort display name
    reaches the single-target prompt, so a preflight that omitted it computed
    cache keys for a question the run does not ask — and then reported the
    resulting misses as recordings that did not exist.

    Built by joining the profile's target contract to the answer key on
    audit_id, because the benchmark's ``group`` values ARE the join keys the
    answer key uses: re-slugging the display names here would silently stop the
    review side and the source side from meeting.
    """
    from react_review.normalize.cohorts import CohortLabel, CohortRegistry

    labels: dict[str, "CohortLabel"] = {}
    for row in rows:
        target = profile.targets.get(row.get("audit_id", "")) if profile else None
        display = (getattr(target, "cohort_label", "") or "").strip()
        key = (row.get("group") or "").strip()
        if not display or not key or key in labels:
            continue
        labels[key] = CohortLabel(key=key, display=display,
                                  raw_variants=[display], source="contract")
    return CohortRegistry(labels=list(labels.values())) if labels else None
