"""Cohorts, discovered from the review rather than assumed.

The predecessor mapped every arm label onto a fixed ``t1dm | control | all``
vocabulary and returned ``all`` for anything it did not recognise. On a review
with Treatment and Placebo arms that made BOTH arms the same cohort, which made
their claims share a join key, which let the audit pair one arm's value against
the other's evidence — with no error, no flag, and a confident verdict. Nothing
in the pipeline could notice, because nothing had recorded that a distinction
had been lost.

So the cohorts are whatever the review's own table calls them. A small alias
file maps surface forms onto a stable key where an answer key needs it; a label
that fits nowhere becomes ``unknown`` and is routed to a human. Two things are
never done: inventing a cohort the table does not mention, and quietly merging
two labels into one.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel, Field

# Labels that explicitly mean "this review reports one combined value here",
# as opposed to a label we failed to place. Universal wording, not domain terms.
_COMBINED = {"all", "-", "total", "overall", "pooled", "combined", "whole cohort",
             "entire cohort", "both groups", "all participants"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def slug(text: str) -> str:
    """A stable match-key token for a cohort's display name."""
    s = re.sub(r"[^a-z0-9]+", "_", _norm(text)).strip("_")
    return s or "cohort"


def _mentions(label: str, variant: str) -> bool:
    """True when ``variant`` appears in ``label`` as whole words.

    Whole words, not a bare substring: "dm" must not match inside "admission",
    which is the class of accident that made the old keyword lists misfire.
    """
    return re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", label) is not None


# Words that appear in almost every arm label and therefore distinguish nothing.
# Dose numerals go too ("3 mg/kg"): a paper naming the same arm without the dose
# is still naming the same arm. Drug words, "placebo" and "plus" all stay — they
# are exactly what separates one arm from the next.
_GENERIC_LABEL_WORDS = {
    "a", "an", "and", "arm", "arms", "cohort", "cohorts", "dose", "group",
    "groups", "in", "n", "of", "participants", "patient", "patients", "people",
    "subject", "subjects", "the", "total", "with",
}
_DOSE_UNITS = {"mg", "kg", "g", "ml", "l", "mcg", "ug", "m2", "mg2"}

# How a comparison is written, in a slug or in prose. Direction is preserved:
# "A vs B" and "B vs A" are different claims and one is not the other's tidy-up.
_VS = re.compile(r"(?<![a-z0-9])(?:vs\.?|versus)(?![a-z0-9])")


class ComparisonTarget(BaseModel):
    """A claim about TWO arms — the shape a single cohort name cannot express."""

    left: str
    right: str
    raw: str = ""

    def sides(self) -> tuple[str, str]:
        return self.left, self.right

    def inverted(self) -> "ComparisonTarget":
        return ComparisonTarget(left=self.right, right=self.left, raw=self.raw)


def parse_comparison(text: str) -> ComparisonTarget | None:
    """Split ``A vs B`` into its two sides, or return None if it is not one.

    Accepts the canonical slug form (``a_vs_b``) and the written forms (``vs``,
    ``vs.``, ``versus``). More than one delimiter is refused rather than guessed
    at: ``a_vs_b_vs_c`` is not a pair, and picking one of its splits would
    invent a comparison the review never made.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    spaced = re.sub(r"[_\s]+", " ", raw).strip().lower()
    matches = list(_VS.finditer(spaced))
    if len(matches) != 1:
        return None
    left = spaced[: matches[0].start()].strip(" -")
    right = spaced[matches[0].end():].strip(" -")
    if not left or not right:
        return None
    return ComparisonTarget(left=left, right=right, raw=raw)


def distinguishing_tokens(label: str) -> set[str]:
    """The words in a label that actually identify which arm it is."""
    folded = unicodedata.normalize("NFKD", label or "").lower()
    tokens = re.findall(r"[a-z0-9]+", folded)
    return {
        token for token in tokens
        if token not in _GENERIC_LABEL_WORDS
        and token not in _DOSE_UNITS
        and not token.isdigit()
    }


def label_affinity(one: str, other: str) -> float:
    """How well two arm labels describe the same arm (0..1), both ways.

    A one-directional overlap is not enough: "nivolumab" is fully contained in
    "nivolumab plus ipilimumab", so measuring only how much of the shorter label
    is explained scores a monotherapy arm and a combination arm identically.
    Scoring both directions and combining them (F1) makes the label that also
    accounts for the OTHER side's words win.
    """
    left, right = distinguishing_tokens(one), distinguishing_tokens(other)
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if not shared:
        return 0.0
    precision, recall = shared / len(right), shared / len(left)
    return 2 * precision * recall / (precision + recall)


class CohortLabel(BaseModel):
    """One cohort this review actually reports."""

    key: str                                   # join-key token
    display: str                               # the review's OWN words
    raw_variants: list[str] = Field(default_factory=list)
    role: str = ""                             # index | comparator | combined | ""
    source: str = "discovered"                 # discovered | alias


class CohortResolution(BaseModel):
    """What one raw label resolved to, and how confident that is."""

    key: str = ""
    status: str = "unknown"        # resolved | alias | combined | ambiguous | unknown
    display: str = ""
    reason: str = ""

    @property
    def known(self) -> bool:
        return self.status in ("resolved", "alias", "combined")


class CohortRegistry(BaseModel):
    """The cohorts of one review, discovered from its own table."""

    labels: list[CohortLabel] = Field(default_factory=list)
    unassigned: list[str] = Field(default_factory=list)

    def by_key(self, key: str) -> CohortLabel | None:
        return next((c for c in self.labels if c.key == key), None)

    @property
    def arms(self) -> list[CohortLabel]:
        """Cohorts that are actual arms (a combined total is not an arm)."""
        return [c for c in self.labels if c.role != "combined"]

    def resolve(self, raw: str) -> CohortResolution:
        """Place a raw label. Never guesses: unplaceable → ``unknown``."""
        label = _norm(raw)
        if not label:
            # The table did not split THIS value by cohort. That is a statement
            # about the table, not a failure — distinct from an unplaceable label.
            return CohortResolution(key="all", status="combined", display="all",
                                    reason="the table reports no cohort for this cell")
        if label in _COMBINED:
            return CohortResolution(key="all", status="combined", display=raw,
                                    reason="an explicitly combined cohort")
        for cohort in self.labels:
            if label == _norm(cohort.display) or any(
                    label == _norm(v) for v in cohort.raw_variants):
                return CohortResolution(key=cohort.key, status="resolved",
                                        display=cohort.display)
        for cohort in self.labels:
            if any(_mentions(label, _norm(v)) for v in cohort.raw_variants):
                return CohortResolution(key=cohort.key, status="alias",
                                        display=cohort.display,
                                        reason=f"matched cohort {cohort.display!r}")
        return CohortResolution(
            key="", status="unknown", display=raw,
            reason=f"cohort {raw!r} is not one this review was found to report")


def load_aliases(path: Path | str | None) -> dict[str, list[str]]:
    """Surface form → stable key mappings (only where an answer key needs them).

    Keys beginning with ``_`` are documentation, not cohorts.
    """
    if path is None or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {k: [str(v) for v in vs] for k, vs in data.items()
            if not k.startswith("_") and isinstance(vs, list)}


def build_cohort_registry(
    raw_labels: list[str], *, aliases: dict[str, list[str]] | None = None,
) -> CohortRegistry:
    """Build the registry from the labels the review actually used.

    Discovery is deterministic and domain-neutral: the distinct labels ARE the
    cohorts. Aliases only re-key an already-discovered label (so a benchmark's
    answer key keeps joining); they never introduce a cohort of their own, which
    is what would quietly re-bind the system to one disease.
    """
    aliases = aliases or {}
    labels: list[CohortLabel] = []
    seen: dict[str, CohortLabel] = {}

    for raw in raw_labels:
        display = (raw or "").strip()
        norm = _norm(display)
        if not norm or norm in _COMBINED:
            continue

        key, source = slug(display), "discovered"
        for alias_key, variants in aliases.items():
            if any(norm == _norm(v) or _mentions(norm, _norm(v)) for v in variants):
                key, source = alias_key, "alias"
                break

        existing = seen.get(key)
        if existing is None:
            cohort = CohortLabel(key=key, display=display, raw_variants=[display],
                                 source=source)
            seen[key] = cohort
            labels.append(cohort)
        elif display not in existing.raw_variants:
            existing.raw_variants.append(display)

    return CohortRegistry(labels=labels)
