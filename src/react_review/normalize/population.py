"""Which population a number counts, read from the paper's own words.

Phase 7 closed the "wrong arm" failure and left a quieter one open. Asked for
the size of a trial arm the extractor returned 313 from a table of one analysis
population, where the review reports the 314 who were allocated to that arm. The
number was right about something, the arm was right, and a 1% band called it a
match: a scope error wearing a correct-looking number, with no guard anywhere in
the pipeline.

So a value carries WHICH POPULATION it counts, on two axes that are orthogonal
in trials and must not be collapsed:

``population_basis``   allocated | treated | analysed | unknown
``analysis_set``       itt | mitt | per_protocol | evaluable | safety | unspecified

plus ``randomisation_stated``, kept separate on purpose. "were assigned to the
ipilimumab group" establishes that people were allocated; it does not say the
allocation was random, and reading one as the other invents a study design the
paper never claimed.

Classification is deterministic and reads only the evidence already anchored to
the document — no prompt changes, so every recorded run stays replayable and
this can be measured on recordings that already exist. When the words are not
there the answer is ``unknown``: an unstated basis is a fact about the evidence,
never a licence to assume the usual one.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from react_review.contracts import ContractError, repo_root
from react_review.normalize.anchors import flatten

UNKNOWN_BASIS = "unknown"
UNSPECIFIED_SET = "unspecified"

BASIS_VALUES = ("allocated", "treated", "analysed", UNKNOWN_BASIS)
ANALYSIS_SETS = ("itt", "mitt", "per_protocol", "evaluable", "safety", UNSPECIFIED_SET)


class PopulationScope(BaseModel):
    """What population a value counts, and the words that decided it."""

    basis: str = UNKNOWN_BASIS
    analysis_set: str = UNSPECIFIED_SET
    randomisation_stated: bool = False
    # The matched wording, so a reader can disagree with the classification
    # instead of having to trust it.
    basis_phrase: str = ""
    analysis_set_phrase: str = ""
    source: str = ""            # quote | column_header | contract

    @property
    def stated(self) -> bool:
        return self.basis != UNKNOWN_BASIS

    def axis(self, name: str) -> str:
        return {"population_basis": self.basis,
                "analysis_set": self.analysis_set}[name]

    def axis_stated(self, name: str) -> bool:
        return self.axis(name) not in (UNKNOWN_BASIS, UNSPECIFIED_SET)

    @classmethod
    def parse(cls, text: str, *, source: str = "contract") -> "PopulationScope":
        """Read the declared form ``basis`` or ``basis/analysis_set``.

        Used where a population is STATED rather than read out of prose — a
        review's own column heading, or an evaluation contract recording what
        that heading says. An empty string is unknown, which is the honest
        answer for the many reviews that never say.
        """
        raw = (text or "").strip().lower()
        if not raw:
            return cls(source=source)
        basis, _, analysis_set = raw.partition("/")
        if basis and basis not in BASIS_VALUES:
            raise ContractError(
                f"unknown population_basis {basis!r} "
                f"(known: {', '.join(BASIS_VALUES)})")
        if analysis_set and analysis_set not in ANALYSIS_SETS:
            raise ContractError(
                f"unknown analysis_set {analysis_set!r} "
                f"(known: {', '.join(ANALYSIS_SETS)})")
        return cls(basis=basis or UNKNOWN_BASIS,
                   analysis_set=analysis_set or UNSPECIFIED_SET, source=source)

    def describe(self) -> str:
        parts = [self.basis]
        if self.analysis_set != UNSPECIFIED_SET:
            parts.append(self.analysis_set)
        return "/".join(parts)


class PopulationContract(BaseModel):
    """The phrase table, as loaded from a hash-pinned contract file."""

    contract_id: str = "population_roles_v1"
    basis: dict[str, list[str]] = {}
    analysis_set: dict[str, list[str]] = {}
    randomisation: list[str] = []
    compatible_basis_pairs: list[list[str]] = []
    compatible_analysis_set_pairs: list[list[str]] = []

    def compatible(self, axis: str, one: str, other: str) -> bool:
        if one == other:
            return True
        pairs = (self.compatible_basis_pairs if axis == "population_basis"
                 else self.compatible_analysis_set_pairs)
        return any({one, other} == set(pair) for pair in pairs)


def load_population_contract(path: Path | str | None = None) -> PopulationContract:
    """Load the phrase table. Unknown axis values are refused, not ignored."""
    path = Path(path) if path else (repo_root() / "configs" / "population_roles.json")
    if not path.is_file():
        raise ContractError(f"population contract does not exist: {path}")
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    basis = {k: v for k, v in (body.get("population_basis") or {}).items()}
    sets = {k: v for k, v in (body.get("analysis_set") or {}).items()}
    for key in basis:
        if key not in BASIS_VALUES:
            raise ContractError(f"unknown population_basis {key!r}")
    for key in sets:
        if key not in ANALYSIS_SETS:
            raise ContractError(f"unknown analysis_set {key!r}")
    return PopulationContract(
        contract_id=str(body.get("contract_id") or path.stem),
        basis=basis, analysis_set=sets,
        randomisation=list(body.get("randomisation_stated") or []),
        compatible_basis_pairs=[list(p) for p in
                                (body.get("compatible_basis_pairs") or [])],
        compatible_analysis_set_pairs=[list(p) for p in
                                       (body.get("compatible_analysis_set_pairs") or [])])


@lru_cache(maxsize=8)
def _default_contract() -> PopulationContract:
    return load_population_contract()


def classify_population(text: str, *, contract: PopulationContract | None = None,
                        source: str = "quote") -> PopulationScope:
    """Read the population this passage is talking about.

    Longest phrase first, so "were randomly assigned" is not decided by the
    "assigned to" inside it, and the matched wording is carried out with the
    answer.
    """
    contract = contract or _default_contract()
    flat = flatten(text or "")
    if not flat:
        return PopulationScope(source=source)

    basis, basis_phrase = _best_match(flat, contract.basis)
    analysis_set, set_phrase = _best_match(flat, contract.analysis_set)
    return PopulationScope(
        basis=basis or UNKNOWN_BASIS,
        analysis_set=analysis_set or UNSPECIFIED_SET,
        randomisation_stated=any(_contains(flat, word)
                                 for word in contract.randomisation),
        basis_phrase=basis_phrase, analysis_set_phrase=set_phrase, source=source)


def _best_match(flat: str, table: dict[str, list[str]]) -> tuple[str, str]:
    best_value = best_phrase = ""
    for value, phrases in table.items():
        for phrase in phrases:
            if len(phrase) > len(best_phrase) and _contains(flat, phrase):
                best_value, best_phrase = value, phrase
    return best_value, best_phrase


def _contains(flat: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])",
                     flat) is not None
