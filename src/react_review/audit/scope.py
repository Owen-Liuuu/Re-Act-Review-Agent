"""Do these two numbers count the same people?

Asked before the units and before the arithmetic, because it decides whether the
arithmetic means anything. 313 analysed and 314 allocated are not a 0.3%
transcription difference; they are two different quantities, and a relative band
that calls them equal is answering a question nobody asked.

Two rules keep this from becoming a machine for refusing everything:

*Only the axes a field needs.* A contract says which axes matter for which
field. An axis nobody required, left unstated, does not block a scope that is
already determined — but an axis both sides DO state, in conflict, is a
mismatch whether it was required or not. A stated contradiction is evidence.

*Refusing is measured.* ``scope_assessable_rate`` and ``scope_resolved_rate``
are reported next to the safety counters, because a system that rejects every
count would otherwise show three green safety numbers and no capability at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from react_review.normalize.population import (
    PopulationContract,
    PopulationScope,
    UNKNOWN_BASIS,
    UNSPECIFIED_SET,
)

OK = "ok"
MISMATCH = "scope_mismatch"
UNRESOLVED = "scope_unresolved"
NOT_REQUIRED = "not_required"

_AXIS_NAMES = {"population_basis": "population", "analysis_set": "analysis set"}


@dataclass
class ScopeOutcome:
    """Whether two values may be compared as counting the same people."""

    status: str = NOT_REQUIRED
    reason: str = ""
    checked_axes: list[str] = field(default_factory=list)
    unresolved_axes: list[str] = field(default_factory=list)

    @property
    def blocks_comparison(self) -> bool:
        return self.status in (MISMATCH, UNRESOLVED)

    @property
    def assessable(self) -> bool:
        """Whether the evidence said enough for the question to be answerable."""
        return self.status in (OK, MISMATCH)


def scope_verdict(review: PopulationScope, source: PopulationScope, *,
                  required_axes: list[str],
                  contract: PopulationContract | None = None) -> ScopeOutcome:
    """Compare two populations on the axes this field requires."""
    from react_review.normalize.population import _default_contract

    contract = contract or _default_contract()

    # A stated conflict counts on ANY axis: if the review says per-protocol and
    # the source says safety, the numbers are about different people whether or
    # not this field's contract happens to require that axis.
    for axis in ("population_basis", "analysis_set"):
        if review.axis_stated(axis) and source.axis_stated(axis):
            if not contract.compatible(axis, review.axis(axis), source.axis(axis)):
                return ScopeOutcome(
                    MISMATCH,
                    f"the review reports the {review.axis(axis)} {_AXIS_NAMES[axis]} "
                    f"and the source the {source.axis(axis)} one, so the two "
                    "numbers do not count the same people",
                    checked_axes=[axis])

    if not required_axes:
        return ScopeOutcome(NOT_REQUIRED)

    unresolved = [axis for axis in required_axes
                  if not (review.axis_stated(axis) and source.axis_stated(axis))]
    if unresolved:
        return ScopeOutcome(
            UNRESOLVED,
            "the population this value counts is not stated on "
            + _which_side(review, source, unresolved[0])
            + f", so {', '.join(_AXIS_NAMES[a] for a in unresolved)} could not be "
              "checked",
            checked_axes=list(required_axes), unresolved_axes=unresolved)

    return ScopeOutcome(OK, checked_axes=list(required_axes))


def _which_side(review: PopulationScope, source: PopulationScope, axis: str) -> str:
    review_ok, source_ok = review.axis_stated(axis), source.axis_stated(axis)
    if review_ok and not source_ok:
        return "the source side"
    if source_ok and not review_ok:
        return "the review side"
    return "either side"


def scope_rates(outcomes: list[ScopeOutcome]) -> dict[str, object]:
    """How much scope checking actually achieved — not just how safe it was.

    Refusing everything makes the safety counters perfect. These rates are what
    make that visible: they say how often the evidence could answer the
    question, and how often it answered it affirmatively.
    """
    required = [o for o in outcomes if o.status != NOT_REQUIRED]
    if not required:
        return {"scope_required": 0, "scope_assessable_rate": None,
                "scope_resolved_rate": None, "scope_unresolved": 0}
    assessable = sum(o.assessable for o in required)
    resolved = sum(o.status == OK for o in required)
    return {
        "scope_required": len(required),
        "scope_assessable_rate": assessable / len(required),
        "scope_resolved_rate": resolved / len(required),
        "scope_unresolved": sum(o.status == UNRESOLVED for o in required),
        "scope_mismatch": sum(o.status == MISMATCH for o in required),
    }
