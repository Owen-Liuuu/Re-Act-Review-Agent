"""Grading a route change row by row — in code, so the verdict is reproducible.

Gate v1 had no classifier. Its transitions were computed by a throwaway script
written next to the result, and that script contradicted the gate's own text:
v1 defines `wrong_released` as a wrong value released WITHOUT review, and the
script called a review-flagged row `wrong_released` anyway. It reported FAIL for
a transition the gate does not define.

Two things follow, and both are here rather than in prose:

*The state space needs a fourth term.* A row can be wrong AND escalated. That is
not a silent release — the whole safety apparatus exists to make it visible —
and it is not correct either. Collapsing it into either neighbour loses the
distinction the project spent Phase 6 building.

*A gate of prohibitions can be passed by refusing everything.* v1's conditions
are all "zero of these bad things", and a system that answers nothing satisfies
every one. A floor is not a nicety; without it the gate cannot distinguish a
route that works from a route that has stopped participating.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The four states a graded row can be in. `wrong_but_flagged` is the one v1
#: could not express, and the one this recording turned on.
CORRECT = "correct"
REFUSED = "refused"
WRONG_BUT_FLAGGED = "wrong_but_flagged"
WRONG_RELEASED = "wrong_released"

#: Labels that mean the audit declined to answer rather than answered wrongly.
_REFUSALS = frozenset({"not_comparable", "review_required"})


def row_state(*, predicted_label: str, expected_label: str,
              review_required: bool) -> str:
    """What a graded row is, in the terms a gate can reason about.

    `review_required` is load-bearing. "Released" is literal everywhere else in
    this project — a row the audit escalated is not a released error — and a
    classifier that ignores it grades a different system than the one that ran.
    """
    if (predicted_label or "") in _REFUSALS:
        return REFUSED
    if predicted_label == expected_label:
        return CORRECT
    return WRONG_BUT_FLAGGED if review_required else WRONG_RELEASED


@dataclass(frozen=True)
class TransitionOutcome:
    """One row's move from the baseline, and whether the gate permits it."""

    audit_id: str
    field_type: str
    before: str
    after: str
    permitted: bool
    reason: str = ""

    @property
    def move(self) -> str:
        return f"{self.before} -> {self.after}"


@dataclass(frozen=True)
class GateResult:
    """A verdict, or a refusal to issue one.

    `NOT_EVALUABLE` is a real outcome and not a soft failure. A gate whose state
    space cannot express what happened has not judged the run, and recording
    that as a FAIL would attribute to the system a fault that belongs to the
    gate.
    """

    verdict: str                     # pass | fail | not_evaluable
    transitions: tuple = ()
    violations: tuple = ()
    unmet_hard_conditions: tuple = ()
    capability: dict | None = None
    #: What each structural condition actually saw, so a PASS can be inspected
    #: rather than believed.
    hard_condition_evidence: dict | None = None
    #: Whether a capability floor was applied at all. A run that meets every
    #: prohibition and was never measured against a floor has not been shown to
    #: work — it has been shown not to have broken a rule, and the two read
    #: identically in a one-word verdict unless this is carried alongside.
    capability_judged: bool = False
    reason: str = ""


#: Where each declared hard condition is READ FROM in a scored artifact, and
#: what it must equal. A condition with no reader here is refused rather than
#: skipped: a gate that declares a rule nothing enforces is worse than one that
#: declares nothing, and this repository has already found that exact failure in
#: an aggregation policy and written a test for it there.
_HARD_CONDITION_READERS = {
    "silent_releases": ("metrics", "safety", "silent_release_count"),
    # The gold-graded counter, not the deprecated numeric one. `target.
    # wrong_target_accepted_count` reads 0 on a run whose comparison identity
    # was wrong and released, because it never looked at the gold.
    "wrong_target_accepted_count": ("metrics", "target", "gold",
                                    "identity_wrong_released"),
    "scope_wrong_released_count": ("metrics", "scope",
                                   "scope_wrong_released_count"),
}


def _read_path(body: dict, path: tuple):
    node = body
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def check_hard_conditions(gate: dict, artifact: dict) -> tuple[list[str], dict]:
    """Every numeric prohibition the gate declares, read from the run.

    Conditions this cannot read are reported as unenforceable rather than
    passed over. `batches_match_expected_plan` and
    `every_batched_row_resolves_its_execution_id` are checked by the preflight
    and the package tests respectively, and are named here so that the gate's
    own declaration is answered rather than ignored.
    """
    unmet: list[str] = []
    evidence: dict = {}
    # `no_forbidden_transition` is decided by the transition table, which grade()
    # computes from these same rows. The other two used to be listed here too,
    # with a note saying a preflight and a test checked them — which is a claim
    # about OTHER runs. A gate has to read THIS artifact, or it is trusting that
    # somebody looked.
    external = {"no_forbidden_transition": "decided by the transition table"}
    for name, expected in (gate.get("hard_conditions") or {}).items():
        if name in external:
            continue
        if name in _STRUCTURAL_CONDITIONS:
            failure, seen = _STRUCTURAL_CONDITIONS[name](artifact, gate)
            evidence[name] = seen
            if failure:
                unmet.append(failure)
            continue
        reader = _HARD_CONDITION_READERS.get(name)
        if reader is None:
            unmet.append(f"hard condition {name!r} has no reader, so the gate "
                         "declares a rule nothing enforces")
            continue
        actual = _read_path(artifact, reader)
        if actual is None:
            unmet.append(f"hard condition {name!r} is declared and the run "
                         f"reports no {'.'.join(reader)}")
        elif actual != expected:
            unmet.append(f"hard condition {name!r}: the gate requires "
                         f"{expected} and the run reports {actual} "
                         f"(from {'.'.join(reader)})")
        else:
            evidence[name] = {"value": actual, "read_from": ".".join(reader)}
    return unmet, evidence


def _batches_of(artifact: dict) -> list[dict]:
    """The batches THIS run made, as the artifact records them."""
    coverage = ((artifact.get("run") or {}).get("excerpt_coverage") or {})
    return list(coverage.get("batches") or ())


def _check_execution_ids(artifact: dict, gate: dict):
    """Every batched row names a reading, and that reading exists.

    Computed from the rows and the batches of this run. A reference that
    resolves nowhere would be worse than no reference — it would look like
    evidence — and a gate that took someone's word for it could not tell.
    """
    batches = _batches_of(artifact)
    known = {b.get("execution_id") for b in batches if b.get("execution_id")}
    claimed = {c for b in batches for c in (b.get("claim_ids") or ())}
    rows = [r for r in (artifact.get("rows") or ())
            if r.get("batch_execution_id")]
    seen = {"batches": len(batches), "batched_rows": len(rows),
            "distinct_execution_ids": len(known)}
    if not batches:
        return ("hard condition 'every_batched_row_resolves_its_execution_id': "
                "the artifact records no batches, so nothing could be "
                "resolved"), seen
    dangling = sorted({r["batch_execution_id"] for r in rows} - known)
    if dangling:
        return (f"hard condition 'every_batched_row_resolves_its_execution_id': "
                f"{len(dangling)} row(s) name a reading the artifact does not "
                f"contain ({dangling[:2]})"), seen
    unnamed = sorted({r.get("audit_id", "") for r in rows} - claimed)
    if unnamed:
        return (f"hard condition 'every_batched_row_resolves_its_execution_id': "
                f"the reading does not name back the claims {unnamed[:3]}, so "
                "the reference resolves in one direction only"), seen
    return "", seen


def _check_batches_match_plan(artifact: dict, gate: dict):
    """The batches this run made are the ones the plan pre-registered.

    Read from the plan the gate names, and compared against this artifact — not
    against a preflight that ran at some other time against some other cache.
    """
    import json
    from pathlib import Path

    from react_review.contracts import repo_root

    applies = gate.get("applies_to") or {}
    name = applies.get("expected_plan") or ""
    benchmark = applies.get("benchmark") or ""
    path = (repo_root() / "eval/benchmarks" / benchmark / name) if name else None
    batches = _batches_of(artifact)
    seen = {"batches": len(batches), "plan": name}
    if path is None or not path.is_file():
        return (f"hard condition 'batches_match_expected_plan': the gate names "
                f"plan {name!r} and it is not in this checkout, so the "
                "condition cannot be read"), seen

    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    # The expected plan names them `claim_ids` and an excerpt key names them
    # `audit_ids`. They are the same claims; reading only one spelling made
    # every batch look unplanned.
    planned = {tuple(sorted(b.get("claim_ids") or b.get("audit_ids") or ()))
               for b in plan.get("expected_batches") or ()}
    made = {tuple(sorted(b.get("claim_ids") or ())) for b in batches}
    seen["plan_batches"] = len(planned)
    if planned != made:
        missing = sorted(planned - made)
        extra = sorted(made - planned)
        return (f"hard condition 'batches_match_expected_plan': "
                f"{len(missing)} planned batch(es) were not made {missing[:2]} "
                f"and {len(extra)} unplanned were {extra[:2]}"), seen
    return "", seen


_STRUCTURAL_CONDITIONS = {
    "every_batched_row_resolves_its_execution_id": _check_execution_ids,
    "batches_match_expected_plan": _check_batches_match_plan,
}


def grade(gate: dict, rows, *, baseline_key: str = "baseline_state",
          artifact: dict | None = None) -> GateResult:
    """Apply a gate to a scored run.

    `rows` are dicts carrying at least audit_id, field_type, predicted_label,
    expected_label and review_required — the fields the eval artifact already
    has, so nothing has to be re-derived to grade a run.
    """
    baseline = {r["audit_id"]: r for r in gate.get("baseline_rows") or ()}
    permitted = set(gate.get("transitions", {}).get("allowed") or ())
    forbidden = set(gate.get("transitions", {}).get("forbidden") or ())
    known = permitted | forbidden

    transitions: list[TransitionOutcome] = []
    unexpressible: list[str] = []
    for row in rows:
        audit_id = row["audit_id"]
        if audit_id not in baseline:
            return GateResult(
                "not_evaluable",
                reason=(f"the run graded {audit_id}, which the gate's baseline "
                        "does not contain. A gate can only judge the rows it "
                        "pre-registered"))
        before = baseline[audit_id][baseline_key]
        after = row_state(predicted_label=row.get("predicted_label", ""),
                          expected_label=row.get("expected_label", ""),
                          review_required=bool(row.get("review_required")))
        move = f"{before} -> {after}"
        if move not in known:
            unexpressible.append(f"{audit_id}: {move}")
        transitions.append(TransitionOutcome(
            audit_id, row.get("field_type", ""), before, after,
            permitted=move not in forbidden,
            reason=(forbidden.get(move, "") if isinstance(forbidden, dict) else "")))

    if unexpressible:
        return GateResult(
            "not_evaluable", transitions=tuple(transitions),
            reason=("the gate's state space does not express what happened: "
                    + "; ".join(unexpressible) + ". Judging it anyway would "
                    "report a verdict the gate never defined"))

    violations = tuple(t for t in transitions if not t.permitted)
    after_states = [t.after for t in transitions]
    capability = {
        "rows": len(transitions),
        "correct": after_states.count(CORRECT),
        "refused": after_states.count(REFUSED),
        "wrong_but_flagged": after_states.count(WRONG_BUT_FLAGGED),
        "wrong_released": after_states.count(WRONG_RELEASED),
        "baseline_correct": sum(1 for r in baseline.values()
                                if r[baseline_key] == CORRECT),
    }

    unmet: list[str] = []
    evidence: dict = {}
    if artifact is not None:
        found, evidence = check_hard_conditions(gate, artifact)
        unmet.extend(found)
    elif gate.get("hard_conditions"):
        unmet.append(
            "the gate declares hard conditions and was applied to rows alone, "
            "so none of them were read. Pass the scored artifact")
    floor = gate.get("capability_floor") or {}
    judged = any(floor.get(name) is not None
                 for name in ("min_correct_rows",
                              "min_fraction_of_baseline_correct"))
    minimum = floor.get("min_correct_rows")
    if minimum is not None and capability["correct"] < minimum:
        unmet.append(
            f"capability floor: {capability['correct']} correct rows, and the "
            f"gate requires at least {minimum}. A gate of prohibitions alone is "
            "passed by refusing everything")
    retained = floor.get("min_fraction_of_baseline_correct")
    if retained is not None and capability["baseline_correct"]:
        kept = capability["correct"] / capability["baseline_correct"]
        if kept < retained:
            unmet.append(
                f"capability floor: kept {kept:.0%} of the baseline's correct "
                f"rows, and the gate requires {retained:.0%}")

    if violations or unmet:
        verdict = "fail"
    elif judged:
        verdict = "pass"
    else:
        # Not the same sentence as "pass". Every hard condition in this family
        # is a prohibition, and refusing every row satisfies all of them.
        verdict = "pass_prohibitions_only"
    return GateResult(verdict, tuple(transitions), violations, tuple(unmet),
                      capability, hard_condition_evidence=(evidence or None),
                      capability_judged=judged)
