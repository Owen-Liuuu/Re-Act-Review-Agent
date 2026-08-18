"""Evaluating a PRE-REGISTERED acceptance gate.

A threshold chosen after seeing the result is not a threshold. So the gate lives
in its own hash-pinned file, written before the data exists, and this module
only applies it. Three things follow from that, and they are the reason this is
code rather than a paragraph in a report:

*The unit of analysis is the study, and the estimand is domain-weighted.*
Fifteen rows from one paper are not fifteen independent observations — one bad
extraction pass moves them together — so studies are resampled, not rows. And
because nine studies from one field beside one study from another would otherwise
mean "how well does this work on the first field", each DOMAIN contributes
equally to the headline figure, with its own studies resampled inside it. A
benchmark with one study supports no interval at all; that is why "60%" and
"80%" from the melanoma checkpoint were never rankable against each other.

*A held-out domain is reported alone.* Averaging it into the domains the system
was built on is how a held-out result stops being held out.

*Not estimable is a third outcome.* A gate that can only pass or fail will be
made to pass by whoever needs it to. Where the evidence cannot support the
question, this says so, and says which minimum was missed.

*Safety is separate from capability.* The hard gates are counts that must be
exactly zero — a discrepancy released silently, a value released about the wrong
arm or the wrong population. They are not averaged, not traded against accuracy,
and not reported as a percentage.

*An absence is not a zero.* The first version of this checker filled a missing
safety number with 0 and reported the gate as passing — the same defect the
audit itself exists to catch: nothing recorded, and read as nothing wrong. There
*Evaluating and releasing are different questions.* A gate whose thresholds
nobody with clinical responsibility has signed off can be met and still not
authorise anything. The two verdicts are reported separately so that "the
numbers cleared a bar we invented" can never be read as "this is fit to use".

are now four states per gate: never reported, reported but with nothing to grade
it against, graded and held, graded and failed. Only the third is a pass.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pydantic import BaseModel, Field

from react_review.contracts import ContractError, read_json_object, sha256_file

NOT_ESTIMABLE = "not_estimable"
PASS = "pass"
FAIL = "fail"


OK = "ok"
MISSING = "evidence_missing"
NO_DENOMINATOR = "nothing_graded"
FAILED = "failed"


class HardGate(BaseModel):
    """A count that must be exactly zero, or a rate that must be exactly one."""

    metric: str
    must_equal: float
    why: str = ""
    # The metric whose count must exceed zero before "no errors" means anything.
    # Zero wrong out of zero graded is not a safety record.
    evidence_denominator: str = ""


class Observation(BaseModel):
    """One measured safety number, and what it was measured over."""

    value: float | None = None
    denominator: int | None = None      # None = not applicable to this metric
    source: str = ""                    # which artifact or attestation said so


class CapabilityGate(BaseModel):
    """A proportion whose CLUSTER-BOOTSTRAP lower bound must clear a bar."""

    metric: str
    lower_bound_at_least: float
    why: str = ""


class SampleRequirement(BaseModel):
    """What has to exist before the question can be asked at all."""

    min_domains: int = 1
    min_studies_per_domain: int = 1
    min_rows_per_domain: int = 1
    min_true_discrepancies: int = 1
    min_rows_per_route: dict[str, int] = Field(default_factory=dict)
    held_out_domain_required: bool = False
    # Twenty-five discrepancies all in one paper is not twenty-five chances to
    # fail. Both the count per domain and how many studies carry one matter.
    min_true_discrepancies_per_domain: int = 0
    min_studies_with_discrepancies: int = 0


class AcceptanceGate(BaseModel):
    """A pre-registered definition of passing."""

    gate_id: str
    version: int
    path: Path
    sha256: str
    status: str = "provisional"     # provisional | signed_off
    signed_off_by: str = ""
    confidence_level: float = 0.95
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20260810
    # Above this share of undefined resamples the interval stops describing the
    # question and starts describing the draws that happened to work.
    max_undefined_draw_rate: float = 0.05
    sample: SampleRequirement = Field(default_factory=SampleRequirement)
    hard_gates: list[HardGate] = Field(default_factory=list)
    capability_gates: list[CapabilityGate] = Field(default_factory=list)
    reported_only: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def signed_off(self) -> bool:
        return self.status == "signed_off"


class GateOutcome(BaseModel):
    """The verdict, and every reason behind it."""

    gate_id: str
    gate_sha256: str
    status: str = NOT_ESTIMABLE
    hard: list[dict[str, Any]] = Field(default_factory=list)
    capability: list[dict[str, Any]] = Field(default_factory=list)
    sample: dict[str, Any] = Field(default_factory=dict)
    reported: dict[str, Any] = Field(default_factory=dict)
    blocking: list[str] = Field(default_factory=list)
    # Whether this evidence may authorise anything, which is a stricter and
    # different question from whether the numbers cleared their bars.
    release_eligible: bool = False
    release_blockers: list[str] = Field(default_factory=list)
    # What the verdict rests on: every input, by hash.
    provenance: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        head = {PASS: "PASS", FAIL: "FAIL",
                NOT_ESTIMABLE: "NOT ESTIMABLE"}[self.status]
        release = ("release: ELIGIBLE" if self.release_eligible
                   else "release: NOT ELIGIBLE")
        lines = [f"{head} — gate {self.gate_id} ({self.gate_sha256[:12]}…) · "
                 f"{release}"]
        lines += [f"  release blocked: {reason}"
                  for reason in self.release_blockers]
        lines += [f"  blocked: {reason}" for reason in self.blocking]
        marks = {OK: "ok", FAILED: "XX", MISSING: "--", NO_DENOMINATOR: "??"}
        for check in self.hard:
            suffix = "" if check["state"] == OK else f"  <- {check['state']}"
            lines.append(
                f"  [{marks[check['state']]}] {check['metric']} = "
                f"{check['observed']} (must be {check['required']}){suffix}")
        for check in self.capability:
            mark = {True: "ok", False: "XX", None: "??"}[check["passed"]]
            line = (f"  [{mark}] {check['metric']} = {_fmt(check['point'])} "
                    f"[{_fmt(check['lower'])}, {_fmt(check['upper'])}] "
                    f"≥ {check['required']}")
            if check.get("held_out") is not None:
                line += f"  held-out: {_fmt(check['held_out'])}"
            lines.append(line)
            lines.append(f"        per domain: " + ", ".join(
                f"{d}={_fmt(v)}" for d, v in sorted(
                    (check.get("per_domain") or {}).items())))
        return "\n".join(lines)


def load_gate(path: Path | str) -> AcceptanceGate:
    """Load a pre-registered gate file, refusing an unusable one."""
    path = Path(path)
    body = read_json_object(path, kind="acceptance gate")
    if not body.get("gate_id"):
        raise ContractError("an acceptance gate must have a gate_id")
    if body.get("status") not in ("provisional", "signed_off"):
        raise ContractError("gate status must be provisional or signed_off")
    return AcceptanceGate(
        path=path, sha256=sha256_file(path),
        gate_id=str(body["gate_id"]), version=int(body.get("version") or 1),
        status=str(body["status"]), signed_off_by=str(body.get("signed_off_by") or ""),
        confidence_level=float(body.get("confidence_level") or 0.95),
        bootstrap_resamples=int(body.get("bootstrap_resamples") or 2000),
        bootstrap_seed=int(body.get("bootstrap_seed") or 20260810),
        max_undefined_draw_rate=float(
            body.get("max_undefined_draw_rate") or 0.05),
        sample=SampleRequirement(**(body.get("sample") or {})),
        hard_gates=[HardGate(**g) for g in (body.get("hard_gates") or [])],
        capability_gates=[CapabilityGate(**g)
                          for g in (body.get("capability_gates") or [])],
        reported_only=[str(m) for m in (body.get("reported_only") or [])],
        notes=[str(n) for n in (body.get("notes") or [])])


def domain_weighted(strata: Sequence[Sequence[Sequence[Any]]],
                    statistic: Callable[[list[Any]], float | None]) -> float | None:
    """Each domain counts once, whatever its row count.

    Pooling rows would make the headline number a report on whichever domain
    happens to be largest. Domains where the statistic is undefined — no true
    discrepancies to recall, say — drop out rather than counting as zero.
    """
    values = [statistic([row for cluster in domain for row in cluster])
              for domain in strata]
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def stratified_bootstrap(
    strata: Sequence[Sequence[Sequence[Any]]],
    statistic: Callable[[list[Any]], float | None],
    *, resamples: int = 2000, seed: int = 20260810, confidence: float = 0.95,
    max_undefined_rate: float = 0.05,
) -> dict[str, Any]:
    """Resample studies WITHIN each domain; weight domains equally.

    Returns the point estimate, the interval, and the share of resamples in
    which the statistic was undefined. That share is reported rather than
    quietly dropped: an interval computed from the half of the draws that
    happened to work is not an interval.
    """
    point = domain_weighted(strata, statistic)
    usable = [domain for domain in strata if domain]
    if len(usable) < 1 or all(len(domain) < 2 for domain in usable):
        return {"point": point, "lower": None, "upper": None,
                "undefined_rate": None,
                "note": ("every domain has fewer than two studies, so there is "
                         "nothing to resample: uncertainty is not estimable")}
    rng = random.Random(seed)
    draws: list[float] = []
    undefined = 0
    for _ in range(resamples):
        picked = [[domain[rng.randrange(len(domain))] for _ in domain]
                  for domain in usable]
        value = domain_weighted(picked, statistic)
        if value is None:
            undefined += 1
        else:
            draws.append(value)
    undefined_rate = undefined / resamples
    if undefined_rate > max_undefined_rate:
        return {"point": point, "lower": None, "upper": None,
                "undefined_rate": undefined_rate,
                "note": (f"the statistic was undefined in {undefined_rate:.0%} of "
                         f"resamples (limit {max_undefined_rate:.0%}), so the "
                         "interval would describe only the draws that happened "
                         "to work")}
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    return {"point": point,
            "lower": draws[max(0, int(tail * len(draws)) - 1)],
            "upper": draws[min(len(draws) - 1, int((1.0 - tail) * len(draws)))],
            "undefined_rate": undefined_rate, "note": ""}


def cluster_bootstrap(
    clusters: Sequence[Sequence[Any]], statistic: Callable[[list[Any]], float | None],
    *, resamples: int = 2000, seed: int = 20260810, confidence: float = 0.95,
) -> tuple[float | None, float | None, float | None, str]:
    """Percentile interval, resampling STUDIES rather than rows.

    Returns ``(point, lower, upper, note)``. With fewer than two clusters every
    resample is the same sample, so the interval would be a restatement of the
    point estimate wearing the clothes of an uncertainty measure: that case
    returns no interval and says why.
    """
    rows = [row for cluster in clusters for row in cluster]
    point = statistic(rows)
    if len(clusters) < 2:
        return point, None, None, (
            f"{len(clusters)} cluster(s): an interval over studies needs at "
            "least two, so uncertainty here is not estimable")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        picked = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        value = statistic([row for cluster in picked for row in cluster])
        if value is not None:
            draws.append(value)
    if not draws:
        return point, None, None, "the statistic was undefined in every resample"
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lower = draws[max(0, int(tail * len(draws)) - 1)]
    upper = draws[min(len(draws) - 1, int((1.0 - tail) * len(draws)))]
    return point, lower, upper, ""


# --- the statistics the gate can name ------------------------------------

FLAG_LABELS = {"mismatch", "unit_mismatch"}


def _recall(rows: list[Any]) -> float | None:
    positives = [r for r in rows if _expected(r) in FLAG_LABELS]
    if not positives:
        return None
    return sum(_predicted(r) in FLAG_LABELS for r in positives) / len(positives)


def _precision(rows: list[Any]) -> float | None:
    flagged = [r for r in rows if _predicted(r) in FLAG_LABELS]
    if not flagged:
        return None
    return sum(_expected(r) in FLAG_LABELS for r in flagged) / len(flagged)


def _label_accuracy(rows: list[Any]) -> float | None:
    return (sum(_predicted(r) == _expected(r) for r in rows) / len(rows)
            if rows else None)


def _retrieval_coverage(rows: list[Any]) -> float | None:
    """A value was located. It says nothing about whether it was the right one."""
    return (sum(bool(_get(r, "found")) for r in rows) / len(rows)) if rows else None


def _auditable_coverage(rows: list[Any]) -> float | None:
    """A value was located AND the audit could actually judge it.

    Retrieval coverage counts a number found in the wrong arm or the wrong
    population; this does not. The gap between the two is the honest measure of
    how much of a review was really audited.
    """
    if not rows:
        return None
    return sum(
        1 for r in rows
        if _get(r, "found")
        and str(_get(r, "target_identity_correct") or "") != "false"
        and str(_get(r, "target_scope_correct") or "") != "false"
        and str(_get(r, "scope_check") or "") not in ("scope_unresolved",
                                                      "scope_mismatch")
        and str(_get(r, "evidence_check") or "ok") == "ok") / len(rows)


def _scope_assessable_rate(rows: list[Any]) -> float | None:
    """Of the rows whose population had to be checked, how many could be."""
    required = [r for r in rows
                if str(_get(r, "scope_check") or "") not in ("", "not_required")]
    if not required:
        return None
    return sum(str(_get(r, "scope_check")) in ("ok", "scope_mismatch")
               for r in required) / len(required)


def _review_burden(rows: list[Any]) -> float | None:
    """Share of rows a human is asked to look at. Reported, never gated."""
    if not rows:
        return None
    return sum(bool(_get(r, "review_required")) for r in rows) / len(rows)


STATISTICS: dict[str, Callable[[list[Any]], float | None]] = {
    "discrepancy_recall": _recall,
    "discrepancy_precision": _precision,
    "label_accuracy": _label_accuracy,
    "retrieval_coverage": _retrieval_coverage,
    "auditable_coverage": _auditable_coverage,
    "scope_assessable_rate": _scope_assessable_rate,
    "review_burden": _review_burden,
}


def evaluate_gate(
    gate: AcceptanceGate, rows: Iterable[Any], *,
    evidence: dict[str, Observation], domains: dict[str, str] | None = None,
    held_out_domain: str = "",
) -> GateOutcome:
    """Apply a pre-registered gate to one set of scored rows.

    ``domains`` maps study_id -> domain; without it every study is treated as
    one domain, which is exactly the situation the sample requirements exist to
    refuse.
    """
    rows = list(rows)
    outcome = GateOutcome(gate_id=gate.gate_id, gate_sha256=gate.sha256)

    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        grouped[str(_get(row, "study_id") or "")].append(row)
    # Ordered by study identity, not by the order reports were named on a
    # command line. The resampling walks cluster INDICES under a fixed seed, so
    # an unstable order makes the interval a function of the argument list: the
    # same evidence gave a precision bound of 0.444 one way round and 0.500 the
    # other, either side of the bar it was being judged against.
    by_study: dict[str, list[Any]] = {k: grouped[k] for k in sorted(grouped)}
    domains = domains or {}
    by_domain = Counter(domains.get(study, "unspecified") for study in by_study)

    outcome.sample = {
        "rows": len(rows),
        "studies": len(by_study),
        "domains": len(by_domain),
        "studies_per_domain": dict(by_domain),
        "true_discrepancies": sum(1 for r in rows if _expected(r) in FLAG_LABELS),
        "held_out_domain": held_out_domain,
        "rows_per_route": dict(Counter(
            str(_get(r, "expected_match_mode") or "unspecified") for r in rows)),
    }

    need = gate.sample
    if outcome.sample["domains"] < need.min_domains:
        outcome.blocking.append(
            f"{outcome.sample['domains']} domain(s), the gate requires "
            f"{need.min_domains}")
    for domain, count in by_domain.items():
        if count < need.min_studies_per_domain:
            outcome.blocking.append(
                f"domain {domain!r} has {count} study/studies, the gate requires "
                f"{need.min_studies_per_domain}")
    rows_per_domain: Counter = Counter()
    for study, study_rows in by_study.items():
        rows_per_domain[domains.get(study, "unspecified")] += len(study_rows)
    outcome.sample["rows_per_domain"] = dict(rows_per_domain)
    for domain, count in sorted(rows_per_domain.items()):
        if count < need.min_rows_per_domain:
            outcome.blocking.append(
                f"domain {domain!r} has {count} row(s), the gate requires "
                f"{need.min_rows_per_domain}")
    if outcome.sample["true_discrepancies"] < need.min_true_discrepancies:
        outcome.blocking.append(
            f"{outcome.sample['true_discrepancies']} true discrepancies, the "
            f"gate requires {need.min_true_discrepancies}")
    for route, minimum in need.min_rows_per_route.items():
        seen = outcome.sample["rows_per_route"].get(route, 0)
        if seen < minimum:
            outcome.blocking.append(
                f"route {route!r} has {seen} row(s), the gate requires {minimum}")
    if need.held_out_domain_required and not held_out_domain:
        outcome.blocking.append(
            "the gate must be evaluated on a domain held out of development, "
            "and none was named")
    elif held_out_domain and held_out_domain not in by_domain:
        outcome.blocking.append(
            f"the held-out domain {held_out_domain!r} is not present in the "
            "evidence supplied, so nothing was actually held out")

    discrepancies_by_domain: Counter = Counter()
    studies_with_discrepancies = 0
    for study, study_rows in by_study.items():
        positives = sum(1 for r in study_rows if _expected(r) in FLAG_LABELS)
        discrepancies_by_domain[domains.get(study, "unspecified")] += positives
        studies_with_discrepancies += bool(positives)
    outcome.sample["discrepancies_per_domain"] = dict(discrepancies_by_domain)
    outcome.sample["studies_with_discrepancies"] = studies_with_discrepancies
    for domain in sorted(by_domain):
        found = discrepancies_by_domain.get(domain, 0)
        if found < need.min_true_discrepancies_per_domain:
            outcome.blocking.append(
                f"domain {domain!r} has {found} true discrepancy/ies, the gate "
                f"requires {need.min_true_discrepancies_per_domain}")
    if studies_with_discrepancies < need.min_studies_with_discrepancies:
        outcome.blocking.append(
            f"{studies_with_discrepancies} study/studies carry a true "
            f"discrepancy, the gate requires "
            f"{need.min_studies_with_discrepancies}")

    # Hard gates are counts, checked exactly, never averaged — and an absence
    # is never a zero.
    for hard in gate.hard_gates:
        seen = evidence.get(hard.metric)
        denominator = None
        if seen is None or seen.value is None:
            state = MISSING
        else:
            denominator = seen.denominator
            if hard.evidence_denominator:
                graded = evidence.get(hard.evidence_denominator)
                denominator = (graded.value if graded and graded.value is not None
                               else 0)
            if denominator is not None and denominator <= 0:
                state = NO_DENOMINATOR
            else:
                state = OK if float(seen.value) == hard.must_equal else FAILED
        outcome.hard.append({
            "metric": hard.metric,
            "observed": None if seen is None else seen.value,
            "required": hard.must_equal, "state": state, "passed": state == OK,
            "denominator": denominator,
            "denominator_metric": hard.evidence_denominator,
            "source": "" if seen is None else seen.source, "why": hard.why})

    # Studies grouped under their domain, both in a stable order. The held-out
    # domain is kept OUT of the pooled estimate and reported on its own.
    strata: dict[str, list[list[Any]]] = defaultdict(list)
    for study, study_rows in by_study.items():
        strata[domains.get(study, "unspecified")].append(study_rows)
    pooled = [strata[domain] for domain in sorted(strata)
              if domain != held_out_domain]
    held_out = strata.get(held_out_domain, [])

    for capability in gate.capability_gates:
        statistic = STATISTICS.get(capability.metric)
        if statistic is None:
            raise ContractError(
                f"the gate names an unknown statistic: {capability.metric}")
        estimate = stratified_bootstrap(
            pooled, statistic, resamples=gate.bootstrap_resamples,
            seed=gate.bootstrap_seed, confidence=gate.confidence_level,
            max_undefined_rate=gate.max_undefined_draw_rate)
        lower = estimate["lower"]
        passed = None if lower is None else bool(
            lower >= capability.lower_bound_at_least)
        check = {"metric": capability.metric, **estimate,
                 "required": capability.lower_bound_at_least,
                 "passed": passed, "why": capability.why,
                 "per_domain": {domain: statistic(
                     [row for cluster in strata[domain] for row in cluster])
                     for domain in sorted(strata)}}
        if held_out:
            check["held_out"] = domain_weighted([held_out], statistic)
        outcome.capability.append(check)
        if estimate["note"]:
            outcome.blocking.append(f"{capability.metric}: {estimate['note']}")

    for metric in gate.reported_only:
        statistic = STATISTICS.get(metric)
        if statistic is not None:
            outcome.reported[metric] = statistic(rows)

    if any(c["state"] == FAILED for c in outcome.hard):
        # An observed safety failure is a FAIL whatever the evidence supports:
        # "we cannot estimate accuracy" is no defence for having released one.
        outcome.status = FAIL
    elif any(c["state"] in (MISSING, NO_DENOMINATOR) for c in outcome.hard):
        # No evidence is not a pass, and it is not a failure either. Saying
        # which is the difference between "we did not look" and "we looked and
        # it broke".
        outcome.status = NOT_ESTIMABLE
        outcome.blocking += [
            (f"{c['metric']}: no evidence was supplied" if c["state"] == MISSING
             else f"{c['metric']}: reported, but nothing was graded against it "
                  f"({c['denominator_metric'] or 'denominator'} = {c['denominator']})")
            for c in outcome.hard if c["state"] in (MISSING, NO_DENOMINATOR)]
    elif outcome.blocking:
        outcome.status = NOT_ESTIMABLE
    elif any(c["passed"] is False for c in outcome.capability):
        outcome.status = FAIL
    else:
        outcome.status = PASS

    # Releasing needs more than passing. A provisional gate is a proposal about
    # what would be good enough; meeting a proposal authorises nothing.
    if outcome.status != PASS:
        outcome.release_blockers.append(
            f"the evaluation returned {outcome.status}")
    if not gate.signed_off:
        outcome.release_blockers.append(
            f"gate {gate.gate_id} is {gate.status}: its thresholds have not been "
            "signed off by anyone answerable for the risk they encode")
    outcome.release_eligible = not outcome.release_blockers
    return outcome


def _get(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def _expected(row: Any) -> str:
    return str(_get(row, "expected_label") or "")


def _predicted(row: Any) -> str:
    return str(_get(row, "predicted_label") or "")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
