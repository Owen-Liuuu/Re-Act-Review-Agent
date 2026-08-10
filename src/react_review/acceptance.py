"""Evaluating a PRE-REGISTERED acceptance gate.

A threshold chosen after seeing the result is not a threshold. So the gate lives
in its own hash-pinned file, written before the data exists, and this module
only applies it. Three things follow from that, and they are the reason this is
code rather than a paragraph in a report:

*The unit of analysis is the study, not the row.* Fifteen rows from one paper
are not fifteen independent observations — one bad extraction pass moves them
together. Intervals are therefore cluster-bootstrapped over studies, and a
benchmark with one study supports no interval at all. That is not a technicality
to work around; it is why "60%" and "80%" from the melanoma checkpoint were
never rankable against each other.

*Not estimable is a third outcome.* A gate that can only pass or fail will be
made to pass by whoever needs it to. Where the evidence cannot support the
question, this says so, and says which minimum was missed.

*Safety is separate from capability.* The hard gates are counts that must be
exactly zero — a discrepancy released silently, a value released about the wrong
arm or the wrong population. They are not averaged, not traded against accuracy,
and not reported as a percentage.
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


class HardGate(BaseModel):
    """A count that must be exactly zero, or a rate that must be exactly one."""

    metric: str
    must_equal: float
    why: str = ""


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
    sample: SampleRequirement = Field(default_factory=SampleRequirement)
    hard_gates: list[HardGate] = Field(default_factory=list)
    capability_gates: list[CapabilityGate] = Field(default_factory=list)
    reported_only: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


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

    def summary(self) -> str:
        head = {PASS: "PASS", FAIL: "FAIL",
                NOT_ESTIMABLE: "NOT ESTIMABLE"}[self.status]
        lines = [f"{head} — gate {self.gate_id} ({self.gate_sha256[:12]}…)"]
        lines += [f"  blocked: {reason}" for reason in self.blocking]
        for check in self.hard:
            mark = "ok" if check["passed"] else "XX"
            lines.append(f"  [{mark}] {check['metric']} = {check['observed']} "
                         f"(must be {check['required']})")
        for check in self.capability:
            mark = {True: "ok", False: "XX", None: "??"}[check["passed"]]
            lines.append(
                f"  [{mark}] {check['metric']} = {_fmt(check['point'])} "
                f"[{_fmt(check['lower'])}, {_fmt(check['upper'])}] "
                f"≥ {check['required']}")
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
        sample=SampleRequirement(**(body.get("sample") or {})),
        hard_gates=[HardGate(**g) for g in (body.get("hard_gates") or [])],
        capability_gates=[CapabilityGate(**g)
                          for g in (body.get("capability_gates") or [])],
        reported_only=[str(m) for m in (body.get("reported_only") or [])],
        notes=[str(n) for n in (body.get("notes") or [])])


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


def _source_coverage(rows: list[Any]) -> float | None:
    return (sum(bool(_get(r, "found")) for r in rows) / len(rows)) if rows else None


STATISTICS: dict[str, Callable[[list[Any]], float | None]] = {
    "discrepancy_recall": _recall,
    "discrepancy_precision": _precision,
    "label_accuracy": _label_accuracy,
    "source_coverage": _source_coverage,
}


def evaluate_gate(
    gate: AcceptanceGate, rows: Iterable[Any], *,
    hard_counts: dict[str, float], domains: dict[str, str] | None = None,
    held_out_domain: str = "",
) -> GateOutcome:
    """Apply a pre-registered gate to one set of scored rows.

    ``domains`` maps study_id -> domain; without it every study is treated as
    one domain, which is exactly the situation the sample requirements exist to
    refuse.
    """
    rows = list(rows)
    outcome = GateOutcome(gate_id=gate.gate_id, gate_sha256=gate.sha256)

    by_study: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        by_study[str(_get(row, "study_id") or "")].append(row)
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
    if outcome.sample["rows"] < need.min_rows_per_domain:
        outcome.blocking.append(
            f"{outcome.sample['rows']} rows, the gate requires "
            f"{need.min_rows_per_domain} per domain")
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

    # Hard gates are counts, checked exactly, never averaged.
    for hard in gate.hard_gates:
        observed = hard_counts.get(hard.metric)
        passed = observed is not None and float(observed) == hard.must_equal
        outcome.hard.append({"metric": hard.metric, "observed": observed,
                             "required": hard.must_equal, "passed": passed,
                             "why": hard.why})

    clusters = list(by_study.values())
    for capability in gate.capability_gates:
        statistic = STATISTICS.get(capability.metric)
        if statistic is None:
            raise ContractError(
                f"the gate names an unknown statistic: {capability.metric}")
        point, lower, upper, note = cluster_bootstrap(
            clusters, statistic, resamples=gate.bootstrap_resamples,
            seed=gate.bootstrap_seed, confidence=gate.confidence_level)
        passed = None if lower is None else bool(lower >= capability.lower_bound_at_least)
        outcome.capability.append({
            "metric": capability.metric, "point": point, "lower": lower,
            "upper": upper, "required": capability.lower_bound_at_least,
            "passed": passed, "note": note, "why": capability.why})
        if note:
            outcome.blocking.append(f"{capability.metric}: {note}")

    for metric in gate.reported_only:
        statistic = STATISTICS.get(metric)
        if statistic is not None:
            outcome.reported[metric] = statistic(rows)

    hard_failed = [c for c in outcome.hard if not c["passed"]]
    if hard_failed:
        # A safety gate failing is a FAIL, whatever the evidence supports:
        # "we cannot estimate accuracy" is no defence for having released one.
        outcome.status = FAIL
    elif outcome.blocking:
        outcome.status = NOT_ESTIMABLE
    elif any(c["passed"] is False for c in outcome.capability):
        outcome.status = FAIL
    else:
        outcome.status = PASS
    return outcome


def _get(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, dict) else getattr(row, name, None)


def _expected(row: Any) -> str:
    return str(_get(row, "expected_label") or "")


def _predicted(row: Any) -> str:
    return str(_get(row, "predicted_label") or "")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
