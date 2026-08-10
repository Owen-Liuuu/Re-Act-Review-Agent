"""Apply a pre-registered acceptance gate to eval artifacts.

    python eval/check_gate.py REPORT.json --domain melanoma
    python eval/check_gate.py A.json B.json --domain melanoma --domain eat \
        --attestation docs/acceptance/attestation_2026-08-10.json

Exit codes: 0 passed AND may authorise a release, 1 failed, 2 could not be
evaluated on the evidence supplied, 3 passed but may not authorise anything
(most often because the gate itself is still provisional). "We have not shown
this", "we tried and it did not hold" and "the numbers cleared a bar we invented
ourselves" are three different sentences, and a release decision needs to know
which one it is reading.

Nothing here invents evidence. A safety number a report does not carry is
absent, not zero; a number with nothing graded behind it is reported as such;
and the two facts no run can certify about itself — that its answer key was not
edited after freezing, and that its results reproduce from their own recording —
must come from a signed attestation file or they count as missing.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from react_review.acceptance import (
    FAIL,
    NOT_ESTIMABLE,
    PASS,
    Observation,
    evaluate_gate,
    load_gate,
)
from react_review.contracts import ContractError, read_json_object, sha256_file

RELEASE_BLOCKED = 3
#: A domain may be held out once. After that it has been seen.
HELD_OUT_REGISTER = Path(__file__).resolve().parent.parent / "configs" / "gates" \
    / "held_out_register.json"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GATE = ROOT / "configs" / "gates" / "cross_domain_v1.json"

#: What an attestation must say, and nobody else may say for it.
ATTESTED_METRICS = ("answer_key_edits_after_freeze", "record_replay_differences")


def _dig(body: dict, *path, default=None):
    for key in path:
        if not isinstance(body, dict) or key not in body:
            return default
        body = body[key]
    return body


def _sum_metric(reports: list[dict], *path) -> tuple[float | None, int]:
    """Sum a metric across reports — and say how many actually reported it.

    A report that carries no such number contributes nothing and is counted, so
    a total of zero over zero reporters can never be mistaken for a clean sheet.
    """
    total, reporting = 0.0, 0
    for report in reports:
        value = _dig(report, "metrics", *path)
        if value is not None:
            total += float(value)
            reporting += 1
    return (total if reporting else None), reporting


def collect_evidence(reports: list[dict], sources: list[str],
                     attestation: dict | None) -> dict[str, Observation]:
    """Every safety number the gate can read, with what it rests on."""
    evidence: dict[str, Observation] = {}
    named = ", ".join(sources)

    silent, reporting = _sum_metric(reports, "safety", "silent_release_count")
    if reporting == len(reports):
        evidence["silent_release_count"] = Observation(
            value=silent, denominator=sum(
                int(_dig(r, "metrics", "safety", "expected_discrepancies", default=0))
                for r in reports), source=named)

    visibility = [float(v) for v in
                  (_dig(r, "metrics", "safety", "review_visibility_rate")
                   for r in reports) if v is not None]
    if len(visibility) == len(reports) and visibility:
        # The weakest report decides: one benchmark hiding a discrepancy is not
        # excused by another that hid none.
        evidence["review_visibility_rate"] = Observation(
            value=min(visibility), denominator=sum(
                int(_dig(r, "metrics", "safety", "expected_discrepancies", default=0))
                for r in reports), source=named)

    for metric, path, denominator_path in (
            ("wrong_target_released_count",
             ("target", "gold", "identity_wrong_released"), ("target", "gold", "rows")),
            ("wrong_scope_released_count",
             ("scope", "scope_wrong_released_count"), ("scope", "gold_rows"))):
        value, reporting = _sum_metric(reports, *path)
        if reporting == len(reports):
            # The denominator is the WEAKEST report, not the total. Summing
            # would let one benchmark's graded rows vouch for another's: eleven
            # identity rows all from one paper would otherwise read as evidence
            # that no wrong arm was released anywhere.
            graded_per_report = [
                _dig(r, "metrics", *denominator_path, default=0) for r in reports]
            evidence[metric] = Observation(
                value=value, denominator=min(int(g or 0) for g in graded_per_report),
                source=named)
            evidence[f"{metric}__graded_rows"] = Observation(
                value=min(int(g or 0) for g in graded_per_report),
                source=f"{named} (weakest of {graded_per_report})")

    if attestation:
        for metric in ATTESTED_METRICS:
            if metric in attestation:
                evidence[metric] = Observation(
                    value=float(attestation[metric]),
                    denominator=None,
                    source=f"attestation by {attestation.get('attested_by') or '?'}")
    return evidence


def load_attestation(path: Path, reports: list[Path]) -> dict:
    """An attestation is only about the artifacts it names and hashes."""
    body = read_json_object(path, kind="attestation")
    if not body.get("attested_by"):
        raise ContractError("an attestation must name who is attesting")
    covered = {str(k): str(v).upper() for k, v in (body.get("artifacts") or {}).items()}
    for report in reports:
        declared = covered.get(report.name)
        if declared is None:
            raise ContractError(
                f"the attestation does not cover {report.name}; it cannot speak "
                "for a report it never saw")
        actual = sha256_file(report)
        if declared != actual:
            raise ContractError(
                f"{report.name} has changed since it was attested "
                f"(attestation {declared[:16]}…, file {actual[:16]}…)")
    return body


def check_held_out(domain: str, register_path: Path) -> list[str]:
    """A held-out domain must be registered, frozen, and not already spent."""
    if not domain:
        return []
    if not register_path.is_file():
        return [f"no held-out register at {register_path.name}, so {domain!r} "
                "cannot be shown to have been held out"]
    register = read_json_object(register_path, kind="held-out register")
    entry = (register.get("domains") or {}).get(domain)
    if entry is None:
        return [f"domain {domain!r} is not in the held-out register: naming a "
                "domain on the command line does not hold it out"]
    problems = []
    if not entry.get("frozen_on"):
        problems.append(f"{domain!r} has no freeze date in the register")
    used = entry.get("used_by") or []
    if used:
        problems.append(
            f"{domain!r} was already used as the held-out domain by "
            f"{', '.join(used)}; a domain is held out once")
    return problems


def collect_provenance(report_paths: list[Path], reports: list[dict],
                       gate, attestation_path: Path | None) -> dict:
    """Everything the verdict rests on, by hash.

    A conclusion that cannot say which answer key, profile and recording
    produced it is a number, not a result.
    """
    inputs = {}
    for path, report in zip(report_paths, reports):
        run = report.get("run") or {}
        inputs[path.name] = {
            "report_sha256": sha256_file(path),
            "benchmark_id": run.get("benchmark_id", ""),
            "benchmark_profile_sha256": run.get("benchmark_profile_sha256", ""),
            "target_contract_sha256": run.get("target_contract_sha256", ""),
            "semantic_overlay_sha256": run.get("semantic_overlay_sha256", ""),
            "run_profile_sha256": run.get("run_profile_sha256", ""),
            "extraction_mode": run.get("extraction_mode", ""),
            "extraction_model_id": run.get("extraction_model_id", ""),
            "metrics_schema_version": (report.get("metrics") or {})
                                      .get("metrics_schema_version"),
        }
    return {
        "inputs": inputs,
        "gate": {"path": str(gate.path), "sha256": gate.sha256,
                 "status": gate.status, "version": gate.version},
        "attestation": ({"path": str(attestation_path),
                         "sha256": sha256_file(attestation_path)}
                        if attestation_path else None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="apply a pre-registered acceptance gate")
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    ap.add_argument("--domain", action="append", default=[],
                    help="domain name for each report, in the same order; one "
                         "per report, no more and no fewer")
    ap.add_argument("--attestation", type=Path, default=None,
                    help="signed statement about answer-key freezing and "
                         "record/replay reproducibility")
    ap.add_argument("--held-out-domain", default="")
    ap.add_argument("--register", type=Path, default=HELD_OUT_REGISTER,
                    help="the one-use register of held-out domains")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.domain and len(args.domain) != len(args.reports):
        # Silently pairing by position was how a report could be filed under the
        # wrong domain, and every per-domain minimum judged against the wrong set.
        ap.error(f"{len(args.reports)} report(s) but {len(args.domain)} --domain "
                 "value(s); give exactly one domain per report")

    gate = load_gate(args.gate)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
    attestation = None
    if args.attestation:
        try:
            attestation = load_attestation(args.attestation, args.reports)
        except ContractError as exc:
            ap.error(f"attestation rejected: {exc}")

    rows, domains = [], {}
    for index, report in enumerate(reports):
        domain = (args.domain[index] if index < len(args.domain)
                  else str((report.get("run") or {}).get("benchmark_id")
                           or f"domain_{index}"))
        for row in report.get("rows") or []:
            rows.append(row)
            domains[str(row.get("study_id") or "")] = domain

    evidence = collect_evidence(reports, [p.name for p in args.reports], attestation)
    outcome = evaluate_gate(gate, rows, evidence=evidence, domains=domains,
                            held_out_domain=args.held_out_domain)
    outcome.blocking += check_held_out(args.held_out_domain, args.register)
    if outcome.blocking and outcome.status == PASS:
        outcome.status = NOT_ESTIMABLE
        outcome.release_eligible = False
        outcome.release_blockers.append("the evaluation returned not_estimable")
    outcome.provenance = collect_provenance(args.reports, reports, gate,
                                            args.attestation if attestation else None)

    print(outcome.summary())
    print(f"  sample: {outcome.sample}")
    if outcome.reported:
        print(f"  reported only: {outcome.reported}")
    if not attestation:
        print("  note: no attestation supplied, so answer-key freezing and "
              "record/replay reproducibility count as unevidenced")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(outcome.model_dump(mode="json"),
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[out] {args.out}")
    if outcome.status == PASS and not outcome.release_eligible:
        return RELEASE_BLOCKED
    return {PASS: 0, FAIL: 1, NOT_ESTIMABLE: 2}[outcome.status]


if __name__ == "__main__":
    raise SystemExit(main())
