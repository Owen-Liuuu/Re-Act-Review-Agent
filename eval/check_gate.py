"""Apply a pre-registered acceptance gate to an eval artifact.

    python eval/check_gate.py output/baselines/<report>.json
    python eval/check_gate.py A.json B.json --domain melanoma --domain eat

Exit code 0 only when the gate PASSES. A gate that cannot be evaluated on the
evidence supplied exits 2 — distinct from failing, because "we do not have
enough evidence to say" and "we tried and it did not hold" are different
sentences and a release decision needs to know which one it is reading.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from react_review.acceptance import FAIL, NOT_ESTIMABLE, PASS, evaluate_gate, load_gate

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GATE = ROOT / "configs" / "gates" / "cross_domain_v1.json"


def _hard_counts(reports: list[dict]) -> dict[str, float]:
    """Safety counts, summed across reports; rates required to hold in ALL."""
    silent = wrong_target = wrong_scope = 0
    visibility = []
    for report in reports:
        metrics = report.get("metrics") or {}
        safety = metrics.get("safety") or {}
        silent += int(safety.get("silent_release_count") or 0)
        gold = (metrics.get("target") or {}).get("gold") or {}
        wrong_target += int(gold.get("identity_wrong_released") or 0)
        wrong_scope += int((metrics.get("scope") or {})
                           .get("scope_wrong_released_count") or 0)
        rate = safety.get("review_visibility_rate")
        if rate is not None:
            visibility.append(float(rate))
    return {
        "silent_release_count": silent,
        "wrong_target_released_count": wrong_target,
        "wrong_scope_released_count": wrong_scope,
        "review_visibility_rate": min(visibility) if visibility else None,
        # Supplied by the release process, not by a report: a run cannot certify
        # that its own answer key was left alone.
        "answer_key_edits_after_freeze": 0,
        "record_replay_differences": 0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="apply a pre-registered acceptance gate")
    ap.add_argument("reports", type=Path, nargs="+")
    ap.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    ap.add_argument("--domain", action="append", default=[],
                    help="domain name for each report, in the same order")
    ap.add_argument("--held-out-domain", default="",
                    help="the domain kept out of development, if any")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    gate = load_gate(args.gate)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in args.reports]
    rows, domains = [], {}
    for index, report in enumerate(reports):
        domain = (args.domain[index] if index < len(args.domain)
                  else str((report.get("run") or {}).get("benchmark_id") or f"domain_{index}"))
        for row in report.get("rows") or []:
            rows.append(row)
            domains[str(row.get("study_id") or "")] = domain

    outcome = evaluate_gate(gate, rows, hard_counts=_hard_counts(reports),
                            domains=domains, held_out_domain=args.held_out_domain)
    print(outcome.summary())
    print(f"  sample: {outcome.sample}")
    if outcome.reported:
        print(f"  reported only: {outcome.reported}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(outcome.model_dump(mode="json"),
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"[out] {args.out}")
    return {PASS: 0, FAIL: 1, NOT_ESTIMABLE: 2}[outcome.status]


if __name__ == "__main__":
    raise SystemExit(main())
