"""Grade the DKB FieldResolver's field_type mapping against the benchmark.

Runs the resolver over the review's Table-1 column headers in
``review_ground_truth.csv`` and reports how many resolve to the ground-truth
field_type, and by which path (cache / deterministic / retrieval_llm).
Deterministic by default (no LLM backend) — the seed KB should cover the
benchmark; any header that would need the LLM is reported as unresolved rather
than silently guessed.

Usage:  python eval/grade_normalize.py
"""
from __future__ import annotations

import asyncio
import csv
from collections import Counter
from pathlib import Path

from react_review.dkb import FieldResolver, load_runtime_knowledge

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "configs" / "knowledge.seed.json"
BENCH = ROOT / "eval" / "benchmark_1" / "review_ground_truth.csv"
CONTEXT = "EAT thickness/volume in type 1 diabetes vs healthy controls"


async def main() -> int:
    with BENCH.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["source_location"] == "Table 1"]

    # backend=None → deterministic KB resolution only; a miss stays unresolved.
    kb = load_runtime_knowledge(SEED, ROOT / "configs" / "ontology")
    resolver = FieldResolver(kb)
    correct = 0
    sources: Counter[str] = Counter()
    wrong: list[tuple[str, str, str, str]] = []
    unresolved: list[tuple[str, str]] = []

    for r in rows:
        rf = await resolver.resolve(
            r["raw_field_name"], unit=r["unit"], research_context=CONTEXT)
        if rf.field_type is None:
            unresolved.append((r["raw_field_name"], r["unit"]))
            continue
        sources[rf.source] += 1
        if rf.field_type == r["field_type"]:
            correct += 1
        else:
            wrong.append((r["raw_field_name"], r["unit"], r["field_type"], rf.field_type))

    n = len(rows)
    print(f"FieldResolver on {n} Table-1 columns")
    print(f"  runtime KB: {len(kb.entries)} concepts, fingerprint {kb.version}")
    print(f"  ontology imports: {[r.source for r in kb.imports]}")
    print(f"  correct: {correct}/{n} ({correct / n:.0%})")
    print(f"  by path: {dict(sources)}")
    for raw, unit, exp, got in wrong:
        print(f"  WRONG '{raw}' ({unit}): expected {exp}, got {got}")
    for raw, unit in unresolved:
        print(f"  UNRESOLVED (needs LLM) '{raw}' ({unit})")

    ok = correct == n and not unresolved
    print("\nGRADE PASS" if ok else "\nGRADE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
