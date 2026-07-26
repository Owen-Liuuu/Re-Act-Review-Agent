"""Grade the normalize_field tool's field_type mapping against the benchmark.

Runs normalize_field over the review's Table-1 column headers in
``review_ground_truth.csv`` and reports how many resolve to the ground-truth
field_type, and by which path (cache / vocabulary / llm). Deterministic by
default (no LLM backend) — the seed vocabulary should cover the benchmark; any
row that would need the LLM is reported rather than silently guessed.

Usage:  python eval/grade_normalize.py
"""
from __future__ import annotations

import asyncio
import csv
from collections import Counter
from pathlib import Path

from react_review.normalize.vocabulary import Vocabulary
from react_review.tools.models import NormalizeInput
from react_review.tools.normalize import NormalizeFieldTool

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "configs" / "vocabulary.seed.json"
BENCH = ROOT / "eval" / "benchmark" / "review_ground_truth.csv"
CONTEXT = "EAT thickness/volume in type 1 diabetes vs healthy controls"


async def main() -> int:
    with BENCH.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["source_location"] == "Table 1"]

    tool = NormalizeFieldTool(Vocabulary.from_json(SEED), backend=None)
    correct = 0
    sources: Counter[str] = Counter()
    wrong: list[tuple[str, str, str, str]] = []
    unresolved: list[tuple[str, str]] = []

    for r in rows:
        try:
            out = await tool.run(NormalizeInput(
                raw_field_name=r["raw_field_name"], unit=r["unit"],
                research_context=CONTEXT,
            ))
        except ValueError:
            unresolved.append((r["raw_field_name"], r["unit"]))
            continue
        sources[out.source] += 1
        if out.field_type == r["field_type"]:
            correct += 1
        else:
            wrong.append((r["raw_field_name"], r["unit"], r["field_type"], out.field_type))

    n = len(rows)
    print(f"normalize_field on {n} Table-1 columns")
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
