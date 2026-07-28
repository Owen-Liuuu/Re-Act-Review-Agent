"""C1 eval runner: Collector + audit accuracy vs the benchmark answer key.

For each answer-key row (audit_template.csv) the Collector reads the REAL local
source PDF and extracts the value with the LLM; the audit compares it to the
review value; results are scored (see react_review.eval_accuracy). Slow + costs
tokens, so drive it with --config and --limit / --studies.

    python eval/run_full_accuracy.py --config configs/config.local.yaml --limit 5
    python eval/run_full_accuracy.py --dry-run          # offline: show the key
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

# stdout may be GBK on Windows consoles; force utf-8 for quotes/units.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import react_review.cli  # noqa: F401  prime the import graph
from react_review.agents.collector import Collector
from react_review.audit import ToleranceTable
from react_review.core.config import load_config
from react_review.csv_io import load_included_studies
from react_review.eval_accuracy import format_report, run_rows, score_rows
from react_review.normalize.vocabulary import Vocabulary
from react_review.pipeline.factory import _create_llm_backend
from react_review.retrieval.local_pdf import LocalPdfRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.registry import ToolRegistry

BENCH = Path(__file__).resolve().parent / "benchmark"
ROOT = BENCH.parent.parent


def _load_answer_key(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("study_id") or "").strip()]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="C1 full-pipeline accuracy eval.")
    ap.add_argument("--config", type=Path, default=None,
                    help="LLM config with an api_key (required unless --dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    ap.add_argument("--studies", default="", help="comma-separated study_id filter")
    ap.add_argument("--context", default="EAT thickness/volume in T1DM vs healthy controls")
    ap.add_argument("--tolerances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write a JSON report")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: just show the answer-key distribution")
    args = ap.parse_args(argv)

    rows = _load_answer_key(BENCH / "audit_template.csv")
    if args.studies:
        keep = {s.strip() for s in args.studies.split(",") if s.strip()}
        rows = [r for r in rows if r["study_id"] in keep]
    if args.limit:
        rows = rows[: args.limit]

    if args.dry_run:
        print(f"rows: {len(rows)}")
        print("by study        :", dict(Counter(r["study_id"] for r in rows)))
        print("by expected_label:", dict(Counter(r["expected_label"] for r in rows)))
        return

    if args.config is None:
        ap.error("--config is required unless --dry-run")

    studies = load_included_studies(BENCH / "included_studies.csv")
    by_id = {s.study_id: s for s in studies}

    backend = _create_llm_backend(load_config(args.config))
    tol = ToleranceTable.from_yaml(args.tolerances) if args.tolerances else ToleranceTable()
    retriever = LocalPdfRetriever(
        {s.doi: s.source_pdf for s in studies if s.doi and s.source_pdf}, base_dir=BENCH)
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(retriever))
    reg.register(ExtractSourceValueTool(backend))
    vocab = Vocabulary.from_json(ROOT / "configs" / "vocabulary.seed.json")
    collector = Collector(reg, vocabulary=vocab)

    def reference_for(study_id: str) -> ReferenceEntry:
        s = by_id.get(study_id)
        return ReferenceEntry(title=(s.review_citation if s else study_id),
                              doi=(s.doi if s else None))

    print(f"running {len(rows)} rows through Collector + audit …")
    results = asyncio.run(run_rows(rows, collector, tol, reference_for, args.context))

    for r in results:
        mark = "ok" if r.predicted_label == r.expected_label else "XX"
        print(f"[{mark}] {r.study_id}/{r.group}/{r.field_type}: "
              f"pred={r.predicted_label} exp={r.expected_label} | "
              f"src '{r.extracted_source}' vs '{r.expected_source}' [{r.outcome}]")

    metrics = score_rows(results)
    print(format_report(metrics))

    if args.out:
        args.out.write_text(
            json.dumps({"metrics": metrics, "rows": [asdict(r) for r in results]},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] {args.out.resolve()}")


if __name__ == "__main__":
    main()
