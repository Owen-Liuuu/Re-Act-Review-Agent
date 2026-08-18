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
from react_review.audit.semantic_cache import SemanticCache
from react_review.audit.semantic_control import (
    format_threshold_sensitivity,
    threshold_sensitivity,
)
from react_review.core.config import load_config
from react_review.csv_io import load_included_studies
from react_review.eval_accuracy import row_payload, format_report, run_rows, score_rows
from react_review.eval_benchmark import (
    benchmark_diagnostics,
    format_benchmark_diagnostics,
    validate_frozen_benchmark,
)
from react_review.eval_excerpt import (
    benchmark_cohorts as _cohort_registry,
    coverage_for_run,
)
from react_review.eval_profile import ProfileError, load_profile
from react_review.dkb import load_runtime_knowledge
from react_review.llm.metered import MeteredBackend
from react_review.schemas.run_manifest import RunManifest
from react_review.schemas.telemetry import (
    BATCH_EXTRACTION,
    SEMANTIC,
    SINGLE_EXTRACTION,
)
from react_review.tools.extract_batch import ExtractSourceBatchTool
from react_review.schemas.telemetry import RunTelemetry, wall_clock
from react_review.llm.factory import create_llm_backend
from react_review.production import evidence_adequacy_runtime
from react_review.retrieval.local_pdf import LocalPdfRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.compare import CompareValuesTool
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import ExtractSourceValueTool
from react_review.tools.extraction_cache import ExtractionCache
from react_review.tools.registry import ToolRegistry
from react_review.tools.semantic_compare import SemanticCompareTool

DEFAULT_BENCH = Path(__file__).resolve().parent / "benchmark_1"
ROOT = DEFAULT_BENCH.parent.parent


def _load_answer_key(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("study_id") or "").strip()]


def _aggregation_runtime(contract):
    """The policy and the identity that cleared it, bound, or nothing.

    Nothing when the contract does not batch: recording an identity a run never
    used would attribute its answers to code that never ran.
    """
    if contract is None or not getattr(contract, "batching", False):
        return None
    from react_review.tools.safe_aggregation import AggregationRuntime

    return AggregationRuntime.resolve(
        policy_id=contract.aggregation_policy_id,
        evaluator_version=contract.evaluator_version)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="C1 full-pipeline accuracy eval.")
    ap.add_argument("--config", type=Path, default=None,
                    help="LLM config with an api_key (required unless --dry-run)")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH,
                    help="benchmark directory (default: the legacy EAT benchmark)")
    ap.add_argument("--studies-file", type=Path, default=None,
                    help="included-study CSV; relative paths are resolved inside "
                         "the benchmark (default: included_studies.csv, then "
                         "selected_studies.csv)")
    ap.add_argument("--benchmark-profile", default=None,
                    help="a profile FILE inside the benchmark (e.g. "
                         "phase7_profile.json) selecting the prompt contracts, "
                         "the target contract and the semantic overlay. Omit to "
                         "run the benchmark's own frozen contract unchanged.")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    ap.add_argument("--studies", default="", help="comma-separated study_id filter")
    ap.add_argument("--fields", default="", help="comma-separated field_type filter")
    ap.add_argument("--context", default=None,
                    help="research context for extraction and semantic comparison; "
                         "defaults to the frozen manifest domain when available")
    ap.add_argument("--tolerances", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write a JSON report")
    ap.add_argument("--html", type=Path, default=None, help="write an HTML test report")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline: just show the answer-key distribution")
    ap.add_argument("--semantic", choices=("off", "cache-only", "on"), default="off",
                    help="judge text the numeric comparison cannot read "
                         "(default off: the eval stays deterministic)")
    ap.add_argument("--semantic-cache", type=Path, default=None,
                    help="the recording both `on` and `cache-only` runs share, so a "
                         "replay reproduces the run that recorded it")
    ap.add_argument("--extraction", choices=("live", "record", "replay"), default="live",
                    help="live calls the LLM; record also saves raw model JSON; "
                         "replay makes no extraction LLM calls")
    ap.add_argument("--extraction-cache", type=Path, default=None,
                    help="raw extraction recording used by record/replay")
    args = ap.parse_args(argv)

    benchmark = args.benchmark.resolve()
    legacy_default = benchmark == DEFAULT_BENCH.resolve()
    if not benchmark.is_dir():
        ap.error(f"benchmark directory does not exist: {benchmark}")
    answer_key = benchmark / "audit_template.csv"
    if not answer_key.is_file():
        ap.error(f"benchmark answer key does not exist: {answer_key}")

    if args.studies_file is None:
        candidates = (benchmark / "included_studies.csv",
                      benchmark / "selected_studies.csv")
        studies_path = next((path for path in candidates if path.is_file()), candidates[0])
    else:
        studies_path = (args.studies_file if args.studies_file.is_absolute()
                        else benchmark / args.studies_file)
    if not studies_path.is_file():
        ap.error(f"included-study CSV does not exist: {studies_path}")

    artifact_dir = ROOT / "output" / "baselines" / benchmark.name
    semantic_cache_path = (args.semantic_cache
                           or (DEFAULT_BENCH / "semantic_cache.json" if legacy_default
                               else artifact_dir / "semantic_cache.json"))
    extraction_cache_path = (args.extraction_cache
                             or (ROOT / "output" / "baselines" /
                                 "extraction_replay.json" if legacy_default
                                 else artifact_dir / "extraction_replay.json"))

    manifest_path = benchmark / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                if manifest_path.is_file() else {})
    context = args.context or str(
        manifest.get("domain") or "EAT thickness/volume in T1DM vs healthy controls")

    all_rows = _load_answer_key(answer_key)
    gate = validate_frozen_benchmark(benchmark, all_rows)
    if gate["status"] == "failed":
        ap.error("frozen benchmark entry gate failed: " + "; ".join(gate["errors"]))

    # A profile applies only when the run names its file: an unprofiled run of a
    # frozen benchmark must keep behaving exactly as it was recorded.
    profile = None
    if args.benchmark_profile:
        try:
            profile = load_profile(
                benchmark, args.benchmark_profile,
                answer_key_ids=[r.get("audit_id", "") for r in all_rows])
        except ProfileError as exc:
            ap.error(f"benchmark profile rejected: {exc}")

    rows = list(all_rows)
    if args.studies:
        keep = {s.strip() for s in args.studies.split(",") if s.strip()}
        rows = [r for r in rows if r["study_id"] in keep]
    if args.fields:
        keep_fields = {s.strip() for s in args.fields.split(",") if s.strip()}
        rows = [r for r in rows if r["field_type"] in keep_fields]
    if args.limit:
        rows = rows[: args.limit]

    print(f"benchmark: {benchmark} [{gate['status']}]")
    if profile is not None:
        contract = profile.run_contract
        print(f"profile  : {profile.path.name} "
              f"[extraction={profile.extraction_profile} "
              f"semantic={profile.semantic_prompt_profile} "
              f"targets={len(profile.targets)} "
              f"semantic_overlay={len(profile.semantic)}]")
        if contract is not None:
            print(f"contract : {contract.profile_id} "
                  f"[tolerances={(contract.tolerances_path.name if contract.tolerances_path else 'defaults')} "
                  f"scope={contract.scope_policy}]")
    if args.dry_run:
        print(f"rows: {len(rows)}")
        print("by study        :", dict(Counter(r["study_id"] for r in rows)))
        print("by expected_label:", dict(Counter(r["expected_label"] for r in rows)))
        modes = Counter((r.get("expected_match_mode") or "") for r in rows)
        if any(key for key in modes):
            print("by expected mode :", dict(modes))
            print("known gaps       :", [r.get("audit_id") for r in rows
                                          if (r.get("known_gap") or "").strip()])
        return

    needs_backend = args.extraction != "replay" or args.semantic == "on"
    if args.config is None and needs_backend:
        ap.error("--config is required for live extraction or live semantic comparison")

    studies = load_included_studies(studies_path)
    by_id = {s.study_id: s for s in studies}

    telemetry = RunTelemetry()
    backend = (create_llm_backend(load_config(args.config))
               if args.config is not None else None)
    # Per-stage measurement is switched on for a run that HAS more than one
    # stage to compare. A single-route run has nothing to compare and its global
    # counters already say what it cost, so labelling it would add a section to
    # every artifact ever replayed for a measurement nobody was making.
    batching = bool(profile is not None and profile.run_contract is not None
                    and profile.run_contract.batching)
    stages = ((SINGLE_EXTRACTION, BATCH_EXTRACTION, SEMANTIC) if batching
              else ("", "", ""))
    if backend is not None:
        raw_backend = backend
        backend = MeteredBackend(raw_backend, telemetry, stages[0])
        batch_backend = MeteredBackend(raw_backend, telemetry, stages[1])
        semantic_backend = MeteredBackend(raw_backend, telemetry, stages[2])
    else:
        batch_backend = semantic_backend = None
    tol = ToleranceTable.from_yaml(args.tolerances) if args.tolerances else ToleranceTable()
    retriever = LocalPdfRetriever(
        {s.doi: s.source_pdf for s in studies if s.doi and s.source_pdf}, base_dir=benchmark)
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(retriever))
    extraction_cache = (None if args.extraction == "live"
                        else ExtractionCache(extraction_cache_path))
    reg.register(ExtractSourceValueTool(
        backend, cache=extraction_cache, cache_mode=args.extraction,
        telemetry=telemetry, stage=stages[0]))
    # Registered whatever the contract says: a contract that routes to it must
    # find it, and one that does not never asks. Its absence is a startup
    # failure rather than a quiet fall back to reading one claim at a time.
    reg.register(ExtractSourceBatchTool(
        batch_backend, cache=extraction_cache, cache_mode=args.extraction,
        telemetry=telemetry))
    kb = load_runtime_knowledge(
        ROOT / "configs" / "knowledge.seed.json", ROOT / "configs" / "ontology")
    # The cohorts come from the profile's target contract — the review's own
    # words for its arms. Without them the Collector had no cohort registry at
    # all, so its wrong-arm guard was inert for the whole benchmark.
    cohorts = _cohort_registry(profile, all_rows) if profile is not None else None
    # The run contract, and — only when it batches — the bound policy and
    # evaluator identity that clears it. Resolved ONCE here, because readiness
    # shells out to git and a check that runs per claim is a check somebody
    # eventually removes for being slow.
    contract = profile.run_contract if profile is not None else None
    runtime = _aggregation_runtime(contract)
    run_meta_runtime = RunManifest.runtime_of(runtime)
    adequacy_evaluator = evidence_adequacy_runtime(contract)
    run_meta_adequacy = RunManifest.adequacy_of(adequacy_evaluator)
    collector = Collector(
        reg, knowledge=kb, cohorts=cohorts, contract=contract,
        aggregation_runtime=runtime,
        adequacy_evaluator=adequacy_evaluator,
        extraction_profile=(profile.extraction_profile if profile is not None
                            else "legacy_v3"))

    # One shared cache file across runs — a fresh empty cache per run would make
    # `cache-only` miss everything the `on` run just recorded.
    cache = None if args.semantic == "off" else SemanticCache(semantic_cache_path)
    semantic_profile = (profile.semantic_prompt_profile if profile is not None
                        else "semantic_v1")
    # The runtime contract decides tolerances and scope requirements. Without a
    # profile nothing changes: no contract, no requirements, no new behaviour.
    run_contract = profile.run_contract if profile is not None else None
    if run_contract is not None and run_contract.tolerances_path is not None:
        tol = ToleranceTable.from_yaml(run_contract.tolerances_path)
    scope_axes = (run_contract.required_scope_axes
                  if run_contract is not None and run_contract.scope_enabled else {})
    comparator = CompareValuesTool(
        tol,
        # The SEMANTIC wrapper, not the extraction one. Both wrap the same
        # backend; passing the wrong label put every semantic request, token and
        # second into the single-extraction bucket and left the semantic stage
        # empty, which is worse than no per-stage numbers at all.
        semantic=(SemanticCompareTool(semantic_backend, profile=semantic_profile)
                  if args.semantic == "on" else None),
        semantic_mode=args.semantic, semantic_cache=cache,
        min_confidence=tol.semantic_min_confidence,
        semantic_profile=semantic_profile,
        required_scope_axes=scope_axes)

    def reference_for(study_id: str) -> ReferenceEntry:
        s = by_id.get(study_id)
        return ReferenceEntry(title=(s.review_citation if s else study_id),
                              doi=(s.doi if s else None))

    print(f"running {len(rows)} rows through Collector + audit …")
    with wall_clock(telemetry):
        results = asyncio.run(run_rows(
            rows, collector, tol, reference_for, context, comparator=comparator,
            targets=(profile.targets if profile is not None else None),
            gold=(profile.gold if profile is not None else None),
            require_evidence_adequacy=(
                bool(contract and contract.adequacy_enabled)),
            adequacy_identity=(adequacy_evaluator.identity
                               if adequacy_evaluator is not None else None)))
    if extraction_cache is not None:
        telemetry.record_cache(hits=extraction_cache.hits,
                               misses=extraction_cache.misses)
        if args.extraction == "record":
            extraction_cache.save()
        print(f"extraction: mode={args.extraction} "
              f"{extraction_cache.hits} reused / {extraction_cache.misses} cache misses "
              f"({len(extraction_cache)} recorded) -> {extraction_cache_path}")
    else:
        print("extraction: mode=live; model variance is not hidden by a replay")
    if cache is not None:
        telemetry.record_cache(hits=cache.hits, misses=cache.misses)
        cache.save()
        print(f"semantic: mode={args.semantic} "
              f"min_confidence={tol.semantic_min_confidence} "
              f"{cache.hits} reused / {cache.misses} new "
              f"({cache.hit_rate:.0%} from cache) -> {semantic_cache_path}")
        vs = cache.verdicts()
        line = format_threshold_sensitivity(threshold_sensitivity(vs), len(vs))
        if line:
            print(f"semantic: {line}")

    for r in results:
        mark = "ok" if r.predicted_label == r.expected_label else "XX"
        print(f"[{mark}] {r.study_id}/{r.group}/{r.field_type}: "
              f"pred={r.predicted_label} exp={r.expected_label} | "
              f"src '{r.extracted_source}' vs '{r.expected_source}' [{r.outcome}]")

    # Was the evidence even sent? Benchmark-only, and it changes no answer: it
    # separates "the paper does not report it" from "the passage was never in
    # the window", which are the same sentence from the extractor and different
    # problems with different owners.
    coverage = coverage_for_run(
        results, studies, benchmark,
        (profile.excerpt_gold_path if profile is not None else None))

    from react_review.schemas.run_manifest import RunManifest as _RM
    _compare_runtime = _RM.compare_of(contract)

    metrics = score_rows(results)
    print(format_report(metrics))
    if coverage is not None and not coverage.assessable:
        print(f"excerpt: NOT ASSESSABLE — {coverage.reason}")
    elif coverage is not None:
        counts = coverage.tally
        print(f"excerpt: {counts.gold_covered_batches}/"
              f"{counts.gold_text_assessable_batches} gold-assessable batches "
              f"covered, {counts.gold_missing_batches} missing "
              f"({counts.windowed_batches} windowed)")
        if coverage.unjudged_run_batches:
            print(f"         {len(coverage.unjudged_run_batches)} batch(es) the "
                  "key says nothing about: "
                  f"{', '.join(coverage.unjudged_run_batches[:4])}")
    print(f"cost: {telemetry.summary()}")
    diagnostics = benchmark_diagnostics(results)
    diagnostic_report = format_benchmark_diagnostics(diagnostics)
    if diagnostic_report:
        print(diagnostic_report)

    if args.html:
        from react_review.report import render_eval_report
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(render_eval_report(metrics, results), encoding="utf-8")
        print(f"[html] {args.html.resolve()}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        run_meta = {
            "benchmark": str(benchmark),
            "benchmark_id": str(manifest.get("benchmark_id") or benchmark.name),
            "benchmark_gate": gate,
            **(profile.provenance() if profile is not None else {}),
            # Present only when this run resolved one. A run that never
            # aggregated must not name an evaluator, or it attributes its
            # answers to code that never ran.
            **({"aggregation_runtime": run_meta_runtime} if run_meta_runtime
               else {}),
            # What decided MATCH. A separate identity from what decided a
            # total, and recorded separately, because a comparator change that
            # rode on an aggregation version is the hole this closes.
            **({"compare_runtime": _compare_runtime} if _compare_runtime else {}),
            **({"adequacy_runtime": run_meta_adequacy}
               if run_meta_adequacy else {}),
            **({"excerpt_coverage": coverage.as_dict()}
               if coverage is not None else {}),
            "studies_file": str(studies_path.resolve()),
            "research_context": context,
            "extraction_mode": args.extraction,
            "extraction_cache": str(extraction_cache_path.resolve())
                                if extraction_cache is not None else "",
            "extraction_model_id": (getattr(backend, "model_id", "")
                                    or getattr(extraction_cache, "model_id", "")),
            "extraction_cache_hits": (extraction_cache.hits
                                      if extraction_cache is not None else 0),
            "extraction_cache_misses": (extraction_cache.misses
                                        if extraction_cache is not None else 0),
            "semantic_mode": args.semantic,
            "semantic_cache": str(semantic_cache_path.resolve())
                              if cache is not None else "",
            "semantic_model_id": (getattr(backend, "model_id", "")
                                  or getattr(cache, "model_id", "")),
            "semantic_cache_hits": cache.hits if cache is not None else 0,
            "semantic_cache_misses": cache.misses if cache is not None else 0,
            "semantic_cache_entries": len(cache) if cache is not None else 0,
            "telemetry": telemetry.model_dump(mode="json"),
        }
        args.out.write_text(
            json.dumps({"run": run_meta, "metrics": metrics,
                        "benchmark_diagnostics": diagnostics,
                        "rows": [row_payload(r) for r in results],
                        # One entry per READING, not per row. Every batched row
                        # names an execution id, and this is the only place that
                        # reference resolves. Absent entirely when nothing
                        # batched, so a legacy report is unchanged.
                        **({"batches": [b.model_dump(mode="json")
                                        for b in getattr(results, "batch_readings", [])]}
                           if getattr(results, "batch_readings", None) else {})},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] {args.out.resolve()}")


if __name__ == "__main__":
    main()
