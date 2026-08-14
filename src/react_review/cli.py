"""Command-line entry point for react_review.

Usage:
    # Run by auto-parsing a systematic review PDF (recommended):
    python -m react_review --config config.yaml --pdf path/to/review.pdf

    # Run with a hand-written student input YAML file:
    python -m react_review --config config.yaml --input data/samples/demo_input.yaml

    # Run with built-in demo data:
    python -m react_review --config config.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from react_review.core.config import load_config
from react_review.core.logging import setup_logging
from react_review.pipeline.factory import create_pipeline
from react_review.pipeline.schemas import (
    DatabaseQuery,
    EvidenceFieldSchema,
    PipelineRunResult,
    SearchStrategy,
    StudentReviewInput,
)


# Old top-level keys that no longer exist in the schema. If a YAML file
# still uses them we abort with a migration message rather than silently
# dropping the data on Pydantic validation.
_DEPRECATED_TOP_LEVEL_KEYS = {
    "search_strategy_text": "search_strategy.raw_text (and search_strategy.extracted_per_database)",
    "search_database": "search_strategy.extracted_per_database[].database",
    "reported_result_count": "search_strategy.reported_total_count",
    "extraction_fields": "evidence_schema (richer per-field schema; see docs)",
}


def _load_student_input(path: Path) -> StudentReviewInput:
    """Load and strictly validate student review input from a YAML file.

    Per project decision #2, YAML inputs are required to use the new rich
    schema directly — there is no LLM fallback for YAML inputs. If a file
    uses the old flat keys (``search_strategy_text``, ``extraction_fields``),
    we raise a clear migration error pointing to the new field names.

    Args:
        path: Path to the YAML input file.

    Returns:
        Validated StudentReviewInput.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    # Detect legacy-format YAMLs and refuse to silently drop their data.
    deprecated_used = [k for k in _DEPRECATED_TOP_LEVEL_KEYS if k in data]
    if deprecated_used:
        lines = [f"  - '{k}' → use '{_DEPRECATED_TOP_LEVEL_KEYS[k]}'" for k in deprecated_used]
        raise ValueError(
            "YAML input uses deprecated keys. The schema has changed; please "
            "migrate to the new structure (see data/samples/demo_input.yaml).\n"
            "Affected keys:\n" + "\n".join(lines)
        )

    return StudentReviewInput(**data)


def _build_demo_input() -> StudentReviewInput:
    """Build a built-in demo input with real DOIs for quick testing."""
    from react_review.steps.data_extraction.schemas import ExtractedField, ExtractedTable
    from react_review.steps.paper_verification.schemas import ReferenceEntry

    selected = [
        ReferenceEntry(
            title="Nivolumab versus Docetaxel in Advanced Nonsquamous Non-Small-Cell Lung Cancer",
            authors=["Borghaei H", "Paz-Ares L", "Horn L"],
            journal="New England Journal of Medicine",
            year=2015,
            doi="10.1056/NEJMoa1507643",
        ),
        ReferenceEntry(
            title="Pembrolizumab versus Chemotherapy for PD-L1-Positive Non-Small-Cell Lung Cancer",
            authors=["Reck M", "Rodriguez-Abreu D", "Robinson AG"],
            journal="New England Journal of Medicine",
            year=2016,
            doi="10.1056/NEJMoa1606774",
        ),
    ]

    submitted_tables = [
        ExtractedTable(
            paper_id="10.1056/NEJMoa1507643",
            fields=[
                ExtractedField(field_name="sample_size", value=582, confidence=1.0),
                ExtractedField(field_name="study_design", value="Phase III RCT", confidence=1.0),
                ExtractedField(field_name="intervention", value="Nivolumab 3mg/kg", confidence=1.0),
                ExtractedField(field_name="primary_outcome", value="Overall survival", confidence=1.0),
                ExtractedField(field_name="overall_survival_hr", value=0.73, confidence=1.0),
                ExtractedField(field_name="p_value", value=0.002, confidence=1.0),
            ],
            extractor_id="student",
        ),
    ]

    # Build the structured search strategy (replaces flat search_strategy_text).
    search_strategy = SearchStrategy(
        raw_text=(
            "We searched PubMed for studies of immunotherapy in advanced "
            "non-small cell lung cancer published between 2014 and 2017."
        ),
        extracted_per_database=[
            DatabaseQuery(
                database="PubMed",
                query="immunotherapy AND non-small cell lung cancer AND systematic review",
                source="verbatim",
            ),
        ],
        reported_total_count=150,
        reported_per_db_count={"PubMed": 150},
    )

    # Rich evidence schema — one entry per column in the submitted tables.
    evidence_schema = [
        EvidenceFieldSchema(
            student_field_name="sample_size",
            canonical_concept="sample_size",
            type="numeric",
            threshold_match=0.0,
            threshold_partial=0.05,
        ),
        EvidenceFieldSchema(
            student_field_name="study_design",
            canonical_concept="study_design",
            type="categorical",
            threshold_match=0.95,
            threshold_partial=0.75,
            synonym_check=False,
        ),
        EvidenceFieldSchema(
            student_field_name="intervention",
            canonical_concept="intervention",
            type="text",
            threshold_match=0.90,
            threshold_partial=0.70,
        ),
        EvidenceFieldSchema(
            student_field_name="primary_outcome",
            canonical_concept="outcome",
            type="text",
            threshold_match=0.90,
            threshold_partial=0.70,
        ),
        EvidenceFieldSchema(
            student_field_name="overall_survival_hr",
            canonical_concept="hazard_ratio",
            type="numeric",
            threshold_match=0.05,
            threshold_partial=0.15,
        ),
        EvidenceFieldSchema(
            student_field_name="p_value",
            canonical_concept="p_value",
            type="numeric",
            threshold_match=0.001,
            threshold_partial=0.01,
        ),
    ]

    return StudentReviewInput(
        student_id="demo-student-001",
        review_title="Immunotherapy in NSCLC: A Systematic Review",
        research_context="Effect of immune checkpoint inhibitors on survival in advanced NSCLC.",
        search_strategy=search_strategy,
        selected_papers=selected,
        evidence_schema=evidence_schema,
        submitted_tables=submitted_tables,
    )


# Printing that survives a console whose encoding can't carry every character
# (Windows GBK + emoji). The implementation lives with the other rendering
# helpers; re-exported here because every subcommand already calls this name.
from react_review.hitl.render import safe_print as _safe_print  # noqa: E402


def _aggregation_runtime(contract):
    """The policy and the identity that cleared it, bound — or nothing.

    Nothing when the contract does not batch. Recording an identity a run never
    used would attribute its answers to code that never ran, and resolving one
    costs a git subprocess that a non-batching run has no reason to pay.
    """
    if contract is None or not getattr(contract, "batching", False):
        return None
    from react_review.tools.safe_aggregation import AggregationRuntime

    return AggregationRuntime.resolve(
        policy_id=contract.aggregation_policy_id,
        evaluator_version=contract.evaluator_version)


def _print_result(result: PipelineRunResult) -> None:
    """Print a human-readable summary of the pipeline result."""
    si = result.student_input
    _safe_print("\n" + "=" * 70)
    _safe_print("  LIT-INSPECTOR PIPELINE RESULT")
    _safe_print("=" * 70)
    _safe_print(f"  Run ID:      {result.run_id}")
    _safe_print(f"  Student:     {si.student_id}")
    _safe_print(f"  Review:      {si.review_title}")
    _safe_print(f"  Selected:    {len(si.selected_papers)} papers")
    _safe_print(f"  Started:     {result.started_at}")
    _safe_print(f"  Completed:   {result.completed_at}")
    _safe_print(f"  Total Flags: {len(result.all_flags)}")
    _safe_print("-" * 70)

    # Step 0 — Ingestion summary (research context + schema sizes)
    if si.research_context:
        _safe_print("\n[Step 0] Review Understood")
        _safe_print(f"  Context:    {si.research_context}")
        _safe_print(f"  Schema:     {len(si.evidence_schema)} evidence fields")
        per_db = si.search_strategy.extracted_per_database
        if per_db:
            _safe_print(f"  Databases:  {', '.join(d.database for d in per_db)}")

    # Step 1 — multi-DB identification count comparison (informational)
    if result.multi_db_identification_check:
        check = result.multi_db_identification_check
        _safe_print("\n[Step 1] Identification count check (informational)")
        for row in check.per_database:
            stu = row.student_reported if row.student_reported is not None else "—"
            ai_ = row.ai_reproduced if row.ai_reproduced is not None else "—"
            _safe_print(f"  [{row.verdict}] {row.database}: student={stu}, ai={ai_}")

    # Step 1 — single-DB legacy search reproducibility
    if result.search_result:
        sr = result.search_result
        _safe_print("\n[Step 1b] Search Validation")
        _safe_print(f"  Query:        {sr.reconstructed_query[:80]}")
        _safe_print(f"  Reported:     {sr.reported_count} results")
        _safe_print(f"  Actual:       {sr.actual_count} results")
        _safe_print(f"  Reproducible: {sr.is_reproducible}")

        if result.papers_in_search:
            _safe_print("\n  Selected papers in search results:")
            for title, found in result.papers_in_search.items():
                icon = "[Y]" if found else "[N]"
                _safe_print(f"    {icon} {title[:60]}")

    # Step 2
    if result.verification_results:
        _safe_print(f"\n[Step 2] Paper Verification ({len(result.verification_results)} papers)")
        for vr in result.verification_results:
            icon = {"verified": "[OK]", "not_found": "[!!]", "uncertain": "[??]", "access_restricted": "[LK]"}.get(
                vr.status.value, "[--]"
            )
            _safe_print(f"  {icon} {vr.reference.title[:55]}...")
            _safe_print(f"     Status: {vr.status.value} | Confidence: {vr.confidence:.0%}")
            if vr.matched_metadata:
                cr_title = vr.matched_metadata.get("title", "")
                if cr_title:
                    _safe_print(f"     CrossRef: {cr_title[:55]}...")

    # Step 3
    if result.extracted_tables:
        _safe_print(f"\n[Step 3] Data Extraction ({len(result.extracted_tables)} tables)")
        for et in result.extracted_tables:
            _safe_print(f"  [T] {et.paper_id}")
            _safe_print(f"     Extractor: {et.extractor_id} | Fields: {len(et.fields)}")
            for f in et.fields[:3]:  # show first 3 fields
                val_str = str(f.value)[:30] if f.value is not None else "null"
                _safe_print(f"       - {f.field_name}: {val_str}")
            if len(et.fields) > 3:
                _safe_print(f"       ... and {len(et.fields) - 3} more fields")

    # Step 4
    if result.report:
        _safe_print("\n[Step 4] Evaluation Report")
        _safe_print(f"  {result.report.summary}")
        if result.comparison_results:
            for cr in result.comparison_results:
                icon = "[OK]" if cr.agreement_rate >= 0.7 else "[!!]"
                _safe_print(f"\n  {icon} Paper {cr.paper_id}: agreement {cr.agreement_rate:.0%}")
                # Show field-level diffs
                for d in cr.field_diffs:
                    mark = "  " if d.is_consistent else ">>"
                    s_val = str(d.student_value)[:25] if d.student_value is not None else "null"
                    m_vals = [str(v)[:25] if v is not None else "null" for v in d.model_values[:2]]
                    m_str = ", ".join(m_vals) if m_vals else "N/A"
                    _safe_print(f"    {mark} {d.field_name}: student={s_val} | ai={m_str}")

    # Flags
    if result.all_flags:
        _safe_print(f"\n{'-' * 70}")
        _safe_print(f"All Flags ({len(result.all_flags)}):")
        for flag in result.all_flags:
            icon = {"error": "[ERR]", "warning": "[WRN]", "info": "[INF]"}.get(flag.severity.value, "[---]")
            _safe_print(f"  {icon} [{flag.step.value}] {flag.message}")

    _safe_print("\n" + "=" * 70)


def _legacy_main(argv: list[str] | None = None) -> None:
    """Legacy 4-step prototype pipeline (reachable via ``react-review legacy``)."""
    parser = argparse.ArgumentParser(
        prog="react-review legacy",
        description="Legacy lit_inspector 4-step pipeline (prototype).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to student input YAML file. If omitted, uses built-in demo data.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to a systematic review PDF. If provided, the LLM will auto-extract "
             "search strategy, selected papers, and extraction tables from it.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON instead of summary.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Generate a DOCX evaluation report at the given path "
             "(e.g. --report output/report.docx).",
    )
    args = parser.parse_args(argv)

    # Load config
    config = load_config(args.config)
    setup_logging(log_file=config.paths.log_file)

    # Load student input — three possible sources, priority: --pdf > --input > demo
    if args.pdf:
        print(f"Parsing PDF with LLM: {args.pdf}")
        from react_review.steps.pdf_parsing import PDFParser
        from react_review.pipeline.factory import _create_llm_backend
        llm_backend = _create_llm_backend(config)
        pdf_parser = PDFParser(llm_backend=llm_backend)
        student_input = asyncio.run(pdf_parser.parse(args.pdf))
    elif args.input:
        print(f"Loading student input from: {args.input}")
        student_input = _load_student_input(args.input)
    else:
        print("Using built-in demo data (use --input or --pdf to provide your own)")
        student_input = _build_demo_input()

    print(f"Student: {student_input.student_id}")
    print(f"Review:  {student_input.review_title}")
    print(f"Papers:  {len(student_input.selected_papers)}")
    print(f"Tables:  {len(student_input.submitted_tables)}")
    print()

    # Create pipeline and run
    pipeline = create_pipeline(config)
    result = asyncio.run(pipeline.run(student_input))

    # Output
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _print_result(result)

    # Generate DOCX report if requested (or auto-generate when --pdf is used)
    report_path = args.report
    if report_path is None and args.pdf:
        # Auto-generate report next to the PDF
        report_path = Path("output") / f"report_{result.run_id}.docx"

    if report_path:
        from react_review.steps.reporting import generate_docx_report
        report_path = generate_docx_report(result, report_path)
        _safe_print(f"\n[REPORT] Saved to: {report_path.resolve()}")


def _audit_main(argv: list[str] | None = None) -> None:
    """New deterministic audit: match review↔source, compare, report, persist.

    Runs the P1 AuditOrchestrator over a review table and a source table (CSV),
    prints the audit report, and saves the full EvidencePackage under --out.
    No LLM / network — this is the deterministic Tier-3 path.
    """
    import uuid

    from react_review.audit import ToleranceTable
    from react_review.core.config import AppConfig
    from react_review.csv_io import load_review_items, load_source_items
    from react_review.orchestrator import AuditOrchestrator, Judge
    from react_review.schemas.package import EvidencePackage
    from react_review.store import EvidencePackageStore
    from react_review.tools import build_catalogue

    parser = argparse.ArgumentParser(
        prog="react-review audit",
        description="Audit a review's reported values against source evidence.",
    )
    parser.add_argument("review_csv", type=Path,
                        help="review-side long table (study_id, group, field_type, value, unit)")
    parser.add_argument("source_csv", type=Path,
                        help="source-side evidence (study_id, group, field_type, source_value, source_unit)")
    parser.add_argument("--out", type=Path, default=Path("output/runs"),
                        help="base dir for the run's evidence package (default: output/runs)")
    parser.add_argument("--tolerances", type=Path, default=None,
                        help="tolerances.yaml (default: shipped configs/tolerances.yaml)")
    parser.add_argument("--run-id", default=None, help="explicit run id (default: random)")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args(argv)

    setup_logging()

    review = load_review_items(args.review_csv)
    source = load_source_items(args.source_csv)
    _safe_print(f"Loaded {len(review)} review rows, {len(source)} source rows.")

    tol = ToleranceTable.from_yaml(args.tolerances) if args.tolerances else None
    catalogue = build_catalogue(AppConfig(mock_mode=True), tolerance=tol)
    orch = AuditOrchestrator(catalogue)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    report = asyncio.run(orch.run(review, source, run_id=run_id))
    final = Judge().adjudicate(report, source)

    store = EvidencePackageStore(args.out)
    pkg_path = store.save(EvidencePackage(
        run_id=run_id, review_items=review, source_items=source, report=report,
        final_verification=final,
    ))

    if args.json:
        _safe_print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    else:
        _safe_print("\n" + report.summary)
        for r in report.results:
            if r.label.value != "match":
                claim_id = r.audit_id or "-"
                _safe_print(
                    f"  [{claim_id}] [{r.label.value}] "
                    f"{r.study_id}/{r.group}/{r.field_type}: {r.reason}")
    _safe_print(f"\n[PACKAGE] {pkg_path.resolve()}")


def _render_saved_package_html(store, run_id: str, out: Path | None = None) -> Path:
    """Load the final package from disk, then atomically render its HTML.

    Loading is deliberate: a run report must describe the exact serialised
    artifact a reviewer can reopen, not a richer or subtly different in-memory
    object.  If saving/loading/rendering fails, no report is presented as a
    successful companion to a package that was never persisted.
    """
    import os

    from react_review.report import render_html_report

    package_path = store.package_path(run_id)
    if not package_path.is_file():
        raise FileNotFoundError(
            f"cannot render report before Evidence Package is saved: {package_path}")
    package = store.load(run_id)
    html = render_html_report(package)
    report_path = out or (store.run_dir(run_id) / "report.html")
    if report_path.resolve() == package_path.resolve():
        raise ValueError("HTML report path must not overwrite package.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    try:
        tmp.write_text(html, encoding="utf-8")
        os.replace(tmp, report_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return report_path


def run_parser() -> argparse.ArgumentParser:
    """The `run` command line, as its own object.

    Built here rather than inside the command so the contract and
    execution flags can be checked without starting an audit — the
    defaults are part of the contract story and deserve a test.
    """
    ap = argparse.ArgumentParser(
        prog="react-review run",
        description="Full audit: review PDF → Collector → Auditor → Judge.",
    )
    ap.add_argument("--pdf", type=Path, required=True, help="the systematic review PDF")
    ap.add_argument("--studies", type=Path, default=None,
                    help="included_studies.csv (study_id, doi, source_pdf) for LOCAL source "
                         "PDFs. Omit for ONLINE mode: references come from the review's own "
                         "reference list and full text is fetched online (DOIs reconciled).")
    ap.add_argument("--config", type=Path, default=Path("configs/config.local.yaml"),
                    help="LLM config with an api_key (default: configs/config.local.yaml)")
    ap.add_argument("--pdf-dir", type=Path, default=None,
                    help="base dir for source_pdf paths (default: the --studies parent)")
    ap.add_argument("--tolerances", type=Path, default=None)
    ap.add_argument("--profile", type=Path, default=None,
                    help="run contract profile: which prompt contracts, "
                         "tolerances and policies decide the answer "
                         "(default: configs/run_profiles/legacy.json)")
    ap.add_argument("--extraction", choices=("live", "record", "replay"),
                    default="live",
                    help="live calls the model; record also saves the raw "
                         "responses so the run can be replayed offline; replay "
                         "makes no extraction calls")
    ap.add_argument("--extraction-cache", type=Path, default=None,
                    help="raw extraction recording used by record/replay "
                         "(default: <out>/<run-id>/extraction_cache.json)")
    ap.add_argument("--context", default="", help="one-sentence research context")
    ap.add_argument("--out", type=Path, default=Path("output/runs"))
    ap.add_argument("--html", type=Path, default=None,
                    help="HTML report path (default: <out>/<run-id>/report.html); "
                         "rendered only after package.json has been saved")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N review items (0 = all)")
    ap.add_argument("--checkpoints", choices=("key", "all", "none"), default="key",
                    help="how much to pause: key = gate each stage (default), "
                         "all = also gate every source paper, none = never pause")
    ap.add_argument("--non-interactive", action="store_true",
                    help="never pause, even on a terminal (the journal is still written)")
    ap.add_argument("--allow-skip", action="store_true",
                    help="offer 'skip the remaining checkpoints' at a prompt")
    ap.add_argument("--journal-dir", type=Path, default=None,
                    help="where to write per-step artifacts (default: <out>/<run_id>)")
    ap.add_argument("--tables", default="",
                    help="only process these captured tables, e.g. table_1,table_2 "
                         "(default: all of them; the checkpoint can also drop one)")
    ap.add_argument("--drop-tables", default="",
                    help="exclude these captured tables, e.g. table_s1")
    checklist_group = ap.add_mutually_exclusive_group()
    checklist_group.add_argument(
        "--checklist", type=Path, default=None,
        help="clinician-editable coverage checklist YAML "
             "(default: configs/checklists/default.yaml)")
    checklist_group.add_argument(
        "--no-checklist", action="store_true",
        help="disable the domain checklist for this run")
    ap.add_argument("--semantic", choices=("off", "cache-only", "on"), default="on",
                    help="judge text values the numeric comparison cannot read: "
                         "on = ask the model (costs tokens), cache-only = replay a "
                         "recording and fail on a miss, off = deterministic only")
    ap.add_argument("--semantic-cache", type=Path, default=None,
                    help="file of recorded judgements to reuse and extend "
                         "(default: <out>/<run_id>/semantic_cache.json)")
    return ap


def _run_main(argv: list[str] | None = None, *, dependencies=None) -> None:
    """Full LLM pipeline: review PDF → Parser → Collector → Auditor → Judge.

    Parses the review PDF, resolves each study to the included-studies registry,
    reads the LOCAL source PDFs, extracts + audits + adjudicates, persists the
    EvidencePackage, then reloads it to render HTML. Requires ``--config`` with
    an LLM api_key (GLM etc.).
    """
    import sys
    import uuid

    from react_review.checklist import Checklist
    from react_review.core.exceptions import RunStopped
    from react_review.dkb import FieldResolver, load_runtime_knowledge
    from react_review.hitl import (
        AutoContinue,
        CheckpointPolicy,
        ConsoleCheckpoint,
        RunJournal,
        StepReporter,
    )
    from react_review.parser.review_parser import ReviewParser
    from react_review.production import ProductionDependencies
    from react_review.store import EvidencePackageStore

    ap = run_parser()
    args = ap.parse_args(argv)

    # Prefer real UTF-8 output; safe_print still covers consoles that refuse.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                          # noqa: BLE001
        pass

    config = load_config(args.config)
    setup_logging(log_file=config.paths.log_file)
    # The model and the paper supply are the only things this entry point
    # reaches outside itself for, and the only things a test may substitute.
    # Everything below is built here, in production, by the code under test.
    dependencies = dependencies or ProductionDependencies()
    backend = dependencies.llm(config)

    # The run id is needed BEFORE parsing: the parser's first checkpoint already
    # writes into this run's journal directory.
    run_id = args.run_id or uuid.uuid4().hex[:12]
    journal = RunJournal(args.journal_dir or (args.out / run_id))
    interactive = sys.stdin.isatty() and not args.non_interactive
    policy = CheckpointPolicy.from_name(args.checkpoints if interactive else "none")
    gate = dependencies.checkpoint(
        lambda: (ConsoleCheckpoint(policy, journal=journal,
                                   allow_skip=args.allow_skip)
                 if interactive else AutoContinue()))
    reporter = StepReporter(run_id, gate=gate, journal=journal)
    store = EvidencePackageStore(args.out)
    if not interactive:
        _safe_print("[unattended] no one is gating this run; every step is still "
                    f"journalled to {journal.run_dir}")

    # The contract is resolved before anything it governs is built. It decides
    # the routes, and the routes decide whether this run has more than one stage
    # to measure — so a parser built before it could not be labelled, which is
    # why the review parser's model calls went uncounted.
    from react_review.contracts import repo_root as _repo_root
    from react_review.production import (
        ProductionBackends,
        ProductionSession,
        ProductionStages,
    )
    from react_review.run_profile import (
        ExecutionMode,
        RunManifest,
        guard_contract_overrides,
        load_run_contract,
    )
    from react_review.schemas.telemetry import RunTelemetry

    contract = load_run_contract(
        args.profile or (_repo_root() / "configs" / "run_profiles" / "legacy.json"))
    guard_contract_overrides(contract, {"--tolerances": args.tolerances})
    # The modes are decided by the flags, not by anything the parser finds, so
    # they are settled here — where an interrupt during parsing can still be
    # recorded under the identity the run was actually using.
    execution = ExecutionMode(
        extraction_mode=args.extraction,
        extraction_cache=(args.extraction_cache
                          or (args.out / run_id / "extraction_cache.json")
                          if args.extraction != "live" else None),
        semantic_mode=args.semantic,
        semantic_cache=(args.semantic_cache or (args.out / run_id / "semantic_cache.json")
                        if args.semantic != "off" else None),
    ).validate_modes()

    # Created before the FIRST model call of the run, not before the extraction.
    telemetry = RunTelemetry()
    stages = ProductionStages.of(contract)
    backends = ProductionBackends(backend, telemetry, stages)

    project_root = Path(__file__).resolve().parents[2]
    seed = project_root / "configs" / "knowledge.seed.json"
    kb = load_runtime_knowledge(seed, project_root / "configs" / "ontology")
    checklist = None
    if not args.no_checklist:
        checklist = Checklist.from_yaml(
            args.checklist or project_root / "configs" / "checklists" / "default.yaml")
    # audit mode: KB is read-only — candidates become proposals, not KB writes.
    resolver = FieldResolver(kb, backend=backends.parsing, write_back=False)
    review_parser = ReviewParser(
        backends.parsing, resolver, reporter=reporter,
        keep_tables=_id_set(args.tables), drop_tables=_id_set(args.drop_tables),
        checklist=checklist,
        table_capture_prompt_profile=(
            contract.table_capture_prompt_profile or "table_capture_v1"),
    )

    # Every way this run can end goes through one object, so all four leave the
    # same three things behind: an artifact saying how it ended, telemetry for
    # the whole execution, and a saved cache.
    session = ProductionSession(store, run_id, telemetry=telemetry,
                                execution=execution, emit=_safe_print)
    # Provisional, so a run interrupted during PARSING still leaves an artifact
    # that names its contract and modes. `context_source` is not knowable yet —
    # it depends on what the review turns out to contain — so it says what is
    # true before parsing, and _run_audit replaces this with the settled one.
    session.manifest = RunManifest.of(
        contract, execution,
        context_source=("cli" if args.context else "default"),
        inputs={"review_pdf": str(args.pdf.name),
                "studies": str(args.studies.name) if args.studies else ""})
    try:
        # The clock covers EXECUTION — parsing and the audit, which is where a
        # real run spends its time and where `wall_seconds` was left at zero.
        # It stops at finalisation, before the report is rendered: rendering a
        # finished package is not part of what the audit cost.
        with session:
            return _run_audit(args, config, backends, kb, resolver, review_parser,
                              reporter, store, run_id, contract=contract,
                              telemetry=telemetry, stages=stages, session=session,
                              execution=execution, dependencies=dependencies)
    except RunStopped as exc:
        session.finalise_stopped(stage=exc.stage, reason=exc.reason)
        raise SystemExit(2)
    except KeyboardInterrupt:
        session.finalise_interrupted()
        raise SystemExit(130)
    except SystemExit:
        raise
    except BaseException as exc:                                   # noqa: BLE001
        # A crash is an outcome, and an artifact that still says `in_progress`
        # claims the run is going. If the package was already finalised, this
        # changes nothing — a report that fails to render must not restate a
        # validated result as a failed audit.
        session.finalise_error(exc)
        raise


def _id_set(raw: str) -> set[str] | None:
    """Parse a comma-separated id list; None when the flag was not used."""
    ids = {part.strip() for part in (raw or "").split(",") if part.strip()}
    return ids or None


def _run_audit(args, config, backends, kb, resolver, review_parser,
               reporter, store, run_id: str, *, contract, telemetry,
               stages, session, execution, dependencies) -> None:
    """The audit itself, wrapped so a stop/interrupt can finalise cleanly."""
    from react_review.agents.collector import Collector
    from react_review.audit import ToleranceTable
    from react_review.audit.semantic_cache import SemanticCache
    from react_review.audit.semantic_control import (
        format_threshold_sensitivity,
        threshold_sensitivity,
    )
    from react_review.csv_io import load_included_studies
    from react_review.orchestrator import AuditOrchestrator, AuditPipeline, Judge
    from react_review.retrieval.local_pdf import LocalPdfRetriever
    from react_review.steps.paper_verification.fulltext_retriever import FullTextRetriever
    from react_review.study_match import (
        apply_modality_disambiguation,
        build_reference_resolver,
        build_reference_resolver_from_parsed,
        resolve_studies,
    )
    from react_review.tools.compare import CompareValuesTool
    from react_review.tools.semantic_compare import SemanticCompareTool
    from react_review.tools.extract import FetchFullTextTool
    from react_review.production import aggregation_runtime, build_collector
    from react_review.tools.extract_batch import ExtractSourceBatchTool
    from react_review.tools.extract_source import ExtractSourceValueTool
    from react_review.contracts import repo_root
    from react_review.run_profile import RunManifest
    from react_review.tools.extraction_cache import ExtractionCache
    from react_review.tools.registry import ToolRegistry
    from react_review.tools.search import (
        CrossRefResolver,
        EuropePMCResolver,
        OpenAlexResolver,
        ReferenceReconciler,
        ResolveReferenceTool,
    )

    _safe_print(f"Parsing review PDF: {args.pdf}")
    parsed = asyncio.run(review_parser.parse(args.pdf, research_context=args.context))
    _safe_print(f"  parsed {len(parsed.items)} review items")

    # Whose words describe this review. The parser reads one out of the review
    # itself, which was being extracted and then dropped; whether the audit may
    # use it is a CONTRACT decision, because the context reaches the prompt and
    # therefore the cache key — a silent fallback would change the question
    # without changing the profile that is supposed to define it.
    research_context, context_source = args.context, "cli"
    if not research_context:
        if contract.context_policy == "cli_then_parsed" and parsed.research_context:
            research_context, context_source = parsed.research_context, "parsed"
            _safe_print(f"  context (from the review): {research_context}")
        else:
            context_source = "default"

    # References + retriever: LOCAL (included_studies.csv → local source PDFs) or
    # ONLINE (references from the review's own reference list → online full text).
    if args.studies:
        studies = load_included_studies(args.studies)
        review_items, sid_map = resolve_studies(parsed.items, studies)
        review_items = apply_modality_disambiguation(
            review_items, sid_map, kb, parsed.field_resolutions)
        base_dir = args.pdf_dir or args.studies.parent
        doi_to_path = {s.doi: s.source_pdf for s in studies if s.doi and s.source_pdf}
        retriever = dependencies.papers(
            lambda: LocalPdfRetriever(doi_to_path, base_dir=base_dir))
        reference_resolver = build_reference_resolver(sid_map)
    else:
        _safe_print(f"  online mode: references from the review's {len(parsed.studies)} "
                    "extracted citations; full text fetched online")
        review_items = parsed.items
        retriever = dependencies.papers(lambda: FullTextRetriever(
            pubmed_settings=config.pubmed,
            unpaywall_email=config.unpaywall.email or config.pubmed.email,
            unpaywall_enabled=config.unpaywall.enabled,
        ))
        reference_resolver = build_reference_resolver_from_parsed(parsed.studies)
    if args.limit:
        review_items = review_items[: args.limit]

    if contract.tolerances_path is not None:
        tol = ToleranceTable.from_yaml(contract.tolerances_path)
    else:
        tol = ToleranceTable.from_yaml(args.tolerances) if args.tolerances else ToleranceTable()
    mailto = config.unpaywall.email or config.pubmed.email or config.crossref.mailto
    reconciler = ReferenceReconciler([
        CrossRefResolver(base_url=config.crossref.base_url, mailto=mailto,
                         timeout=config.crossref.timeout),
        OpenAlexResolver(mailto=mailto, timeout=config.crossref.timeout),
        EuropePMCResolver(timeout=config.crossref.timeout),
    ])
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(retriever))
    extraction_cache = (None if execution.extraction_mode == "live"
                        else ExtractionCache(execution.extraction_cache))
    # A production run should be able to say what it spent. Until now only the
    # eval harness could, so "what did batching cost" was answerable about the
    # benchmark and not about a real audit. Three labelled views of one backend,
    # exactly as the harness builds them.
    reg.register(ExtractSourceValueTool(
        backends.single, cache=extraction_cache,
        cache_mode=execution.extraction_mode, telemetry=telemetry,
        stage=stages.single))
    reg.register(ExtractSourceBatchTool(
        backends.batch, cache=extraction_cache,
        cache_mode=execution.extraction_mode, telemetry=telemetry))
    reg.register(ResolveReferenceTool(reconciler))          # no-DOI refs → gated online DOI
    # Text the numeric comparison cannot read ("ICU" vs "intensive care unit")
    # goes to the model, and its answer then goes through the deterministic
    # controls. Every judgement is recorded so the run can be replayed offline.
    semantic_cache = (SemanticCache(execution.semantic_cache)
                      if execution.semantic_cache is not None else None)
    # Handed to the session as they are built, so an interrupt at any point
    # after this line still writes the totals and saves the judgements.
    session.extraction_cache, session.semantic_cache = extraction_cache, semantic_cache
    if semantic_cache is not None:
        semantic_cache.measure_into(telemetry, stages.semantic)
    reg.register(CompareValuesTool(
        tol,
        semantic=(SemanticCompareTool(
            backends.semantic, profile=contract.semantic_prompt_profile)
            if args.semantic == "on" else None),
        semantic_mode=args.semantic, semantic_cache=semantic_cache,
        min_confidence=tol.semantic_min_confidence,
        semantic_profile=contract.semantic_prompt_profile))
    runtime = aggregation_runtime(contract)
    manifest = RunManifest.of(
        contract, execution, context_source=context_source,
        inputs={"review_pdf": str(args.pdf.name),
                "studies": str(args.studies.name) if args.studies else ""})
    # What decided, recorded beside what it decided under — including on the
    # partial manifest a stopped run leaves behind.
    manifest.aggregation_runtime = RunManifest.runtime_of(runtime)
    # And what decided MATCH, which is a different identity from what decided a
    # total. Empty when the contract names no comparator version, so nothing
    # already recorded gains a key.
    manifest.compare_runtime = RunManifest.compare_of(contract)
    # So a run interrupted before the first paper still leaves an artifact that
    # says which contract it was running under.
    session.manifest = manifest
    _safe_print(f"Contract:   {contract.profile_id} "
                f"[extraction={contract.extraction_profile} "
                f"semantic={contract.semantic_prompt_profile} "
                f"scope={contract.scope_policy} context={context_source}]")
    _safe_print(f"Execution:  extraction={execution.extraction_mode} "
                f"semantic={execution.semantic_mode}")
    # The whole contract, not one field of it. A run may legitimately read
    # values in batch and arm identities one at a time, and the Collector can
    # only honour that if it holds the routes — handing it a single profile
    # would let a v2 contract be loaded, recorded, and then ignored.
    if runtime is not None:
        _safe_print(f"Aggregation: {runtime.evaluator.describe()}")
    pipeline = AuditPipeline(
        build_collector(reg, contract=contract, knowledge=kb,
                        cohorts=parsed.cohorts,
                        knowledge_fingerprint=parsed.knowledge_fingerprint,
                        telemetry=telemetry, runtime=runtime),
        AuditOrchestrator(reg), Judge(),
        store=store, reporter=reporter, run_manifest=manifest,
        telemetry=telemetry,
        # Per-study partials still come from the pipeline; the finished package
        # does not. It belongs to the session, which saves it once, after the
        # books are closed.
        owns_final_save=False,
    )

    _safe_print(f"Auditing {len(review_items)} items (run {run_id}) …")
    pkg = asyncio.run(pipeline.run(
        review_items, reference_resolver,
        research_context=research_context, run_id=run_id, parser_record=parsed.record,
        captured_tables=parsed.tables, cohorts=parsed.cohorts,
        field_resolutions=parsed.field_resolutions,
        knowledge_imports=parsed.knowledge_imports,
        knowledge_fingerprint=parsed.knowledge_fingerprint,
        knowledge_concept_count=parsed.knowledge_concept_count,
        checklist=parsed.checklist,
    ))

    # One save, by the object that owns it, after the clock has stopped and the
    # books are closed — and the reloaded file is what everything downstream
    # reads, so `run` and `report` cannot render different object states.
    pkg = session.finalise_success(pkg)
    _safe_print(f"Cost:       {telemetry.summary()}")

    pkg_path = store.package_path(run_id)
    _safe_print(f"\n[PACKAGE] {pkg_path.resolve()}")
    report_path = _render_saved_package_html(
        store, run_id, getattr(args, "html", None))
    _safe_print(f"[REPORT]  {report_path.resolve()}")

    fv = pkg.final_verification
    _safe_print("\n" + fv.summary)
    for f in fv.human_review_flags[:40]:
        _safe_print(
            f"  [{f.audit_id or '-'}] [{f.label}] "
            f"{f.study_id}/{f.group}/{f.field_type}: {f.reason}")
    if semantic_cache is not None:
        path = execution.semantic_cache
        _safe_print(
            f"[SEMANTIC] mode={args.semantic}  min_confidence={tol.semantic_min_confidence}"
            f"  {len(semantic_cache)} judgement(s), "
            f"{semantic_cache.hits} reused / {semantic_cache.misses} new"
            f"  ({semantic_cache.hit_rate:.0%} from cache)")
        vs = semantic_cache.verdicts()
        line = format_threshold_sensitivity(threshold_sensitivity(vs), len(vs))
        if line:
            _safe_print(f"           {line}")
        _safe_print(f"           replay this run offline with: "
                    f"--semantic cache-only --semantic-cache {path}")

    # The audit itself never writes the KB. It only COLLECTS candidate concepts as
    # proposals; a developer later curates them with `react-review learn`.
    if resolver.proposals:
        from react_review.dkb import save_proposals
        prop_path = args.out / run_id / "proposals.json"
        save_proposals(resolver.proposals, prop_path)
        _safe_print(f"\n[PROPOSALS] {len(resolver.proposals)} candidate concept(s) → {prop_path.resolve()}")
        _safe_print("            curate with:  react-review learn " + str(prop_path))


def _report_main(argv: list[str] | None = None) -> None:
    """Render a saved EvidencePackage into a standalone HTML report.

    react-review report RUN_ID [--runs output/runs] [--out report.html]
    """
    from react_review.store import EvidencePackageStore

    ap = argparse.ArgumentParser(prog="react-review report",
                                 description="Render an audit run to an HTML report.")
    ap.add_argument("run_id", help="the run id (folder under --runs)")
    ap.add_argument("--runs", type=Path, default=Path("output/runs"))
    ap.add_argument("--out", type=Path, default=None, help="output .html (default: <run>/report.html)")
    args = ap.parse_args(argv)

    out = _render_saved_package_html(
        EvidencePackageStore(args.runs), args.run_id, args.out)
    _safe_print(f"[report] {out.resolve()}")


def _learn_main(argv: list[str] | None = None) -> None:
    """Developer LEARN mode (DKB-4) — curate audit proposals into the KB.

    This is deliberately OUTSIDE the client audit path (which is read-only). It
    takes one or more proposals JSON files saved by ``react-review run`` — each
    file is treated as ONE run's batch — ingests them, and promotes a concept to
    authoritative only after repeated agreement across runs (``--threshold``) or
    an explicit ``--confirm``. The result is a new curated KB written to ``--out``.

        react-review learn output/runs/*/proposals.json \\
            --kb configs/knowledge.seed.json --out configs/knowledge.learned.json \\
            --threshold 3 --confirm hba1c
    """
    from react_review.dkb import KnowledgeBase, LearningSession, load_proposals

    ap = argparse.ArgumentParser(
        prog="react-review learn",
        description="Developer mode: curate audit proposals into trusted KB knowledge.",
    )
    ap.add_argument("proposals", type=Path, nargs="*",
                    help="proposals JSON file(s); each file counts as one run's batch")
    ap.add_argument("--kb", type=Path, default=None,
                    help="base KB to grow (default: shipped configs/knowledge.seed.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the curated KB here (default: overwrite --kb; requires --kb or this)")
    ap.add_argument("--threshold", type=int, default=3,
                    help="agreements ACROSS runs needed to auto-promote (default: 3)")
    ap.add_argument("--confirm", action="append", default=[], metavar="FIELD_TYPE",
                    help="force-promote a provisional concept (repeatable)")
    ap.add_argument("--list", action="store_true",
                    help="only list pending provisional concepts, then exit")
    args = ap.parse_args(argv)

    setup_logging()

    kb_path = args.kb or (Path(__file__).resolve().parents[2] / "configs" / "knowledge.seed.json")
    out_path = args.out or args.kb
    kb = KnowledgeBase.from_json(kb_path)
    session = LearningSession(kb, threshold=args.threshold)

    total = 0
    for pf in args.proposals:
        batch = load_proposals(pf)
        total += len(batch)
        promoted = session.ingest(batch)                 # one file == one run's agreement
        tag = f" → promoted {promoted}" if promoted else ""
        _safe_print(f"[ingest] {pf} : {len(batch)} proposal(s){tag}")

    for ft in args.confirm:
        ok = session.confirm(ft)
        _safe_print(f"[confirm] {ft} : {'promoted → authoritative' if ok else 'not a pending concept'}")

    pending = session.pending()
    _safe_print(f"\nPending provisional concepts ({len(pending)}):")
    for ft in pending:
        _safe_print(f"  - {ft}")

    if args.list:
        return
    if out_path is None:
        _safe_print("\n[skip] no --out and no --kb given; not writing. "
                    "Pass --out to persist the curated KB.")
        return
    session.save(out_path)
    _safe_print(f"\n[KB] curated knowledge base ({len(kb.entries)} concepts) → {out_path.resolve()}")
    _safe_print(f"     ingested {total} proposal(s) across {len(args.proposals)} run batch(es).")


def main() -> None:
    """CLI entry point. Subcommands: ``run`` / ``report`` / ``audit`` / ``learn`` / ``legacy``."""
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        return _run_main(argv[1:])
    if argv and argv[0] == "report":
        return _report_main(argv[1:])
    if argv and argv[0] == "audit":
        return _audit_main(argv[1:])
    if argv and argv[0] == "learn":
        return _learn_main(argv[1:])
    if argv and argv[0] == "legacy":
        return _legacy_main(argv[1:])
    # Back-compat: no recognised subcommand falls through to the legacy pipeline.
    return _legacy_main(argv)
