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


def _safe_print(text: str) -> None:
    """Print text safely, replacing unencodable characters (e.g. emoji on Windows GBK)."""
    import sys
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


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
                _safe_print(f"  [{r.label.value}] {r.study_id}/{r.group}/{r.field_type}: {r.reason}")
    _safe_print(f"\n[PACKAGE] {pkg_path.resolve()}")


def _run_main(argv: list[str] | None = None) -> None:
    """Full LLM pipeline: review PDF → Parser → Collector → Auditor → Judge.

    Parses the review PDF, resolves each study to the included-studies registry,
    reads the LOCAL source PDFs, extracts + audits + adjudicates, and persists the
    EvidencePackage. Requires ``--config`` with an LLM api_key (GLM etc.).
    """
    import uuid

    from react_review.agents.collector import Collector
    from react_review.audit import ToleranceTable
    from react_review.csv_io import load_included_studies
    from react_review.normalize.vocabulary import Vocabulary
    from react_review.orchestrator import AuditOrchestrator, AuditPipeline, Judge
    from react_review.parser.review_parser import ReviewParser
    from react_review.pipeline.factory import _create_llm_backend
    from react_review.retrieval.local_pdf import LocalPdfRetriever
    from react_review.store import EvidencePackageStore
    from react_review.study_match import build_reference_resolver, resolve_studies
    from react_review.tools.compare import CompareValuesTool
    from react_review.tools.extract import FetchFullTextTool
    from react_review.tools.extract_source import ExtractSourceValueTool
    from react_review.tools.normalize import NormalizeFieldTool
    from react_review.tools.registry import ToolRegistry

    ap = argparse.ArgumentParser(
        prog="react-review run",
        description="Full audit: review PDF → Collector → Auditor → Judge.",
    )
    ap.add_argument("--pdf", type=Path, required=True, help="the systematic review PDF")
    ap.add_argument("--studies", type=Path, required=True,
                    help="included_studies.csv (study_id, doi, source_pdf)")
    ap.add_argument("--config", type=Path, default=Path("configs/config.local.yaml"),
                    help="LLM config with an api_key (default: configs/config.local.yaml)")
    ap.add_argument("--pdf-dir", type=Path, default=None,
                    help="base dir for source_pdf paths (default: the --studies parent)")
    ap.add_argument("--tolerances", type=Path, default=None)
    ap.add_argument("--context", default="", help="one-sentence research context")
    ap.add_argument("--out", type=Path, default=Path("output/runs"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="audit only the first N review items (0 = all)")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    setup_logging(log_file=config.paths.log_file)
    backend = _create_llm_backend(config)

    seed = Path(__file__).resolve().parents[2] / "configs" / "vocabulary.seed.json"
    vocab = Vocabulary.from_json(seed)
    review_parser = ReviewParser(backend, NormalizeFieldTool(vocab, backend))

    _safe_print(f"Parsing review PDF: {args.pdf}")
    parsed = asyncio.run(review_parser.parse(args.pdf, research_context=args.context))
    _safe_print(f"  parsed {len(parsed.items)} review items")

    studies = load_included_studies(args.studies)
    review_items, sid_map = resolve_studies(parsed.items, studies)
    if args.limit:
        review_items = review_items[: args.limit]

    base_dir = args.pdf_dir or args.studies.parent
    doi_to_path = {s.doi: s.source_pdf for s in studies if s.doi and s.source_pdf}
    retriever = LocalPdfRetriever(doi_to_path, base_dir=base_dir)

    tol = ToleranceTable.from_yaml(args.tolerances) if args.tolerances else ToleranceTable()
    reg = ToolRegistry()
    reg.register(FetchFullTextTool(retriever))
    reg.register(ExtractSourceValueTool(backend))
    reg.register(CompareValuesTool(tol))
    pipeline = AuditPipeline(
        Collector(reg, vocabulary=vocab), AuditOrchestrator(reg), Judge(),
        store=EvidencePackageStore(args.out),
    )

    run_id = args.run_id or uuid.uuid4().hex[:12]
    _safe_print(f"Auditing {len(review_items)} items (run {run_id}) …")
    pkg = asyncio.run(pipeline.run(
        review_items, build_reference_resolver(sid_map),
        research_context=args.context, run_id=run_id, parser_record=parsed.record,
    ))

    fv = pkg.final_verification
    _safe_print("\n" + fv.summary)
    for f in fv.human_review_flags[:40]:
        _safe_print(f"  [{f.label}] {f.study_id}/{f.group}/{f.field_type}: {f.reason}")
    _safe_print(f"\n[PACKAGE] {(args.out / run_id / 'package.json').resolve()}")


def _report_main(argv: list[str] | None = None) -> None:
    """Render a saved EvidencePackage into a standalone HTML report.

    react-review report RUN_ID [--runs output/runs] [--out report.html]
    """
    from react_review.report import render_html_report
    from react_review.store import EvidencePackageStore

    ap = argparse.ArgumentParser(prog="react-review report",
                                 description="Render an audit run to an HTML report.")
    ap.add_argument("run_id", help="the run id (folder under --runs)")
    ap.add_argument("--runs", type=Path, default=Path("output/runs"))
    ap.add_argument("--out", type=Path, default=None, help="output .html (default: <run>/report.html)")
    args = ap.parse_args(argv)

    pkg = EvidencePackageStore(args.runs).load(args.run_id)
    out = args.out or (args.runs / args.run_id / "report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html_report(pkg), encoding="utf-8")
    _safe_print(f"[report] {out.resolve()}")


def main() -> None:
    """CLI entry point. Subcommands: ``run`` / ``report`` / ``audit`` / ``legacy``."""
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        return _run_main(argv[1:])
    if argv and argv[0] == "report":
        return _report_main(argv[1:])
    if argv and argv[0] == "audit":
        return _audit_main(argv[1:])
    if argv and argv[0] == "legacy":
        return _legacy_main(argv[1:])
    # Back-compat: no recognised subcommand falls through to the legacy pipeline.
    return _legacy_main(argv)
