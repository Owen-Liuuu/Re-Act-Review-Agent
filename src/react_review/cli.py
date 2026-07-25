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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="react-review",
        description="LLM-powered academic literature integrity detection",
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
    args = parser.parse_args()

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
