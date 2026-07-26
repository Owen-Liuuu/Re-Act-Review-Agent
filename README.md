# ReAct-Review

An autonomous agentic pipeline for clinical evidence synthesis and audit.

ReAct-Review (a) extracts structured evidence tables from clinical publications and
(b) cross-validates already-published systematic reviews against their source papers.
A deterministic Python orchestrator sequences the stages; specialized bounded ReAct
agents (Evidence Collector, Evidence Auditor, Judge/Arbiter) operate over a shared,
typed tool catalogue.

## Status

**P1 — data contract + audit core (in progress).** The deterministic audit
chain is built and validated against the hand-labelled benchmark: a review table
and a source table are joined on `(study, group, timepoint, field_type)`, each
pair is compared with a dual-band tolerance (mean 1% + SD 3%; unit as a separate
axis), and the result is aggregated into an audit report. Done: schemas,
syntax-normalize, audit compare, the typed tool catalogue, and the thin
orchestrator. Remaining: the LLM `normalize_field` tool (Tier-2 semantic) and the
ReAct agents / real parser (P2). Migrated from the `lit_inspector` prototype in
P0 (lifted, renamed, dead code dropped, test baseline restored).

## Layout

```
src/react_review/
  core/          config, logging, exceptions, enums (+ AuditLabel)
  schemas/       4-table data contract: ReviewDataItem / SourceEvidenceItem /
                 IncludedStudy / MatchResult / ToleranceRule / AuditReport
  normalize/     Tier-1 syntax (deterministic): NumericValue parse (mean+SD),
                 unit normalization
  audit/         ToleranceTable (dual band) + compare_values
  tools/         typed tool catalogue (Search/Verify/Extract/Compare) + registry
  orchestrator/  matcher (4-tuple join) + thin deterministic pipeline
  llm/           backend ABC + retry engine + provider adapters
  pipeline/      legacy orchestrator/factory/schemas (reused impls live under steps/)
  steps/         reused implementations (search/verify/fetch/extract/report)
eval/            benchmark (ground truth) + run_benchmark.py + run_pipeline.py
docs/            normalization_pipeline.md
tests/           unit + integration (mock-mode) tests
```

## Quick start

```bash
pip install -e ".[dev]"
pytest                          # full test suite
python eval/run_benchmark.py    # score the audit core vs the answer key
python eval/run_pipeline.py     # end-to-end audit over the benchmark tables
```

## Reuse provenance

The following prototype modules carried over as high-value reusable assets:
four-tier full-text retrieval, CrossRef verification + confidence scoring,
field-level comparison primitives + per-concept tolerance table, the LLM retry
engine, and the DOCX report renderer.
