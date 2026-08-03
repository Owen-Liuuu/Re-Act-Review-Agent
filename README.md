# ReAct-Review

An autonomous agentic pipeline for clinical evidence synthesis and audit.

ReAct-Review (a) extracts structured evidence tables from clinical publications and
(b) cross-validates already-published systematic reviews against their source papers.
A deterministic Python orchestrator sequences the stages; specialized bounded ReAct
agents (Evidence Collector, Evidence Auditor, Judge/Arbiter) operate over a shared,
typed tool catalogue.

## Status

**Phase 6E — final acceptance complete.** The review-to-source pipeline now
includes review-derived cohort identities, structured numeric comparison,
controlled semantic escalation, governed DKB/checklist checkpoints, auditable
source-extraction replay, and HTML rendering from a previously saved Evidence
Package.

The frozen EAT/T1DM replay scores 89.47% label accuracy, 80% strict discrepancy
precision/recall/F1, zero silent releases, and 100% human-review visibility.
The melanoma checkpoint exercised all expected cross-domain paths (semantic,
numeric, and structured) with zero silent releases, but its 60% label accuracy
failed the accuracy gate. Those multi-arm extraction and semantic issues are
preserved for later work; the project therefore claims cross-domain mechanism
coverage, not established cross-domain accuracy. See
`docs/baselines/phase6e_acceptance.json` and
`docs/deferred/phase6b-melanoma-audit.md`.

## Layout

```
src/react_review/
  core/          config, logging, exceptions, enums (+ AuditLabel)
  schemas/       review/source evidence, match results, reasons, and reports
  normalize/     review-derived cohorts, units, and structured numeric values
  audit/         component comparison, semantic controls, caches, aggregation
  dkb/           governed field resolution and provisional knowledge lifecycle
  tools/         typed Search/Verify/Extract/Compare catalogue and replay hooks
  orchestrator/  matching, collection, judging, checkpoints, and pipeline
  agents/        bounded collector/auditor/judge workflows
  llm/           backend ABC + retry engine + provider adapters
eval/            frozen EAT and melanoma benchmarks + accuracy runners
docs/            architecture, limitations, sanitized baselines, deferred issues
tests/           unit + integration (mock-mode) tests
```

## Quick start

```bash
pip install -e ".[dev]"
pytest                          # full test suite
python eval/run_benchmark.py    # score the audit core vs the answer key
python eval/run_pipeline.py     # end-to-end audit over the benchmark tables

# Deterministic audit from the CLI (no LLM): match review vs source, compare,
# print the report, and persist the run's evidence package under --out.
react-review audit review.csv source.csv --out output/runs

# Full review-to-source run. The final Evidence Package is saved atomically
# first; report.html is then rendered by reloading that saved package.
react-review run --pdf review.pdf --studies included_studies.csv \
  --config configs/config.local.yaml --out output/runs --run-id example

# Re-render the same deterministic HTML later from package.json only.
react-review report example --runs output/runs
```

A successful full run writes `output/runs/<run-id>/package.json` followed by
`output/runs/<run-id>/report.html`. Use `run --html another/path.html` to choose
a different report location. The HTML includes the source file/URI, verbatim
quote, deterministic derivation, semantic relation and controls, and every
human-review flag carried by the saved Evidence Package.

## Reuse provenance

The following prototype modules carried over as high-value reusable assets:
four-tier full-text retrieval, CrossRef verification + confidence scoring,
field-level comparison primitives + per-concept tolerance table, the LLM retry
engine, and the DOCX report renderer.
