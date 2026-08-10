# ReAct-Review

A step-gated, human-in-the-loop pipeline for auditing systematic reviews against
their source papers — every step shows what it read and asks before continuing.

ReAct-Review cross-validates an already-published systematic review against the
papers it cites. It stops at each structural decision — the captured review table,
the cohorts it found, how it mapped columns to concepts, which references it could
resolve — and prints that step in full before asking whether to go on. A run can be
halted at any checkpoint, and the artefacts written up to that point are kept.

The language model only reads and proposes. Every judgement — whether two values
agree, whether a cohort matches, whether a total may be derived from its parts — is
made by deterministic code, and anything the code cannot settle is surfaced for a
human rather than resolved quietly. A deterministic orchestrator sequences the
stages; bounded ReAct agents (Evidence Collector, Evidence Auditor, Judge/Arbiter)
operate over a shared, typed tool catalogue.

## Status

**Phase 7 complete; Phase 8 in progress.** The review-to-source pipeline
includes review-derived cohort identities, structured numeric comparison,
controlled semantic escalation, governed DKB/checklist checkpoints, auditable
source-extraction replay, and HTML rendering from a previously saved Evidence
Package. Phase 7 added directed multi-arm extraction, typed value components,
confidence-level comparison and a self-consistency control over semantic
verdicts.

Every number below comes from a deterministic replay of a recorded run, and
each one names the artifact it comes from — see `docs/baselines/README.md` for
which file publishes which figure.

| Benchmark | Contract | Label accuracy | Discrepancy P/R/F1 | Silent releases |
| --- | --- | --- | --- | --- |
| EAT/T1DM (57 rows) | legacy | 89.47% | 80% / 80% / 80% | 0 |
| melanoma (15 rows) | Phase 7 | 80.0% | 100% / 100% / 100% | 0 |
| melanoma (15 rows) | Phase 8 (scope + exact counts) | 66.7% | 100% / 100% / 100% | 0 |

The Phase 8 figure is **lower on purpose**. It refuses two rows whose numbers
looked right while the evidence never said which population it counted — a
scope error that a relative tolerance had been reading as agreement. Refusing
is measured too: half the rows that require a population could not be assessed
at all, which is the capability cost of the fix and the reason it is reported
beside the safety numbers rather than instead of them.

**The cross-domain accuracy gate has not been passed, and passing it is not a
Phase 7 or Phase 8 acceptance target.** Fifteen rows cannot establish
cross-domain accuracy: one row moves label accuracy by 6.7 points. What the
melanoma checkpoint establishes is categorical — every route is reached,
failures are visible and reproducible, and three of the four archived defects
are now closed.

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
