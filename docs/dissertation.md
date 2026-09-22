# Benchmarks, accuracy, and reproduction

Moved out of the README when the repository was refocused as a portfolio
project. Nothing here has been re-scored; the figures are the ones the
dissertation reports, and each still names the artifact it came from.

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


## Reproducing the dissertation experiments

**Full-text accessibility (dissertation Section 4.5).** The two runs differ only
in `--studies` and `--pdf-dir`:

```bash
react-review run --pdf eval/benchmark_1/raw/EAT_T1DM_SRMA.pdf \
  --studies eval/benchmark_1/included_studies.csv --pdf-dir eval/benchmark_1 \
  --config configs/config.local.yaml --checkpoints none --non-interactive \
  --run-id access-local

react-review run --pdf eval/benchmark_1/raw/EAT_T1DM_SRMA.pdf \
  --config configs/config.local.yaml --checkpoints none --non-interactive \
  --run-id access-online
```

**Deterministic replay (Sections 4.2 and 4.6).** Record a live run with
`--extraction record`, then replay it offline with `--extraction replay`.
Replay makes no model calls, so differences between control layers come from
deterministic code alone.

**Benchmark scoring.** The scripts under `eval/` score runs against the
hand-built answer keys of `benchmark_1` (EAT/T1DM) and `benchmark_2`
(melanoma). Answer keys, checklists and source PDFs are frozen; their hashes
are recorded under `docs/baselines/`.


## Reuse provenance

The following prototype modules carried over as high-value reusable assets:
four-tier full-text retrieval, CrossRef verification + confidence scoring,
field-level comparison primitives + per-concept tolerance table, and the LLM
retry engine.
