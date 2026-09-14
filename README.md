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
  llm/           backend ABC + retry engine + provider adapters + factory
eval/            frozen EAT and melanoma benchmarks + accuracy runners
docs/            architecture, limitations, sanitized baselines, deferred issues
                 version_numbering_zh.md maps the six independent v-number
                 namespaces and says which one production runs
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

## Choosing and changing models

Three gears carry the models. Each of the 13 model tasks is routed to one:

| Gear | Web-page default | Tasks |
|---|---|---|
| reasoning (`llm`) | `deepseek-v4-pro` | judgement: `evidence_localize`, and any task not listed under `routing` |
| transcribe (`backend_profiles.transcribe`) | `deepseek-v4-flash`, reasoning off | copying and simple labelling, as listed under `routing` |
| vision (`vision`) | `glm-4.6v` | `forest_ocr_vision` |

`configs/config.example.yaml` ships `llm` as a `mock` placeholder so the test
suite runs without keys; put real providers in `configs/config.local.yaml`.

There are three ways to change them.

1. **Web page.** Open *Account* (top right) and paste one native vendor key per
   gear (not an OpenRouter key). On the home page, under *Three gears*, pick a
   vendor and model for Complex, Simple and Visual. The page lists, under each
   gear, the tasks it will serve in that run; the routing itself still comes
   from the host config, with `table_capture`, `forest_ocr_text`,
   `claim_origin`, `unpivot` and `references` always on Simple. With no keys in
   Account the run uses the host config, but only if the host has its own
   model key; otherwise the page asks for the three keys.
2. **Config file.** Edit `llm`, `backend_profiles.transcribe`, `vision` and
   `routing` in `configs/config.local.yaml`. Unknown task names under
   `routing` are a hard error.
3. **One run.** `react-review run ... --profile-all transcribe` sends all 13
   tasks to one named gear, which is useful for comparisons.

To see what actually ran, read `backend_model_id` and `backend_reasoning` in
`output/runs/<run-id>/steps/NNN_<step>.json`; web runs also write `gears.json`
(vendors and models, never keys).

**Pitfall:** reasoning belongs to the gear, not to the model name. A task left
on `llm` keeps reasoning after you change the model name to a lighter model;
route it to a gear that sets `reasoning: off` instead.

## Deploying with Docker

One container serves the web UI on port 8080.

```bash
cp .env.example .env            # fill in keys and the Basic Auth login
docker compose up -d --build    # build and start; restarts unless stopped
docker compose logs -f          # follow the server log
```

- `serve` refuses to start without `REACT_REVIEW_BASIC_USER` and
  `REACT_REVIEW_BASIC_PASSWORD` (use `--auth-off` only for local tests). Put
  the container behind a reverse proxy with HTTPS: Basic Auth sends the
  password with every request.
- Keys come only from `.env`; they are never baked into the image.
  `REACT_REVIEW_TRANSCRIBE_API_KEY` falls back to `REACT_REVIEW_LLM_API_KEY`.
- The image runs as a non-root user and reads `configs/config.example.yaml`,
  whose model entries are placeholders: in the container the models come from
  the three gears on the page, and a run needs three vendor keys in *Account*
  unless `REACT_REVIEW_LLM_API_KEY` is set. Routing comes from that file; edit
  it before building to change routing.
- Runs and uploads live on the `react-review-data` volume at `/data`
  (`/data/runs`, `/data/uploads`) and survive rebuilds.
- PDFs and artifacts stay on the host. The text each step reads is sent to the
  model vendor of its gear.
- To update: pull, then `docker compose up -d --build` again.

## What a run writes

Every run writes `output/runs/<run-id>/` (in the container, `/data/runs/<run-id>/`):

| File | Content |
|---|---|
| `journal.ndjson` | one line per step: step name, title, warning count, artefact pointers |
| `steps/NNN_<step>.json` | the full state of each step: rendered text, options, warnings, decision, interaction mode, and the model that ran |
| `checkpoints.log` | a readable transcript of the whole run with volatile fields suppressed, so two runs can be diffed |
| `report.html` | the final audit report |
| `package.json` | summary and telemetry once the run completes; `package.partial.json` while it has not |
| `semantic_cache.json` | recorded semantic comparisons, reused on replay |
| `proposals.json` | candidate knowledge-base concepts for `react-review learn` (only when the run collected any) |
| `gears.json` | web runs only: vendor and model of each gear, never keys |

Use a different `--run-id` for every run: two runs writing into one directory
overwrite each other's step files by index.

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

## Uploads, caches, and retention

Full-text PDFs are stored locally only; they are never redistributed.

- Web uploads land in `output/uploads/<run-id>/` (review PDF, source PDFs,
  `included_studies.csv`). Each folder contains `RETENTION.txt`.
- Extraction and semantic caches live in `output/runs/<run-id>/`. A run that
  used `--studies` / uploaded sources marks those caches `shareable: false`.
- There is no automatic cleanup. Delete both `output/uploads/<run-id>/` and
  `output/runs/<run-id>/` when the audit should leave the disk.

## Reuse provenance

The following prototype modules carried over as high-value reusable assets:
four-tier full-text retrieval, CrossRef verification + confidence scoring,
field-level comparison primitives + per-concept tolerance table, and the LLM
retry engine.
