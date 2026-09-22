# Running it: models, deployment, and what a run leaves behind

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


## Uploads, caches, and retention

Full-text PDFs are stored locally only; they are never redistributed.

- Web uploads land in `output/uploads/<run-id>/` (review PDF, source PDFs,
  `included_studies.csv`). Each folder contains `RETENTION.txt`.
- Extraction and semantic caches live in `output/runs/<run-id>/`. A run that
  used `--studies` / uploaded sources marks those caches `shareable: false`.
- There is no automatic cleanup. Delete both `output/uploads/<run-id>/` and
  `output/runs/<run-id>/` when the audit should leave the disk.

