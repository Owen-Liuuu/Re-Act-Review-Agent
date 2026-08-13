# D1-7 pre-registration — the one live recording of the batch route

Written before the run, so that what counts as success is fixed before anyone
sees a number. Nothing here has been executed against a model. The batch route
has never been sent to one: the repository holds two extraction recordings,
`phase6b` (14 entries) and `phase7` (23), both single-target under
`glm-4.5-flash`.

**Status: BLOCKED.** The preflight fails. See §7 — the decision it needs is
yours, and the run must not be requested until it is made.

---

## 1. The one harness, and the exact command

The recording runs through `eval/run_full_accuracy.py` and nothing else. It
reads the benchmark's answer key directly and never invokes `ReviewParser`, so
the parser's swallow-everything failure path (L17 in
`docs/known-limitations.md`) cannot affect this recording. That is the reason
the eval harness is pinned rather than `react-review run`.

```
python eval/run_full_accuracy.py \
  --config configs/config.local.yaml \
  --benchmark eval/benchmarks/melanoma_checkpoint_2017 \
  --benchmark-profile phase8_batch_v3_profile.json \
  --extraction record \
  --extraction-cache output/baselines/melanoma_checkpoint_2017/phase8_batch_extraction_cache.json \
  --semantic off \
  --out output/baselines/melanoma_checkpoint_2017/d1_7_record.json \
  --html output/baselines/melanoma_checkpoint_2017/d1_7_record.html
```

`--semantic off` is deliberate and is settled: the semantic cache key includes
`source_value`, `source_unit` and `source_quote`
(`src/react_review/tools/compare.py:104`), and v5 changes exactly those, so
`--semantic cache-only` would miss and hard-fail. Semantic compatibility is
audited AFTER the extraction recording exists (§5), not during it.

`--out` and `--html` name paths that do not exist yet. **A path is never
reused** — overwriting the pre-fix report once already cost a `git stash` back
to recording-time code to regenerate it.

## 2. What "one recording" means

One recording is ONE run of the command above. Pre-registered before it starts:

| | |
|---|---|
| model | `glm-4.5-flash`, thinking disabled via `extra_body`, `max_tokens ≥ 4096` |
| `max_attempts` | 3 (the batch tool's default; retried only for `TRANSPORT` and `NOT_JSON`) |
| expected batches | **7**, enumerated in §3 |
| expected cache state before | every `targeted_v5_batch` key MISS |
| expected cache state after | one entry per attempt actually made |

If the run does not produce a usable recording, **re-running requires a new
authorization**. "Run it again and keep the better one" is the failure this
project has already recorded once, and a second run silently chosen from is not
one experiment.

The recording holds the **decoded JSON**, not the model's literal bytes — the
cache has never held those (`tools/extract_batch.py:262`). A `NOT_JSON` failure
therefore leaves only its classification and a truncated detail string. If that
is what happens, the recording is a record of the failure, not of a reading.

## 3. The seven batches, pre-registered by identity

Computed by `eval/d1_7_preflight.py` from the real Collector, the real
knowledge base, the real target contract and the real selector — no model. Any
difference between this list and what the run produces invalidates the run.

| field | shape | claims | question_id (first 32) |
|---|---|---|---|
| `study_design` | study | MA001 | `16A425BFFD2818783358F3C467B6C923` |
| `sample_size` | study | MA002 | `D336DC8264D266604706557A6E6DA0AA` |
| `cohort_n` | arm | MA004 | `0E3F3515C76AE52E421E834903252B32` |
| `cohort_n` | arm | MA006 | `DE13718F93084B07B86EAFB80EF1BEA7` |
| `cohort_n` | arm | MA008 | `B0432596973FE4BFF34E859588177E61` |
| `progression_free_survival` | arm | MA009, MA010, MA011 | `CF707C87E4F1D4EDD075FA5D998B0740` |
| `hazard_ratio` | comparison | MA012, MA013, MA014, MA015 | `3357EF6D5EADDEBDE2DC548F3CC7CFAF` |

Note what this says about the benchmark: only two of the seven batches carry
more than one claim. The three arm counts come from three different review
columns, so the group key separates them and batching buys nothing there. A
"cost of batching" claim on this benchmark is a claim about 7 claims read in 2
batches plus 5 read alone — not about 15 claims read in 5.

## 4. Preflight — what must be true before the model is asked

`python eval/d1_7_preflight.py --cache <the cache the run will use>`

1. `targeted_v4` (arm identity): **every** expected key HIT.
2. `targeted_v5_batch`: **every** expected key MISS.
3. The seven batches above, by claim binding, unchanged.
4. The cache's `model_id` equals the pinned model — it is part of every key, so
   a different one turns every expected HIT into a MISS.
5. `python eval/excerpt_dry_run.py --benchmark-profile phase8_batch_v3_profile.json`
   returns 0 (currently: 7/7 covered, 0 missing, 0 unjudged).

Condition 1 is what makes the recording *about batching*. Both tools share one
`ExtractionCache`, so a run against an empty cache goes live for the arm
identities too, and the recording stops isolating the route.

## 5. Order of operations

```
preflight passes
  → extraction record, semantic OFF                ← the only live step
  → immediately re-run with --extraction replay, assert identical
  → semantic compatibility audit: do the new source values and quotes hit the
    existing phase7 semantic cache?
       all hit  → score with --semantic cache-only
       any miss → STOP and report. A new semantic recording is a separate
                  authorization; it is a second live variable.
  → grade: metrics v2, eval/compare_projection.py against phase8_profile.json
  → write the recording manifest (§6) and freeze its hashes
```

## 6. The recording manifest

The cache itself lives under `output/`, which is gitignored, so the manifest is
what enters the repository. It is not a single `prompt_sha`: one batched run
issues several prompts, and one number cannot describe them.

```
model_id, provider, thinking-disabled flag, max_tokens
commit (40 chars), run profile + benchmark profile hashes
prompt contract hash (configs/prompt_contracts/batch_v5.json)
excerpt gold hash, document sha256 of every source paper
the exact command line, and the wall-clock date
per prompt, sorted:  question_id · prompt_sha256 · attempt · cache_key
a set hash over that sorted list
cache file sha256 · entry count · storage path
```

## 7. Why this is BLOCKED, and the decision needed

Preflight run on 2026-08-13 against
`output/baselines/melanoma_checkpoint_2017/phase7_extraction_cache.json`:

```
targeted_v4 (arm identity): 6/9 HIT      ← must be 9/9
            HIT MISS MISS HIT HIT HIT HIT HIT MISS
targeted_v5_batch         : 7/7 MISS     ← as required
PREFLIGHT FAIL
```

Condition 2 holds. **Condition 1 does not.** A MISS after a HIT is a retry: the
recorded phase7 answer was replayed, failed a deterministic check under the
phase8 contract, and the run asked again — and the retry attempt was never
recorded, because under phase7 it never happened. The number is a lower bound,
not a count: past the first miss the probe answers where a model would have, so
the trajectory diverges.

So the run as specified would go live for the batch route **and** for at least
one arm-identity retry, and could not be described as isolating batching.

Three ways forward. This is a decision about what the experiment is, so it is
not mine to make:

- **(a) Record both, and say so.** One run, one authorization. The result
  measures "the phase8_batch contract end to end", not "batching alone". Cheapest
  and honest, but it forfeits the comparison the route change was meant to
  support.
- **(b) Two authorizations.** First record the arm-identity route alone under
  the phase8 contract until every v4 key is covered; freeze that cache; then
  record v5 with v4 fully replayed. This is the only path that yields a genuine
  single-variable recording, and it costs two live runs.
- **(c) Drop the arm-identity route from this recording.** Restrict the run to
  the value route, and grade only the claims that route answers. No v4 call
  happens at all, so the isolation is exact — at the price of leaving MA003,
  MA005 and MA007 ungraded in this experiment.

My recommendation is **(b)**, because the whole reason D1-7 exists is to
measure what the batch route does differently, and (a) cannot answer that while
(c) shrinks an already small benchmark. But (a) is defensible if one live run is
the budget.

## 8. What the result may and may not claim

- Report v5's measured cost. **Do not compute a speedup.** Under v5 the
  `single_extraction` bucket holds arm identities and the `batch` bucket holds
  value claims: different tasks, not a before/after of the same one. A speedup
  claim needs a same-model, same-scope `targeted_v4` baseline, which is another
  recording and another authorization.
- Grade against the D1 feature gate and per-row transitions.
- The D6 cross-domain conclusion stays **NOT ESTIMABLE**. One study cannot move
  it, and nothing in this recording is evidence about a third domain.
- Expect defects. Phase 7's single recording exposed a fault no test had, and
  the fix was deterministic projection over the SAME recording. Budget one such
  round before considering a re-record.
