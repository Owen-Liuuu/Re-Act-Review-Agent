# ReAct-Review

Audits a published systematic review against the papers it cites — and stops to
show its work at every step where a person should be the one deciding.

<!-- TODO: screenshot of a gated checkpoint in the web UI -->
<!-- ![A checkpoint waiting for a decision](docs/screenshots/gate.png) -->

![tests](https://img.shields.io/badge/tests-2009%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![docker](https://img.shields.io/badge/deploy-docker%20compose-blue)

## The problem

A systematic review pools numbers from dozens of primary papers. When one of
those numbers is copied wrong, nothing downstream notices: the meta-analysis,
the forest plot and the clinical recommendation all inherit the error silently.
Checking by hand means re-reading every source paper.

This tool reads the review's own tables and forest plots, fetches the source
papers, and reports — cell by cell — whether the source actually says what the
review attributed to it.

## The constraint that shaped everything

**The language model reads and proposes. It never decides.**

Whether two values agree, whether a cohort matches, whether a total may be
derived from its parts — each is settled by deterministic code. Anything the
code cannot settle is escalated to a human instead of resolved quietly. The
pipeline stops at nine structural decisions and prints what it read before
asking to continue.

That constraint is what makes the output auditable. It is also what made the
engineering interesting.

## Three decisions worth a look

### Prompts are frozen by the bytes they render to

Every prompt ships as a contract pinning the SHA-256 of its *rendered* output:

```json
{
  "contract_id": "table_locate_v1",
  "rendered_prompt_sha256": "BA7B328B503D7DE3B9AB6DEA060AFD147B49816392FCB791BE6F637C7FE8889A",
  "governance": "A rendered prompt that changes is a NEW PROMPT VERSION. Do not
                 edit this file to make an edited prompt pass."
}
```

Twenty such contracts are in the repo. Editing a prompt in place fails the
build; a change means publishing a new version beside the old one, which keeps
every recorded run replayable. Extraction caches key on the prompt hash, so a
reworded prompt is a clean cache miss rather than a silent mix of two prompts'
answers.

### Human-in-the-loop is a protocol, not an `input()` call

```python
class CheckpointGate(Protocol):
    async def check(self, event: StepEvent, *, force_gate: bool = False,
                    hold_display: bool = False) -> Decision: ...
```

One method, four implementations: a terminal gate, a scripted one for tests, an
auto-continue for CI, and a web gate that awaits a decision over HTTP. The same
nine checkpoints run unchanged in the CLI, in the test suite, and in the
browser. Each step records *how* it was cleared — `gate`, `show`, `silent` or
`auto` — so "a person reviewed this" is provable after the fact rather than
assumed.

### Failures have to say their own name

The recurring bug class in this project is the one where a failure returns the
same value as a legitimate answer, so nothing can see it. A retrieval helper
that swallowed its exception and returned `[]` is indistinguishable from a
paper that genuinely has no tables.

Concretely: an LLM whose reasoning consumed the entire token budget returned
empty content, which surfaced downstream as `Failed to parse JSON`. Three
debugging rounds chased the wrong cause. It now reports:

```
truncated: reasoning used the whole budget
(reasoning_tokens=16384 of max_tokens=16384)
```

The same rule was applied to read timeouts, rate limits and permanent provider
errors — each is now distinguishable from "no result", and each is retried
according to whether retrying can actually help.

## Architecture

```
src/react_review/
  parser/        read the review: tables, forest plots, cohorts, claims
  retrieval/     find and fetch source papers (PMC, Unpaywall, OpenAlex, PDF)
  agents/        bounded ReAct workers — collector, auditor, judge
  tools/         typed search / extract / compare catalogue with replay hooks
  audit/         deterministic comparison, semantic escalation, aggregation
  normalize/     cohorts and units discovered from the review, not hardcoded
  orchestrator/  stage sequencing, checkpoints, the pipeline itself
  hitl/          the checkpoint protocol, its gates, and the run journal
  llm/           provider adapters, retry policy, reasoning control
  web/           upload, run, and answer checkpoints in a browser
configs/         prompt contracts, evaluator versions, model routing
eval/            frozen benchmarks and accuracy scoring
```

The language model is reachable only through `tools/`. Nothing in `audit/`
calls one.

## Quick start

```bash
docker compose up -d --build     # web UI on http://localhost:8080
```

Or from the CLI:

```bash
pip install -e ".[dev]"
react-review run --pdf review.pdf --config configs/config.local.yaml
```

Copy `.env.example` to `.env` and add one API key per model gear. Keys are read
from the environment and never written to the image or the repo.

## Digging deeper

| | |
| --- | --- |
| [Running it](docs/operations.md) | model gears and routing, Docker, and what each run writes to disk |
| [Benchmarks and reproduction](docs/dissertation.md) | frozen answer keys, accuracy figures, and the replay procedure |
| [Known limitations](docs/known-limitations.md) | what it cannot do yet, and why |
| [Baselines](docs/baselines/README.md) | what is published, what is withheld, and the hashes to verify a local copy |

Built as an MSc dissertation project. 38,500 lines across 186 modules, covered
by 2,009 tests.
