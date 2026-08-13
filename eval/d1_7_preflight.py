"""Which recordings a D1-7 run would need, and whether the plan still describes it.

The contract routes values to `targeted_v5_batch` and arm identities to
`targeted_v4`, and both tools share ONE extraction cache. A `record` run against
an empty cache therefore goes live for both, and the recording stops isolating
the batch route — which is the only reason to make it.

So this answers four questions before anything is recorded, from the keys the
run will actually look up rather than from a count:

    every expected targeted_v4 key is a HIT, per claim and per attempt
    every targeted_v5_batch slot is a MISS — all `max_attempts` of them, not
      only the first, or a failed first attempt could read a stale recording
    the batches, by identity, are the ones the plan pre-registered
    the model settings are the ones the plan pre-registered

Both tools compute their cache key and consult the cache BEFORE reaching for a
model, so the keys can be enumerated without one. Nothing is written, the probe
answers are never recorded, and the target cache is opened read-only.

    python eval/d1_7_preflight.py --cache <the cache the recording will use>
    python eval/d1_7_preflight.py --emit-plan <path>     # regenerate the plan
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_review.csv_io import load_included_studies  # noqa: E402
from react_review.dkb import load_runtime_knowledge  # noqa: E402
from react_review.eval_excerpt import (  # noqa: E402
    benchmark_cohorts,
    benchmark_reviews,
)
from react_review.eval_profile import load_profile  # noqa: E402
from react_review.production import build_collector  # noqa: E402
from react_review.retrieval.local_pdf import LocalPdfRetriever  # noqa: E402
from react_review.steps.paper_verification.schemas import ReferenceEntry  # noqa: E402
from react_review.tools.extract import FetchFullTextTool  # noqa: E402
from react_review.tools.extract_batch import ExtractSourceBatchTool  # noqa: E402
from react_review.tools.extract_source import ExtractSourceValueTool  # noqa: E402
from react_review.tools.extraction_cache import ExtractionCache  # noqa: E402
from react_review.tools.extraction_profile import PROMPT_VERSIONS  # noqa: E402
from react_review.tools.registry import ToolRegistry  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

#: What a probe hands back for a key the cache does not hold. Valid shapes that
#: assert nothing: the run continues, and no answer here is ever recorded.
_EMPTY_BATCH = {"readings": []}
_EMPTY_SINGLE = {"found": False, "value": None, "unit": "", "quote": "",
                 "source_field_name": "", "location": "",
                 "not_found_reason": "preflight probe — the model was not asked"}

#: Settings that reach the model's behaviour, and therefore the recording. The
#: api key is not among them and is never read: what has to be frozen is how the
#: model was configured, not who was allowed to call it.
_PINNED_SETTINGS = ("provider", "model", "base_url", "temperature",
                    "max_tokens", "extra_body")


def model_settings(config_path: Path) -> dict:
    """The model configuration, without secrets.

    `configs/config.local.yaml` is gitignored and mutable, so a pre-registration
    that pinned only `model_id` would be pinning the smallest part of what
    decides what comes back. Temperature and max_tokens are not in the cache key
    and change the answer anyway.
    """
    import yaml

    body = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    llm = body.get("llm") or {}
    return {key: llm.get(key) for key in _PINNED_SETTINGS if key in llm}


def settings_digest(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")
                          ).hexdigest().upper()


class PinnedModel:
    """Names the model the RECORDING will use, and can do nothing else.

    The cache key contains the model id, and the tools take it from the backend
    first and from the open cache second — so a preflight run against an empty
    cache and no backend computes keys under the literal string "replay" and
    checks slots no recording will ever occupy. Pinning it here makes the
    preflight compute the same addresses the recording will write to.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        raise AssertionError("a preflight never asks the model")


class ProbeCache:
    """A read-only cache that remembers every key the run asked it for."""

    def __init__(self, recorded: ExtractionCache | None, empty: dict) -> None:
        self._recorded = recorded
        self._empty = empty
        self.lookups: list[tuple[str, bool]] = []
        self.model_id = getattr(recorded, "model_id", "") if recorded else ""
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        found = self._recorded.get(key) if self._recorded is not None else None
        self.lookups.append((key, found is not None))
        if found is not None:
            self.hits += 1
            return found
        self.misses += 1
        return dict(self._empty)

    def holds(self, key: str) -> bool:
        """Whether a key exists, WITHOUT counting as a lookup the run made."""
        return (self._recorded is not None
                and self._recorded.get(key) is not None)

    def put(self, key: str, value: dict, *, model_id: str = "") -> None:
        raise AssertionError("a preflight records nothing")

    def save(self):
        raise AssertionError("a preflight writes nothing")

    def __len__(self) -> int:
        return len(self._recorded) if self._recorded is not None else 0


class TracingSingleTool(ExtractSourceValueTool):
    """The real single-target tool, remembering WHICH claim asked for what.

    A count of hits cannot tell a covered claim from a claim whose retry is
    uncovered, and the difference is the whole question: a MISS after a HIT is a
    live call the recording does not contain.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace: list[dict] = []

    async def run(self, payload):
        before = len(self._cache.lookups)
        result = await super().run(payload)
        for key, hit in self._cache.lookups[before:]:
            self.trace.append({"group": payload.group,
                               "field_type": payload.field_type,
                               "attempt": payload.attempt,
                               "cache_key": key, "hit": hit})
        return result


class PlanningBatchTool(ExtractSourceBatchTool):
    """The real batch tool, remembering the prompt it built for each question.

    The prompt is what the keys are computed from, and computing them with the
    tool's own `_key` is the only way to be sure the preflight is checking the
    slots the run will use rather than its own arithmetic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asked: list[tuple] = []

    async def read(self, *, question, prompt, document):
        self.asked.append((question, prompt))
        return await super().read(question=question, prompt=prompt,
                                  document=document)

    def slots(self, prompt: str, attempts: int) -> list[str]:
        return [self._key(prompt, attempt) for attempt in range(attempts)]


def observe(benchmark: Path, profile_name: str, cache_path: Path | None,
            max_attempts: int, model_id: str):
    """Run the real Collector offline and report what it would need."""
    rows = list(csv.DictReader(
        (benchmark / "audit_template.csv").open(encoding="utf-8-sig")))
    profile = load_profile(benchmark, profile_name,
                           answer_key_ids=[r["audit_id"] for r in rows])
    contract = profile.run_contract
    if contract is None or not contract.batching:
        raise SystemExit(f"{profile_name} routes nothing to a batch")

    recorded = ExtractionCache(cache_path) if cache_path else None
    batch_probe = ProbeCache(recorded, _EMPTY_BATCH)
    single_probe = ProbeCache(recorded, _EMPTY_SINGLE)

    studies = load_included_studies(benchmark / "selected_studies.csv")
    retriever = LocalPdfRetriever(
        {s.doi: s.source_pdf for s in studies if s.doi and s.source_pdf},
        base_dir=benchmark)
    references = {s.study_id: ReferenceEntry(study_id=s.study_id,
                                             title=s.study_id, doi=s.doi)
                  for s in studies}

    model = PinnedModel(model_id)
    single = TracingSingleTool(model, cache=single_probe, cache_mode="replay")
    batch = PlanningBatchTool(model, cache=batch_probe, cache_mode="replay",
                              max_attempts=max_attempts)
    registry = ToolRegistry()
    registry.register(FetchFullTextTool(retriever))
    registry.register(single)
    registry.register(batch)
    # Wired as eval/run_full_accuracy.py wires it. The cohort display name
    # reaches the single-target prompt, so a collector built without the
    # registry asks a different question and computes keys for recordings that
    # were never going to exist.
    collector = build_collector(
        registry, contract=contract,
        cohorts=benchmark_cohorts(profile, rows),
        knowledge=load_runtime_knowledge(REPO / "configs/knowledge.seed.json",
                                         REPO / "configs/ontology"))

    # The context reaches the prompt, so a preflight that invented its own would
    # compute keys for a run nobody is going to make. Derived exactly as
    # eval/run_full_accuracy.py derives it.
    manifest_path = benchmark / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                if manifest_path.is_file() else {})
    research_context = str(manifest.get("domain") or "")

    reviews = benchmark_reviews(rows, profile.targets)
    batches: list[dict] = []
    for study_id in sorted({r.study_id for r in reviews}):
        claims = [r for r in reviews if r.study_id == study_id]
        result = asyncio.run(collector.collect_study(
            claims, references.get(study_id, ReferenceEntry(title=study_id)),
            research_context=research_context))
        for record in result.batch_records:
            persistent = record.persistent()
            question, prompt = next(
                (q, p) for q, p in batch.asked
                if q.identity() == persistent.question_id)
            batches.append({
                "field_type": persistent.field_type,
                "target_shape": persistent.target_shape,
                "raw_field_name": question.raw_field_name,
                "claim_ids": sorted(persistent.claim_ids),
                "question_id": persistent.question_id,
                # Every slot, not only the first: a failed first attempt reaches
                # for attempt 1, and a stale recording there would be replayed
                # into a run that believes it is live.
                "cache_key_slots": batch.slots(prompt, max_attempts),
            })
    batches.sort(key=lambda b: (b["field_type"], b["raw_field_name"]))
    return {"profile": profile, "contract": contract, "model_id": model_id,
            "batch_tool": batch,
            "research_context": research_context, "batches": batches,
            "v4_trace": single.trace, "batch_probe": batch_probe,
            "single_probe": single_probe}


def build_plan(observed: dict, *, benchmark: Path, profile_name: str,
               settings: dict, max_attempts: int) -> dict:
    return {
        "plan_id": "d1_7_expected_plan_v1",
        "written_on": "2026-08-13",
        "purpose": ("What a D1-7 recording is expected to ask for, fixed before "
                    "it is asked. A pre-registration nothing compares against is "
                    "a description, and this one was already stale once: the "
                    "identities in the first draft were computed under an empty "
                    "research context and never regenerated after it was fixed."),
        "benchmark": benchmark.name,
        "benchmark_profile": profile_name,
        "run_contract": observed["contract"].profile_id,
        "research_context": observed["research_context"],
        "max_attempts": max_attempts,
        "model_id": observed["model_id"],
        "model_settings": settings,
        "model_settings_sha256": settings_digest(settings),
        "prompt_versions": {
            "value": PROMPT_VERSIONS["targeted_v5_batch"],
            "arm_identity": PROMPT_VERSIONS["targeted_v4"],
        },
        "expected_batches": observed["batches"],
    }


def compare(plan: dict, observed: dict, settings: dict) -> list[str]:
    """Every way the run stopped being the run the plan describes."""
    drifts: list[str] = []
    if plan.get("research_context") != observed["research_context"]:
        drifts.append("the research context changed, and it reaches the prompt")
    if plan.get("model_id") != observed["model_id"]:
        drifts.append(
            f"the plan pre-registered model {plan.get('model_id')!r} and this is "
            f"{observed['model_id']!r}. The model id is inside every cache key, "
            "so this is a different set of recordings entirely")
    if plan.get("run_contract") != observed["contract"].profile_id:
        drifts.append(
            f"the plan pre-registered contract {plan.get('run_contract')!r} and "
            f"this is {observed['contract'].profile_id!r}")
    if settings and plan.get("model_settings_sha256") != settings_digest(settings):
        drifts.append(
            f"the model settings changed: pinned "
            f"{plan.get('model_settings_sha256', '')[:16]}, now "
            f"{settings_digest(settings)[:16]}. config.local.yaml is mutable and "
            "gitignored, so this is the only thing watching it")

    planned = {tuple(b["claim_ids"]): b for b in plan.get("expected_batches") or ()}
    actual = {tuple(b["claim_ids"]): b for b in observed["batches"]}
    for claims in sorted(set(planned) - set(actual)):
        drifts.append(f"the plan expects a batch for {list(claims)} and the run "
                      "does not make one")
    for claims in sorted(set(actual) - set(planned)):
        drifts.append(f"the run makes a batch for {list(claims)} that the plan "
                      "does not expect")
    for claims in sorted(set(planned) & set(actual)):
        for field in ("question_id", "field_type", "target_shape",
                      "raw_field_name", "cache_key_slots"):
            if planned[claims].get(field) != actual[claims].get(field):
                drifts.append(
                    f"batch {list(claims)}: {field} differs from the plan "
                    f"(planned {str(planned[claims].get(field))[:24]!r}, "
                    f"now {str(actual[claims].get(field))[:24]!r})")
    return drifts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="d1_7_preflight",
        description="Enumerate the recordings a D1-7 run needs, without making any.")
    ap.add_argument("--benchmark", type=Path,
                    default=REPO / "eval/benchmarks/melanoma_checkpoint_2017")
    ap.add_argument("--benchmark-profile", default="phase8_batch_v3_profile.json")
    ap.add_argument("--cache", type=Path, default=None,
                    help="the cache the RECORDING run would use")
    ap.add_argument("--config", type=Path, default=None,
                    help="the LLM config the recording would use; only "
                         "non-secret settings are read")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--model-id", default="",
                    help="the model the recording will use; taken from --config "
                         "when omitted. It is inside every cache key, so a "
                         "preflight that guesses it checks the wrong slots")
    ap.add_argument("--plan", type=Path, default=None,
                    help="the pre-registered plan to check against")
    ap.add_argument("--emit-plan", type=Path, default=None,
                    help="write the plan this checkout would produce")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    settings = model_settings(args.config) if args.config else {}
    model_id = args.model_id or str(settings.get("model") or "")
    if not model_id:
        raise SystemExit(
            "no model id: pass --model-id, or --config to take it from. It is "
            "part of every cache key, so a preflight without one checks "
            "addresses no recording will use")
    observed = observe(args.benchmark, args.benchmark_profile, args.cache,
                       args.max_attempts, model_id)

    if args.emit_plan:
        plan = build_plan(observed, benchmark=args.benchmark,
                          profile_name=args.benchmark_profile, settings=settings,
                          max_attempts=args.max_attempts)
        args.emit_plan.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8", newline="\n")
        print(f"[plan] {args.emit_plan.resolve()} "
              f"({len(plan['expected_batches'])} batches)")
        return 0

    trace = observed["v4_trace"]
    v4_uncovered = [t for t in trace if not t["hit"]]
    #: A retry the recording does not cover, counted only where the trajectory
    #: is still the run's own: after the first miss for a claim the probe answers
    #: where a model would have, so anything past it is the probe's doing.
    genuine = []
    seen_miss: set[tuple] = set()
    for entry in trace:
        claim = (entry["group"], entry["field_type"])
        if entry["hit"] or claim in seen_miss:
            continue
        seen_miss.add(claim)
        genuine.append(entry)

    slots = [(b, key) for b in observed["batches"] for key in b["cache_key_slots"]]
    occupied = [(b, key) for b, key in slots
                if observed["batch_probe"].holds(key)]

    print(f"benchmark : {args.benchmark.name} [{args.benchmark_profile}]")
    print(f"contract  : {observed['contract'].profile_id}")
    print(f"cache     : {args.cache or '(none — every key would be live)'}")
    print(f"context   : {observed['research_context']!r}")
    print(f"model     : {model_id}")
    if settings:
        print(f"settings  : {settings.get('model')} "
              f"temp={settings.get('temperature')} "
              f"max_tokens={settings.get('max_tokens')} "
              f"settings={settings_digest(settings)[:16]}")
    print()

    print(f"targeted_v4 (arm identity): {len(trace) - len(v4_uncovered)}/"
          f"{len(trace)} HIT")
    for entry in trace:
        print(f"    {'HIT ' if entry['hit'] else 'MISS'} {entry['group']:28s} "
              f"attempt={entry['attempt']}")
    if genuine:
        print(f"  -> AT LEAST {len(genuine)} live call(s) the recording does not "
              "cover:")
        for entry in genuine:
            print(f"       {entry['group']} attempt={entry['attempt']} — the "
                  "recorded answer failed a check under this contract and the "
                  "run asked again")
        print("     A lower bound, never a count: past a claim's first miss the "
              "probe answers where a model would have.")

    print(f"\ntargeted_v5_batch: {len(slots) - len(occupied)}/{len(slots)} slots "
          f"free ({len(observed['batches'])} batches x {args.max_attempts} "
          "attempts)")
    for batch, key in occupied:
        print(f"    OCCUPIED {key[:16]} {batch['claim_ids']} — a recording "
              "already exists for a slot this run would write")

    drifts: list[str] = []
    if args.plan:
        plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
        drifts = compare(plan, observed, settings)
        print(f"\nplan      : {args.plan.name} — "
              + ("matches" if not drifts else f"{len(drifts)} drift(s)"))
        for drift in drifts:
            print(f"    {drift}")
    else:
        print("\nplan      : none given — identity is NOT being checked")

    ok = (not v4_uncovered and not occupied and args.plan is not None
          and not drifts and bool(observed["batches"]))
    print("\nPREFLIGHT " + ("PASS" if ok else "FAIL"))

    if args.json:
        args.json.write_text(json.dumps({
            "benchmark": args.benchmark.name,
            "benchmark_profile": args.benchmark_profile,
            "contract": observed["contract"].profile_id,
            "model_settings_sha256": settings_digest(settings) if settings else "",
            "v4_trace": trace,
            "v4_uncovered_lower_bound": len(genuine),
            "v5_slots": len(slots),
            "v5_slots_occupied": [k for _, k in occupied],
            "expected_batches": observed["batches"],
            "plan_drifts": drifts,
            "preflight": "pass" if ok else "fail",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[json] {args.json.resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
