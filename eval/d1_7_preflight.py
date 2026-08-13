"""Which recordings a D1-7 run would need, and which of them already exist.

The contract routes values to `targeted_v5_batch` and arm identities to
`targeted_v4`, and both tools share ONE extraction cache. So a `record` run
against an EMPTY cache would go live for both — and the recording would no
longer isolate the batch route, which is the only reason to make it.

The preflight therefore has to answer two questions before anything is
recorded, and answer them from the keys the run will actually look up rather
than from a count:

    every expected targeted_v4 key is a HIT   (the arm identities are replayed)
    every expected targeted_v5_batch key is a MISS   (the batch route is new)

Both tools compute their cache key and consult the cache BEFORE reaching for a
model, so the keys can be enumerated without one. This runs the real Collector
over the benchmark's own claims with a cache that records every lookup and,
on a miss, hands back a syntactically valid empty answer so the run continues
to the end instead of stopping at the first gap.

Nothing is written. The probe answers are never recorded, and the target cache
is opened read-only.

    python eval/d1_7_preflight.py --cache output/baselines/melanoma_checkpoint_2017/phase7_extraction_cache.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_review.agents.collector import _claim_id  # noqa: E402
from react_review.csv_io import load_included_studies  # noqa: E402
from react_review.dkb import load_runtime_knowledge  # noqa: E402
from react_review.eval_excerpt import benchmark_reviews  # noqa: E402
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


class ProbeCache:
    """A read-only cache that remembers every key the run asked it for.

    Wrapping rather than subclassing the real cache: what matters is the exact
    sequence of lookups the tools perform, and a probe that reimplemented the
    key would be checking its own arithmetic.
    """

    def __init__(self, recorded: ExtractionCache | None, empty: dict) -> None:
        self._recorded = recorded
        self._empty = empty
        self.lookups: list[tuple[str, bool]] = []
        self.model_id = getattr(recorded, "model_id", "") if recorded else ""
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        found = self._recorded.get(key) if self._recorded is not None else None
        hit = found is not None
        self.lookups.append((key, hit))
        if hit:
            self.hits += 1
            return found
        self.misses += 1
        return dict(self._empty)

    def put(self, key: str, value: dict, *, model_id: str = "") -> None:
        raise AssertionError("a preflight records nothing")

    def save(self):
        raise AssertionError("a preflight writes nothing")

    def __len__(self) -> int:
        return len(self._recorded) if self._recorded is not None else 0


def _run(benchmark: Path, profile_name: str, cache_path: Path | None,
         model_id: str):
    rows = list(csv.DictReader(
        (benchmark / "audit_template.csv").open(encoding="utf-8-sig")))
    profile = load_profile(benchmark, profile_name,
                           answer_key_ids=[r["audit_id"] for r in rows])
    contract = profile.run_contract
    if contract is None or not contract.batching:
        raise SystemExit(f"{profile_name} routes nothing to a batch")

    recorded = ExtractionCache(cache_path) if cache_path else None
    if recorded is not None and model_id and recorded.model_id != model_id:
        raise SystemExit(
            f"the cache was recorded under model {recorded.model_id!r} and this "
            f"preflight assumes {model_id!r}. The model id is part of every "
            "cache key, so a different one turns every expected HIT into a MISS")

    batch_probe = ProbeCache(recorded, _EMPTY_BATCH)
    single_probe = ProbeCache(recorded, _EMPTY_SINGLE)

    studies = load_included_studies(benchmark / "selected_studies.csv")
    retriever = LocalPdfRetriever(
        {s.doi: s.source_pdf for s in studies if s.doi and s.source_pdf},
        base_dir=benchmark)
    references = {s.study_id: ReferenceEntry(study_id=s.study_id, title=s.study_id,
                                             doi=s.doi) for s in studies}

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(retriever))
    # replay mode: neither tool may reach a model, and the probe never misses
    # in a way that stops the run, so every key the run needs gets asked for.
    registry.register(ExtractSourceValueTool(None, cache=single_probe,
                                             cache_mode="replay"))
    registry.register(ExtractSourceBatchTool(None, cache=batch_probe,
                                             cache_mode="replay"))
    collector = build_collector(
        registry, contract=contract,
        knowledge=load_runtime_knowledge(REPO / "configs/knowledge.seed.json",
                                         REPO / "configs/ontology"))

    # The context reaches the prompt, so a preflight that invented its own
    # would compute keys for a run nobody is going to make. Derived exactly as
    # eval/run_full_accuracy.py derives it.
    manifest_path = benchmark / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                if manifest_path.is_file() else {})
    research_context = str(manifest.get("domain") or "")

    reviews = benchmark_reviews(rows, profile.targets)
    plans: list[dict] = []
    for study_id in sorted({r.study_id for r in reviews}):
        claims = [r for r in reviews if r.study_id == study_id]
        result = asyncio.run(collector.collect_study(
            claims, references.get(study_id, ReferenceEntry(title=study_id)),
            research_context=research_context))
        for record in result.batch_records:
            persistent = record.persistent()
            plans.append({
                "route": "targeted_v5_batch",
                "question_id": persistent.question_id,
                "execution_id": persistent.execution_id,
                "field_type": persistent.field_type,
                "target_shape": persistent.target_shape,
                "claim_ids": sorted(persistent.claim_ids),
                "attempts": persistent.attempts,
                "cache_keys": [a.cache_key for a in record.attempts],
            })
    return (profile, contract, plans, batch_probe, single_probe, reviews,
            research_context)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="d1_7_preflight",
        description="Enumerate the recordings a D1-7 run needs, without making any.")
    ap.add_argument("--benchmark", type=Path,
                    default=REPO / "eval/benchmarks/melanoma_checkpoint_2017")
    ap.add_argument("--benchmark-profile", default="phase8_batch_v3_profile.json")
    ap.add_argument("--cache", type=Path, default=None,
                    help="the cache the RECORDING run would use; omit to see "
                         "what an empty one would cost")
    ap.add_argument("--model-id", default="glm-4.5-flash",
                    help="pinned, because it is part of every cache key")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    (profile, contract, plans, batch_probe, single_probe, reviews,
     research_context) = _run(args.benchmark, args.benchmark_profile,
                              args.cache, args.model_id)

    v4_total = len(single_probe.lookups)
    v4_hits = sum(1 for _, hit in single_probe.lookups if hit)
    v5_total = len(batch_probe.lookups)
    v5_hits = sum(1 for _, hit in batch_probe.lookups if hit)

    print(f"benchmark : {args.benchmark.name} [{args.benchmark_profile}]")
    print(f"contract  : {contract.profile_id} value="
          f"{contract.extraction_routes.get('value')} arm_identity="
          f"{contract.extraction_routes.get('arm_identity')}")
    print(f"cache     : {args.cache or '(none — every key would be live)'}")
    print(f"context   : {research_context!r}")
    print(f"model_id  : {args.model_id} "
          f"[prompt versions {PROMPT_VERSIONS['targeted_v4']} / "
          f"{PROMPT_VERSIONS['targeted_v5_batch']}]")
    print()
    print(f"targeted_v4 (arm identity): {v4_hits}/{v4_total} HIT  "
          f"— must be all HIT, or the recording is not only about batching")
    if v4_hits != v4_total:
        print("            " + " ".join(
            ("HIT" if hit else "MISS") for _, hit in single_probe.lookups))
        print("            a MISS after a HIT is a RETRY the recording does not "
              "cover: the recorded answer failed a deterministic check under "
              "this contract and the run asked again.")
        print("            The exact number of live calls cannot be read off "
              "this: past the first miss the probe answers where a model would "
              "have, so the trajectory diverges. Treat it as AT LEAST "
              f"{v4_total - v4_hits} and never as exactly that.")
    print(f"targeted_v5_batch         : {v5_total - v5_hits}/{v5_total} MISS "
          f"— must be all MISS, or this run was already recorded")
    print()
    print(f"batches the run would make: {len(plans)}")
    for plan in plans:
        print(f"  {plan['field_type']:26s} {plan['target_shape']:11s} "
              f"claims={plan['claim_ids']}")
        print(f"      question_id  {plan['question_id'][:32]}")
        print(f"      cache_key(s) {[k[:16] for k in plan['cache_keys']]}")

    ok = (v4_hits == v4_total) and (v5_hits == 0) and bool(plans)
    print()
    print("PREFLIGHT " + ("PASS — a record run would go live for the batch "
                          "route and nothing else" if ok else
                          "FAIL — see the two lines above"))

    if args.json:
        args.json.write_text(json.dumps({
            "benchmark": args.benchmark.name,
            "benchmark_profile": args.benchmark_profile,
            "contract": contract.profile_id,
            "model_id": args.model_id,
            "cache": str(args.cache) if args.cache else "",
            "targeted_v4": {"lookups": v4_total, "hits": v4_hits},
            "targeted_v5_batch": {"lookups": v5_total, "hits": v5_hits},
            "expected_batches": plans,
            "preflight": "pass" if ok else "fail",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[json] {args.json.resolve()}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
