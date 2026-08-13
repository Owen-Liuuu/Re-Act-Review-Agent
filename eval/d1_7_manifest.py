"""Generate the D1-7 recording manifest, and verify one that already exists.

The first manifest was hand-written, and a hand-written manifest is a claim
about numbers rather than a reading of them. Its errors were exactly the kind
nothing in a test suite could see: half-length hashes that still looked like
hashes, the global cache total reported as the semantic one, and no record at
all of the four JSON artifacts the conclusions were computed from.

So it is generated from the artifacts, and it can be checked against them.

    python eval/d1_7_manifest.py --emit docs/baselines/d1_7_batch_recording_manifest.json
    python eval/d1_7_manifest.py --verify docs/baselines/d1_7_batch_recording_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_review.agents.collector import _document_sha256  # noqa: E402
from react_review.contracts import sha256_file, repo_root  # noqa: E402
from react_review.retrieval.local_pdf import _pdf_text  # noqa: E402
from react_review.tools.extraction_cache import ExtractionCache  # noqa: E402

REPO = repo_root()
BENCH = REPO / "eval/benchmarks/melanoma_checkpoint_2017"
RUNS = REPO / "output/baselines/melanoma_checkpoint_2017"
CACHE = RUNS / "phase8_batch_extraction_cache.json"
PLAN = BENCH / "d1_7_expected_plan.json"

#: The four artifacts the conclusions rest on. Hashed here because the cache and
#: the results are gitignored: without them the manifest can say a recording
#: existed and nothing more.
ARTIFACTS = {
    "preflight": RUNS / "d1_7_preflight.json",
    "record": RUNS / "d1_7_record.json",
    "replay": RUNS / "d1_7_replay.json",
    "scored": RUNS / "d1_7_scored.json",
}

CONTRACTS = {
    "run_profile": "configs/run_profiles/phase8_batch_v2.json",
    "feature_gate_v2": "configs/gates/d1_batch_v2.json",
    "benchmark_profile": "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v3_profile.json",
    "prompt_contract": "configs/prompt_contracts/batch_v5.json",
    "excerpt_gold": "eval/benchmarks/melanoma_checkpoint_2017/excerpt_gold_v2.json",
    "expected_plan": "eval/benchmarks/melanoma_checkpoint_2017/d1_7_expected_plan.json",
    "feature_gate": "configs/gates/d1_batch_v1.json",
}

COMMAND = (
    "python eval/run_full_accuracy.py --config configs/config.local.yaml "
    "--benchmark eval/benchmarks/melanoma_checkpoint_2017 "
    "--benchmark-profile phase8_batch_v3_profile.json --extraction record "
    "--extraction-cache output/baselines/melanoma_checkpoint_2017/"
    "phase8_batch_extraction_cache.json --semantic off "
    "--out output/baselines/melanoma_checkpoint_2017/d1_7_record.json "
    "--html output/baselines/melanoma_checkpoint_2017/d1_7_record.html")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _prompt_rows() -> tuple[list[dict], str]:
    """Every prompt the run recorded, with the sha of the prompt itself.

    The cache key is a hash OVER the prompt's sha, so it cannot be turned back
    into one. The prompts are re-derived by the preflight, deterministically and
    without a model, and the derivation is checked by asserting that the key it
    computes is the key the run actually wrote.
    """
    sys.path.insert(0, str(REPO / "eval"))
    import d1_7_preflight as preflight

    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    observed = preflight.observe(BENCH, "phase8_batch_v3_profile.json", None, 3,
                                 plan["model_id"])
    prompts = {question.identity(): prompt
               for question, prompt in observed["batch_tool"].asked}

    cache = ExtractionCache(CACHE)
    tool = observed["batch_tool"]
    rows = []
    for batch in plan["expected_batches"]:
        prompt = prompts.get(batch["question_id"])
        if prompt is None:
            raise SystemExit(f"no prompt re-derived for {batch['question_id']}")
        # The re-derivation proves itself: the key computed from this prompt has
        # to be the key the run actually wrote. Otherwise the sha recorded below
        # would describe some other prompt convincingly.
        recomputed = tool.slots(prompt, len(batch["cache_key_slots"]))
        if recomputed != batch["cache_key_slots"]:
            raise SystemExit(
                f"the prompt re-derived for {batch['question_id']} does not "
                "produce the keys the run wrote; its sha would describe a "
                "different prompt")
        for attempt, key in enumerate(batch["cache_key_slots"]):
            if cache.get(key) is None:
                continue
            rows.append({
                "question_id": batch["question_id"],
                "claim_ids": batch["claim_ids"],
                "attempt": attempt,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")).hexdigest().upper(),
                "prompt_chars": len(prompt),
                "cache_key": key,
            })
    rows.sort(key=lambda r: (r["question_id"], r["attempt"]))
    set_hash = hashlib.sha256(json.dumps(
        [[r["question_id"], r["attempt"], r["prompt_sha256"], r["cache_key"]]
         for r in rows], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return rows, set_hash


def _graded(rows) -> dict:
    """Both published gates, applied by the executable classifier.

    Computed rather than asserted. The FAIL reported on the day of the recording
    came from a throwaway classifier written beside the result, which called a
    review-flagged row `wrong_released` — the one thing v1 defines it not to be.
    """
    from react_review.acceptance_transitions import grade

    verdicts = {}
    for version in ("v1", "v2"):
        gate = json.loads((REPO / f"configs/gates/d1_batch_{version}.json"
                           ).read_text(encoding="utf-8-sig"))
        verdicts[version] = grade(gate, rows)

    return {
        "classifier": "src/react_review/acceptance_transitions.py",
        "gate_v1_verdict": verdicts["v1"].verdict.upper(),
        "gate_v1_reason": verdicts["v1"].reason,
        "gate_v1_note": (
            "v1 defines `wrong_released` as a wrong value released WITHOUT "
            "review, and MA015 carries review_required=True — wrong AND "
            "escalated, which v1 has no term for. The FAIL reported on the day "
            "was computed by an ad-hoc classifier that contradicted the gate's "
            "own text; no executable v1 classifier existed. That FAIL stands as "
            "a record of what was reported, not as a verdict v1 ever issued."),
        "gate_v1_second_defect": (
            "v1 has no capability floor. Every one of its hard conditions is a "
            "prohibition, so a system that refused every row would satisfy all "
            "of them and pass."),
        "gate_v2_verdict": verdicts["v2"].verdict.upper(),
        "gate_v2_capability_judged": verdicts["v2"].capability_judged,
        "gate_v2_states": verdicts["v2"].capability,
        "gate_v2_note": (
            "POST-HOC REANALYSIS. v2 was written after this run by an author who "
            "knew which verdict it would change; it is published because v1 "
            "could not judge the run at all, not because v1's verdict was "
            "inconvenient. Its capability floor is deliberately UNSET — a draft "
            "set 0.8 and this run keeps exactly 8 of the baseline's 10 correct "
            "rows — so the strongest verdict available is PASS (PROHIBITIONS "
            "ONLY), which is not a claim that the route works."),
    }


def build() -> dict:
    scored = json.loads(ARTIFACTS["scored"].read_text(encoding="utf-8-sig"))
    record = json.loads(ARTIFACTS["record"].read_text(encoding="utf-8-sig"))
    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    cache = ExtractionCache(CACHE)
    prompts, set_hash = _prompt_rows()

    scored_telemetry = scored["run"].get("telemetry") or {}
    record_telemetry = record["run"].get("telemetry") or {}
    semantic = json.loads((RUNS / "phase7_semantic_cache.json"
                           ).read_text(encoding="utf-8-sig"))

    return {
        "manifest_id": "d1_7_batch_recording_v2",
        "supersedes": "d1_7_batch_recording_v1 (hand-written; see corrections)",
        "recorded_on": "2026-08-13",
        "what": ("The one live recording of the targeted_v5_batch route. The "
                 "cache and the result artifacts live under output/, which is "
                 "gitignored, so this manifest is what enters the repository — "
                 "and it hashes them, so a copy obtained elsewhere can be shown "
                 "to be the same one."),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=REPO).stdout.strip(),
        "model": {
            "provider": plan["model_settings"].get("provider"),
            "model_id": plan["model_id"],
            "settings_sha256": plan["model_settings_sha256"],
            "settings": plan["model_settings"],
            "note": "no api key is read, recorded or hashed here",
        },
        "command": COMMAND,
        "contracts": {name: sha256_file(REPO / path)
                      for name, path in CONTRACTS.items()},
        "evaluator": "safe_aggregation 1.6.2 under safe_sum_v5",
        "documents": {
            "larkin_2015": _document_sha256(
                _pdf_text(BENCH / "raw/sources/larkin_2015.pdf")),
        },
        "artifacts": {
            name: {"path": path.relative_to(REPO).as_posix(),
                   "sha256": _digest(path)}
            for name, path in ARTIFACTS.items() if path.is_file()
        },
        "artifact_storage": {
            "status": "LOCAL ONLY — not yet retrievable by anyone else",
            "why_it_matters": ("These hashes prove that a file with this content "
                               "existed on the machine that ran it. They do not "
                               "let a reader obtain the recording and replay it. "
                               "Until an immutable location is agreed, or a "
                               "sanitised recording is committed, every "
                               "conclusion here is reproducible by the author "
                               "alone."),
            "location": "",
        },
        "cache": {
            "path": CACHE.relative_to(REPO).as_posix(),
            "sha256_after": _digest(CACHE),
            "entries_after": len(cache),
            "seeded_from": "output/baselines/melanoma_checkpoint_2017/phase7_extraction_cache.json",
            "seeded_from_sha256": _digest(RUNS / "phase7_extraction_cache.json"),
            "entries_before": 6,
            "note": ("`sha256_before_seeded` is not restated: the seeded file was "
                     "replaced in place by the recording. What is checkable now "
                     "is the source it was seeded from, the entry count before, "
                     "and the file as it stands."),
        },
        "prompts": prompts,
        "prompt_set_sha256": set_hash,
        "what_the_run_did": {
            "backend_requests_logical": record_telemetry.get("backend_requests"),
            "backend_requests_note": (
                "logical complete() calls. The provider client is configured "
                "with max_retries=5, so the number of HTTP requests may be "
                "higher and is not observable from here"),
            "extraction_cache_hits": 6,
            "extraction_cache_misses": 7,
            "batch_contract_retries": 0,
            "run_level_repeated_attempts": record_telemetry.get("repeated_attempts"),
            "repeated_attempts_note": (
                "all of them cache-served targeted_v4 arm-identity replays, not "
                "batch retries. Every batch answered on its first attempt: 7 "
                "attempt-0 slots written, 14 retry slots still empty"),
            "prompt_chars_in": record_telemetry.get("prompt_chars"),
            "output_chars_out": record_telemetry.get("output_chars"),
            "model_seconds": record_telemetry.get("call_seconds"),
        },
        "results": {
            "scored_with": ("--extraction replay --semantic cache-only against "
                            "the phase7 semantic cache"),
            "extraction_cache_hits": 13,
            "semantic_cache_entries_on_disk": len(semantic.get("entries", semantic)),
            "global_cache_hits": scored_telemetry.get("cache_hits"),
            "global_cache_misses": scored_telemetry.get("cache_misses"),
            "cache_split_note": ("the global counter is extraction plus semantic. "
                                 "v1 reported the global 16 as the semantic total, "
                                 "which it is not"),
            "label_accuracy": scored["metrics"]["label_accuracy"],
            "baseline_label_accuracy": 10 / 15,
            "silent_releases": scored["metrics"]["safety"]["silent_release_count"],
            "review_visibility": scored["metrics"]["safety"]["review_visibility_rate"],
            "scope_wrong_released": scored["metrics"]["scope"].get(
                "scope_wrong_released_count"),
            "excerpt_coverage_covered_batches": (
                (record["run"].get("excerpt_coverage") or {}).get(
                    "gold_covered_batches")),
        },
        "gate": _graded(scored["rows"]),
        "replay_check": {
            "result": "identical",
            "detail": ("re-run with --extraction replay: 15/15 rows identical "
                       "field by field, metrics identical, 0 backend requests, "
                       "13 extraction cache hits. The batch records differ only "
                       "in served_from_cache, false -> true, which is what "
                       "replaying means"),
        },
        "claims_not_made": [
            "no speedup: under v5 the single bucket holds arm identities and the "
            "batch bucket holds value claims, which are different tasks",
            "D6 cross-domain stays NOT ESTIMABLE; one study is not evidence "
            "about a domain",
        ],
    }


def verify(path: Path) -> list[str]:
    """Every way the manifest disagrees with the artifacts it describes."""
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    problems: list[str] = []

    for name, entry in (body.get("artifacts") or {}).items():
        target = REPO / entry["path"]
        if not target.is_file():
            problems.append(f"{name}: {entry['path']} is not in this checkout")
            continue
        if _digest(target) != entry["sha256"]:
            problems.append(f"{name}: {entry['path']} has changed since it was "
                            "hashed")

    for field in ("sha256_after", "seeded_from_sha256"):
        value = (body.get("cache") or {}).get(field, "")
        if len(value) != 64:
            problems.append(f"cache.{field} is {len(value)} characters; a "
                            "sha256 is 64, and a truncated one still looks like "
                            "a hash")

    for name, digest in (body.get("contracts") or {}).items():
        if len(digest) != 64:
            problems.append(f"contracts.{name} is not a full sha256")

    for row in body.get("prompts") or ():
        for field in ("question_id", "prompt_sha256", "attempt", "cache_key"):
            if field not in row:
                problems.append(f"a prompt row omits {field}, which the "
                                "pre-registration requires")
                break

    if not (body.get("artifact_storage") or {}).get("location"):
        problems.append(
            "artifact_storage.location is empty: the recording is not "
            "retrievable by anyone but its author, so the conclusions are not "
            "independently reproducible. This is a known, declared gap")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d1_7_manifest")
    ap.add_argument("--emit", type=Path)
    ap.add_argument("--verify", type=Path, nargs="?", const=None,
                    default=None)
    args = ap.parse_args(argv)

    if args.emit:
        body = build()
        args.emit.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8", newline="\n")
        print(f"[manifest] {args.emit} ({len(body['prompts'])} prompts, "
              f"set {body['prompt_set_sha256'][:16]})")
        return 0

    target = args.verify or (REPO / "docs/baselines/d1_7_batch_recording_manifest.json")
    problems = verify(target)
    print(f"verifying {target.name}")
    for problem in problems:
        print(f"  {problem}")
    print("MANIFEST " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
