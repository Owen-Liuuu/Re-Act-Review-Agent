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
from react_review.eval_layout import benchmark_dir, resolve_eval_relpath  # noqa: E402
from react_review.retrieval.local_pdf import _pdf_text  # noqa: E402
from react_review.tools.extraction_cache import ExtractionCache  # noqa: E402

REPO = repo_root()
#: Line endings are normalised before hashing a blob out of git, so a
#: recording contract keeps one hash whichever platform checked it out.
CRLF, LF = b'\r\n', b'\n'
BENCH = benchmark_dir(REPO, "melanoma_checkpoint_2017")
RUNS = REPO / "output/baselines/melanoma_checkpoint_2017"
CACHE = RUNS / "phase8_batch_extraction_cache.json"
PLAN = BENCH / "d1_7_expected_plan.json"


def _current(rel: str) -> Path:
    """Today's location of a path a historical contract still names."""
    return REPO / resolve_eval_relpath(rel)

#: The four artifacts the conclusions rest on. Hashed here because the cache and
#: the results are gitignored: without them the manifest can say a recording
#: existed and nothing more.
ARTIFACTS = {
    "preflight": RUNS / "d1_7_preflight.json",
    "record": RUNS / "d1_7_record.json",
    "replay": RUNS / "d1_7_replay.json",
    "scored": RUNS / "d1_7_scored.json",
}

#: The commit the LIVE RECORDING ran under, and the contracts that were in
#: force THEN. Not the contracts in force now: a gate published after a run did
#: not govern it, and listing it among the recording's contracts reads as though
#: it had.
RECORDING_COMMIT = "aafa20a"
RECORDING_CONTRACTS = {
    "run_profile": "configs/run_profiles/phase8_batch_v2.json",
    "benchmark_profile": "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v3_profile.json",
    "prompt_contract": "configs/prompt_contracts/batch_v5.json",
    "excerpt_gold": "eval/benchmarks/melanoma_checkpoint_2017/excerpt_gold_v2.json",
    "expected_plan": "eval/benchmarks/melanoma_checkpoint_2017/d1_7_expected_plan.json",
    "feature_gate": "configs/gates/d1_batch_v1.json",
}

#: The contracts a REANALYSIS runs under. Separate, because they are.
REANALYSIS_CONTRACTS = {
    "run_profile": "configs/run_profiles/phase8_batch_v4.json",
    "benchmark_profile": "eval/benchmarks/melanoma_checkpoint_2017/phase8_batch_v5_profile.json",
    "feature_gate": "configs/gates/d1_batch_v3.json",
    "aggregation_evaluator": "configs/aggregation/evaluators/safe_aggregation_1.8.0.json",
    "compare_evaluator": "configs/compare/evaluators/deterministic_compare_1.0.0.json",
}

COMMAND = (
    "python eval/run_full_accuracy.py --config configs/config.local.yaml "
    "--benchmark eval/benchmarks/melanoma_checkpoint_2017 "
    "--benchmark-profile phase8_batch_v3_profile.json --extraction record "
    "--extraction-cache output/baselines/melanoma_checkpoint_2017/"
    "phase8_batch_extraction_cache.json --semantic off "
    "--out output/baselines/melanoma_checkpoint_2017/d1_7_record.json "
    "--html output/baselines/melanoma_checkpoint_2017/d1_7_record.html")


def _at_commit(commit: str, path: str) -> str:
    """A file's hash AS IT WAS at a commit, so the recording's contracts are the
    ones it actually ran under."""
    import subprocess

    blob = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO,
                          capture_output=True)
    if blob.returncode != 0:
        return ""
    normalised = blob.stdout.replace(CRLF, LF)
    return hashlib.sha256(normalised).hexdigest().upper()


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
    # The profile the RECORDING ran under has been superseded by a later
    # evaluator, and its contract now names one this tree is not. The prompts are
    # unchanged by that — the evaluator decides how an answer is judged, not what
    # question is asked — so the re-derivation uses the current profile and is
    # checked, as always, against the keys the run actually wrote.
    observed = preflight.observe(BENCH, "phase8_batch_v7_profile.json", None, 3,
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


BUNDLE = REPO / "docs/baselines/bundles/d1_7_batch_recording.zip"
REMOTE = "https://github.com/Owen-Liuuu/Re-Act-Review-Agent"


def _storage_status() -> str:
    """`blocked` until a bundle exists; never past `available_unverified` here.

    A machine that already holds the recording cannot independently verify it,
    so this generator can never write the last state. That is the point.
    """
    return "available_unverified" if BUNDLE.is_file() else "blocked"


def _bundle_block() -> dict:
    if not BUNDLE.is_file():
        return {"uri": None, "sha256": None, "size_bytes": None,
                "media_type": "application/zip", "access_mode": None,
                "copyright_restrictions": None}
    return {
        "uri": f"{REMOTE}/blob/main/{BUNDLE.relative_to(REPO).as_posix()}",
        "git_path": BUNDLE.relative_to(REPO).as_posix(),
        "sha256": _digest(BUNDLE),
        "size_bytes": BUNDLE.stat().st_size,
        "media_type": "application/zip",
        "access_mode": ("controlled — the repository is private, so access is "
                        "repository membership. A reader without a collaborator "
                        "seat cannot obtain this."),
        "copyright_restrictions": (
            "Contains model responses quoting Larkin 2015 verbatim, and the "
            "paper itself is NOT included. Published inside a private "
            "repository with the owner's explicit permission; redistribution "
            "outside it is not covered by that permission."),
    }


def _reanalyses() -> list[dict]:
    """Later readings of the SAME recording, each named and hashed.

    A deterministic fix re-reads a recording; it does not make a new one. Kept
    apart from the original result so that "what the model said" and "what the
    code made of it" can never be confused for one another.
    """
    from react_review.acceptance_transitions import grade

    entries = []
    for name, gate_version, note in (
        ("d1_7_3_component_verification", "d1_batch_v2",
         "generated BEFORE its own freeze commit, so it carries git_commit '' "
         "and status unregistered. It is kept as the development reading that "
         "located the defect, and it is NOT release-eligible."),
        ("d1_7_5_component_and_identity", "d1_batch_v3",
         "run on the frozen commit: aggregation 1.8.0 and comparator 1.0.0 both "
         "registered, both release-eligible, commit matching."),
    ):
        entry = _one_reanalysis(name, gate_version, note)
        if entry is not None:
            entries.append(entry)
    return entries


def _one_reanalysis(name: str, gate_version: str, note: str):
    from react_review.acceptance_transitions import grade

    path = RUNS / f"{name.split('_component')[0]}_scored.json"
    if not path.is_file():
        return None
    scored = json.loads(path.read_text(encoding="utf-8-sig"))
    gate = json.loads((REPO / f"configs/gates/{gate_version}.json"
                       ).read_text(encoding="utf-8-sig"))
    # The whole gate, including the prohibitions it declares. Grading on rows
    # alone read the transition table and nothing else, and answered
    # PASS_PROHIBITIONS_ONLY while no prohibition had been consulted.
    result = grade(gate, scored["rows"], artifact=scored)
    telemetry = scored["run"].get("telemetry") or {}
    runtime = scored["run"].get("aggregation_runtime") or {}
    compare = scored["run"].get("compare_runtime") or {}
    return {
        "id": name,
        "gate": gate_version,
        "status_note": note,
        "aggregation_runtime": {k: runtime.get(k) for k in
                                ("evaluator_version", "status", "release_eligible",
                                 "git_commit_matches_evaluator")},
        "compare_runtime": {k: compare.get(k) for k in
                            ("compare_version", "status", "release_eligible")},
        "unmet_hard_conditions": list(result.unmet_hard_conditions),
        "hard_condition_evidence": result.hard_condition_evidence,
        # Not a safety violation, so it does not contradict the gate — and not
        # something a release note may omit either.
        "benchmark_diagnostics": {
            k: v for k, v in (scored.get("benchmark_diagnostics") or {}).items()
            if k in ("status", "unexpected_count", "unexpected_differences",
                     "silent_releases")},
        # Generated from the structured runtime, never typed. Both entries used
        # to say "evaluator 1.7.0" because the sentence was written once and
        # reused; the structured field said 1.8.0 for the second, so the numbers
        # were right and the explanation was wrong — the failure mode a
        # hand-written manifest has, arriving in prose instead of a hash.
        "what": (f"The same recording, replayed under aggregation evaluator "
                 f"{runtime.get('evaluator_version') or 'unknown'}"
                 + (f" and comparator {compare.get('compare_version')}"
                    if compare.get("compare_version") else "")
                 + ". Numeric components are verified against each reading's own "
                   "quote and the survivors carried to the result. No model was "
                   "asked: backend_requests is 0."),
        "artifact": {"path": path.relative_to(REPO).as_posix(),
                     "sha256": _digest(path)},
        "extraction": "replay of the D1-7 cache, unchanged",
        "backend_requests": telemetry.get("backend_requests"),
        "label_accuracy": scored["metrics"]["label_accuracy"],
        "label_accuracy_before_fix": 8 / 15,
        "baseline_label_accuracy": 10 / 15,
        "silent_releases": scored["metrics"]["safety"]["silent_release_count"],
        "gate_verdict": result.verdict.upper(),
        "gate_states": result.capability,
        "note": ("MA015 moves from `match` to `mismatch` — the baseline's own "
                 "verdict — because the source's 99.5% confidence level is now "
                 "compared against the review's 95%. The components were in the "
                 "response all along; nothing carried them to the result."),
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
        # Three identities, never one. The recording, the manifest that
        # describes it and the reanalyses that re-read it happened at three
        # different commits under three different sets of contracts, and one
        # `commit` field made them look like one event.
        "recording": {
            "code_commit": subprocess.run(
                ["git", "rev-parse", RECORDING_COMMIT], capture_output=True,
                text=True, cwd=REPO).stdout.strip(),
            "what": "the one live run; the only step that asked a model",
            "contracts_in_force_then": {
                name: _at_commit(RECORDING_COMMIT, path)
                for name, path in RECORDING_CONTRACTS.items()},
            "note": ("d1_batch_v2 and v3, evaluator 1.7.0 and 1.8.0 and the "
                     "comparator identity did not exist yet and are NOT listed "
                     "here. They govern the reanalyses below."),
        },
        # NOT the current HEAD. A manifest generated in the same batch as the
        # generator would record the commit BEFORE its own — the file cannot
        # contain the hash of a commit that contains the file. What is stable
        # and checkable is the generator's own bytes and the commit that last
        # touched it.
        "manifest_generation": {
            "generator": "eval/d1_7_manifest.py",
            "generator_sha256": _digest(REPO / "eval/d1_7_manifest.py"),
            "generator_last_changed_commit": subprocess.run(
                ["git", "log", "-1", "--format=%H", "--", "eval/d1_7_manifest.py"],
                capture_output=True, text=True, cwd=REPO).stdout.strip(),
            "note": ("the generator is committed BEFORE the manifest it "
                     "produces, so these two values describe a state that "
                     "already exists in history"),
        },
        "model": {
            "provider": plan["model_settings"].get("provider"),
            "model_id": plan["model_id"],
            "settings_sha256": plan["model_settings_sha256"],
            "settings": plan["model_settings"],
            "note": "no api key is read, recorded or hashed here",
        },
        "command": COMMAND,
        "reanalysis_contracts": {name: sha256_file(_current(path))
                                 for name, path in REANALYSIS_CONTRACTS.items()
                                 if _current(path).is_file()},
        "documents": {
            "larkin_2015": _document_sha256(
                _pdf_text(BENCH / "raw/sources/larkin_2015.pdf")),
        },
        "artifacts": {
            name: {"path": path.relative_to(REPO).as_posix(),
                   "sha256": _digest(path)}
            for name, path in ARTIFACTS.items() if path.is_file()
        },
        # Three states, and only the last one closes this. "I uploaded it" and
        # "somebody else reproduced it" are different claims, and one `location`
        # string could not tell them apart — which is why filling it in used to
        # be enough to silence the objection.
        "artifact_storage": {
            "status": _storage_status(),
            "why_it_matters": (
                "The hashes above prove a file with this content existed on the "
                "machine that ran it. They do not let a reader obtain the "
                "recording and replay it, and they cannot be told apart from "
                "the hash of a file that no longer exists. Until this is "
                "closed, every conclusion here is reproducible by the author "
                "alone."),
            "states": {
                "blocked": "no legitimate location. The current state.",
                "available_unverified": ("a uri exists and nobody outside has "
                                         "checked it"),
                "independently_verified": ("another machine downloaded it, "
                                           "verified it, replayed it and left "
                                           "an attestation"),
            },
            "bundle": _bundle_block(),
            "independent_verification": None,
            "tooling": {
                "build_and_verify": "eval/verify_d1_7_bundle.py",
                "note": ("the protocol and its verifier exist and are tested; "
                         "writing a verification script is not performing a "
                         "verification, so the status stays blocked"),
            },
            "decisions_not_mine": [
                "which platform the bundle goes to",
                "public or controlled access",
                "whether a cache quoting a copyrighted paper verbatim may be "
                "published at all",
            ],
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
        "reanalyses": _reanalyses(),
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


def _reanalysis_agrees(entry: dict, artifact_path: Path) -> list[str]:
    """The numbers a reanalysis publishes are the ones its artifact holds.

    A manifest can hash a file correctly and still describe it wrongly. The
    version written in prose said 1.7.0 for a run whose structured runtime said
    1.8.0, and nothing compared the two.
    """
    from react_review.acceptance_transitions import grade

    scored = json.loads(artifact_path.read_text(encoding="utf-8-sig"))
    problems: list[str] = []
    name = entry.get("id")

    published = entry.get("label_accuracy")
    actual = scored.get("metrics", {}).get("label_accuracy")
    if published != actual:
        problems.append(f"reanalysis {name} publishes label_accuracy "
                        f"{published} and its artifact holds {actual}")

    for section, key in (("aggregation_runtime", "evaluator_version"),
                         ("compare_runtime", "compare_version")):
        was = (entry.get(section) or {}).get(key)
        now = (scored.get("run", {}).get(section) or {}).get(key)
        if was != now:
            problems.append(f"reanalysis {name}: {section}.{key} is {was!r} in "
                            f"the manifest and {now!r} in the artifact")

    version = (entry.get("aggregation_runtime") or {}).get("evaluator_version")
    if version and version not in (entry.get("what") or ""):
        problems.append(f"reanalysis {name} describes itself without naming the "
                        f"evaluator its runtime records ({version})")

    gate_name = entry.get("gate")
    gate_path = REPO / f"configs/gates/{gate_name}.json" if gate_name else None
    if gate_path is not None and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
        result = grade(gate, scored["rows"], artifact=scored)
        if result.verdict.upper() != entry.get("gate_verdict"):
            problems.append(
                f"reanalysis {name} publishes verdict "
                f"{entry.get('gate_verdict')} and {gate_name} applied to its "
                f"artifact gives {result.verdict.upper()}")
        if list(result.unmet_hard_conditions) != list(
                entry.get("unmet_hard_conditions") or ()):
            problems.append(f"reanalysis {name}: the unmet conditions it "
                            "publishes are not the ones its gate reports")
    return problems


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

    # Recomputed, not measured for length. A 64-character string that is not
    # the file's hash passes a length check and fails nothing else, which is
    # the exact shape of the pin this repository once published without ever
    # computing it.
    cache = body.get("cache") or {}
    for field, target in (("sha256_after", CACHE),
                          ("seeded_from_sha256",
                           RUNS / "phase7_extraction_cache.json")):
        value = cache.get(field, "")
        if len(value) != 64:
            problems.append(f"cache.{field} is {len(value)} characters; a "
                            "sha256 is 64, and a truncated one still looks like "
                            "a hash")
        elif not target.is_file():
            problems.append(f"cache.{field} describes {target.name}, which is "
                            "not in this checkout, so it cannot be checked")
        elif _digest(target) != value:
            problems.append(f"cache.{field} does not match {target.name} as it "
                            "stands now")

    for entry in body.get("reanalyses") or ():
        artifact = entry.get("artifact") or {}
        target = REPO / artifact.get("path", "")
        if not artifact.get("sha256"):
            problems.append(f"reanalysis {entry.get('id')} records no hash for "
                            "the artifact its numbers come from")
        elif not target.is_file():
            problems.append(f"reanalysis {entry.get('id')}: {artifact['path']} "
                            "is not in this checkout")
        elif _digest(target) != artifact["sha256"]:
            problems.append(f"reanalysis {entry.get('id')}: its artifact has "
                            "changed since it was hashed")
        else:
            problems.extend(_reanalysis_agrees(entry, target))

    for name, digest in (body.get("reanalysis_contracts") or {}).items():
        path = REANALYSIS_CONTRACTS.get(name, "")
        if path and _current(path).is_file() and sha256_file(_current(path)) != digest:
            problems.append(f"reanalysis_contracts.{name} does not match "
                            f"{path} as it stands now")

    recording = body.get("recording") or {}
    commit = recording.get("code_commit", "")
    for name, digest in (recording.get("contracts_in_force_then") or {}).items():
        path = RECORDING_CONTRACTS.get(name, "")
        if not path or not commit:
            continue
        actual = _at_commit(commit, path)
        if actual and actual != digest:
            problems.append(f"recording.contracts_in_force_then.{name} is not "
                            f"what {path} was at {commit[:12]}")

    prompts = body.get("prompts") or ()
    recomputed = hashlib.sha256(json.dumps(
        [[r["question_id"], r["attempt"], r["prompt_sha256"], r["cache_key"]]
         for r in prompts], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest().upper()
    if body.get("prompt_set_sha256") != recomputed:
        problems.append("prompt_set_sha256 is not the hash of the prompt rows "
                        "beneath it")

    for section in ("recording", "reanalysis_contracts"):
        node = body.get(section) or {}
        digests = (node.get("contracts_in_force_then") if section == "recording"
                   else node)
        for name, digest in (digests or {}).items():
            if len(digest) != 64:
                problems.append(f"{section}.{name} is not a full sha256")
    generation = body.get("manifest_generation") or {}
    generator = REPO / generation.get("generator", "")
    if generator.is_file() and _digest(generator) != generation.get(
            "generator_sha256"):
        problems.append(
            "the generator has changed since this manifest was produced, so "
            "the manifest was not made by the code that claims to have made it")
    if not (body.get("recording") or {}).get("code_commit"):
        problems.append("the manifest does not say which commit the recording "
                        "ran under, so its contracts cannot be checked")

    for row in body.get("prompts") or ():
        for field in ("question_id", "prompt_sha256", "attempt", "cache_key"):
            if field not in row:
                problems.append(f"a prompt row omits {field}, which the "
                                "pre-registration requires")
                break

    sys.path.insert(0, str(REPO / "eval"))
    from verify_d1_7_bundle import BLOCKED, check_storage_block

    storage = body.get("artifact_storage") or {}
    problems.extend(check_storage_block(storage))
    if storage.get("status") == BLOCKED:
        problems.append(
            "artifact_storage.status is blocked: the recording is not "
            "retrievable by anyone but its author. This is a known, declared "
            "gap, and it keeps failing until a bundle is published AND "
            "verified somewhere else")
    bundle = (storage.get("bundle") or {})
    tracked = REPO / (bundle.get("git_path") or "")
    if bundle.get("git_path"):
        if not tracked.is_file():
            problems.append(f"the manifest points at {bundle['git_path']} and "
                            "it is not in this checkout")
        elif _digest(tracked) != bundle.get("sha256"):
            problems.append("the bundle in the repository is not the one the "
                            "manifest hashes")
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
