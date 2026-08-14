"""Prove a D1-7 bundle reproduces the published result — on a machine that has nothing.

A hash in a manifest shows that a file with that content existed somewhere. It
does not let anyone else obtain the recording, and it cannot be told apart from
a hash of a file that no longer exists. The gap has been declared since the
recording was made; this is the machinery that can close it, and it deliberately
cannot close it by itself.

Three states, and only the last one counts:

``blocked``                 no legitimate location. The current state.
``available_unverified``    a URI exists and nobody has checked it from outside.
``independently_verified``  another machine downloaded it, verified it, replayed
                            it and produced an attestation.

The middle state is the one worth naming. "I uploaded it" and "somebody else
reproduced it" are different claims, and a single `location` string could not
tell them apart — which is why filling one in used to be enough to silence the
objection.

    python eval/verify_d1_7_bundle.py --build   out/d1_7_bundle.zip
    python eval/verify_d1_7_bundle.py --verify  out/d1_7_bundle.zip --attest out/attestation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_review.contracts import repo_root  # noqa: E402

REPO = repo_root()
RUNS = REPO / "output/baselines/melanoma_checkpoint_2017"

BLOCKED = "blocked"
AVAILABLE_UNVERIFIED = "available_unverified"
INDEPENDENTLY_VERIFIED = "independently_verified"
STATES = (BLOCKED, AVAILABLE_UNVERIFIED, INDEPENDENTLY_VERIFIED)

#: A URI a human could act on. `foo` is not one, and neither is an empty string
#: — the previous check was "is it non-empty", which any word satisfied.
ALLOWED_SCHEMES = ("https://", "s3://", "ipfs://", "doi:")

#: What a replay needs, and nothing it does not. The extraction cache alone is
#: not enough: the scoring run also replays semantic judgements, and without
#: them the numbers cannot be reproduced at all.
REQUIRED_MEMBERS = (
    "phase8_batch_extraction_cache.json",
    "phase7_semantic_cache.json",
    "d1_7_batch_recording_manifest.json",
    "BUNDLE.json",
)

#: What the replay must produce. Not "roughly this": the recording is
#: deterministic, so anything else means the bundle is not the recording.
EXPECTED = {
    "backend_requests": 0,
    "extraction_cache_hits": 13,
    "extraction_cache_misses": 0,
    "semantic_cache_hits": 3,
    "semantic_cache_misses": 0,
    "correct_rows": 9,
    "total_rows": 15,
    "label_accuracy": 9 / 15,
    "silent_release_count": 0,
    "identity_wrong_released": 0,
    "scope_wrong_released_count": 0,
}

REPLAY = (
    "python eval/run_full_accuracy.py --config <your config> "
    "--benchmark eval/benchmarks/melanoma_checkpoint_2017 "
    "--benchmark-profile phase8_batch_v5_profile.json --extraction replay "
    "--extraction-cache <bundle>/phase8_batch_extraction_cache.json "
    "--semantic cache-only --semantic-cache <bundle>/phase7_semantic_cache.json "
    "--out <somewhere>/d1_7_5_scored.json")


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def build(target: Path) -> dict:
    """Assemble the bundle a third party would need. Uploading is not this."""
    scored = RUNS / "d1_7_5_scored.json"
    members = {
        "phase8_batch_extraction_cache.json": RUNS / "phase8_batch_extraction_cache.json",
        "phase7_semantic_cache.json": RUNS / "phase7_semantic_cache.json",
        "d1_7_batch_recording_manifest.json":
            REPO / "docs/baselines/d1_7_batch_recording_manifest.json",
    }
    missing = [name for name, path in members.items() if not path.is_file()]
    if missing or not scored.is_file():
        raise SystemExit(f"cannot build: {missing or [scored.name]} not here")

    descriptor = {
        "bundle_id": "d1_7_batch_recording",
        "repo_commit": _commit(),
        "benchmark_profile": "phase8_batch_v5_profile.json",
        "replay_command": REPLAY,
        "expected_scored_sha256": digest(scored),
        "expected": EXPECTED,
        "members": {name: digest(path) for name, path in members.items()},
        "source_paper": {
            "study_id": "larkin_2015",
            "document_sha256": json.loads(
                members["d1_7_batch_recording_manifest.json"].read_text(
                    encoding="utf-8-sig")).get("documents", {}).get("larkin_2015"),
            "how_to_obtain": ("NOT included. The paper is copyrighted; obtain it "
                              "from the publisher and confirm the document hash "
                              "above with react_review.retrieval.local_pdf."
                              "_pdf_text before replaying."),
            "retrieval_url": "https://doi.org/10.1056/NEJMoa1504030",
        },
        "environment": {
            "python": sys.version.split()[0],
            "note": ("record the resolved dependency versions of the checkout "
                     "used, so a replay that diverges can be told from a "
                     "recording that changed"),
        },
        "copyright": ("This bundle contains model responses quoting a "
                      "copyrighted paper verbatim. Distribution mode is a "
                      "decision for the project owner and is NOT settled by "
                      "building this file."),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in members.items():
            archive.write(path, name)
        archive.writestr("BUNDLE.json",
                         json.dumps(descriptor, indent=2, ensure_ascii=False))
    return {"path": str(target), "sha256": digest(target),
            "size_bytes": target.stat().st_size}


def check_storage_block(storage: dict) -> list[str]:
    """The manifest's own claim about where the recording is.

    Structural, and deliberately unsatisfiable by a word. `location: "foo"` used
    to close this, because the only test was that the string was non-empty.
    """
    problems: list[str] = []
    status = str(storage.get("status") or "")
    if status not in STATES:
        return [f"artifact_storage.status is {status!r}, and the only states "
                f"are {', '.join(STATES)}"]

    bundle = storage.get("bundle") or {}
    uri = bundle.get("uri")
    if status == BLOCKED:
        if uri:
            problems.append("status is blocked and a uri is recorded; if the "
                            "bundle exists, the status is at least "
                            "available_unverified")
        return problems

    if not isinstance(uri, str) or not uri.startswith(ALLOWED_SCHEMES):
        problems.append(
            f"bundle.uri {uri!r} is not an address anyone can act on "
            f"(expected one of {', '.join(ALLOWED_SCHEMES)})")
    if len(str(bundle.get("sha256") or "")) != 64:
        problems.append("bundle.sha256 is not a whole sha256")
    if not isinstance(bundle.get("size_bytes"), int) or bundle["size_bytes"] <= 0:
        problems.append("bundle.size_bytes is not a positive integer")
    if not bundle.get("access_mode"):
        problems.append("bundle.access_mode is unset: whether a reviewer can "
                        "actually reach this is part of the claim")
    if not bundle.get("copyright_restrictions"):
        problems.append("bundle.copyright_restrictions is unset, and the bundle "
                        "quotes a copyrighted paper verbatim")

    attestation = storage.get("independent_verification")
    if status == INDEPENDENTLY_VERIFIED:
        problems.extend(_check_attestation(attestation, bundle))
    elif attestation:
        problems.append("an attestation is recorded but the status does not "
                        "claim independent verification")
    return problems


def _check_attestation(attestation, bundle: dict) -> list[str]:
    """What another machine has to have said, for the claim to stand."""
    if not isinstance(attestation, dict) or not attestation:
        return ["status is independently_verified and there is no attestation, "
                "which is the one thing that state means"]
    problems: list[str] = []
    required = ("attestation_sha256", "verified_at", "repo_commit",
                "downloaded_sha256", "replay_backend_requests",
                "label_accuracy", "correct_rows", "total_rows", "output_sha256")
    for field in required:
        if attestation.get(field) in (None, ""):
            problems.append(f"the attestation omits {field}")
    if attestation.get("downloaded_sha256") != bundle.get("sha256"):
        problems.append("the attestation verified a different file than the one "
                        "the manifest points at")
    if attestation.get("replay_backend_requests") not in (0, "0"):
        problems.append("the verification reached a model, so it replayed "
                        "nothing")
    if attestation.get("correct_rows") != EXPECTED["correct_rows"]:
        problems.append(
            f"the verification scored {attestation.get('correct_rows')} correct "
            f"rows and the published result is {EXPECTED['correct_rows']}")
    return problems


def verify_bundle(path: Path, *, uri: str = "", expect_sha: str = "",
                  expect_size: int | None = None) -> tuple[list[str], dict]:
    """Everything checkable about a bundle FILE, before any replay."""
    problems: list[str] = []
    seen: dict = {}
    if uri and not uri.startswith(ALLOWED_SCHEMES):
        problems.append(f"uri {uri!r} is not an address anyone can act on")
    if not path.is_file():
        return problems + [f"{path} is not here: a uri that resolves to nothing "
                           "is not an available artifact"], seen

    seen["sha256"] = digest(path)
    seen["size_bytes"] = path.stat().st_size
    if expect_sha and seen["sha256"] != expect_sha:
        problems.append("the downloaded file is not the one the manifest names")
    if expect_size is not None and seen["size_bytes"] != expect_size:
        problems.append(f"the downloaded file is {seen['size_bytes']} bytes and "
                        f"the manifest says {expect_size}")
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            seen["members"] = sorted(names)
            for member in REQUIRED_MEMBERS:
                if member not in names:
                    problems.append(f"the bundle has no {member}, so the replay "
                                    "cannot be reproduced from it")
            if "BUNDLE.json" in names:
                descriptor = json.loads(archive.read("BUNDLE.json"))
                seen["descriptor"] = descriptor
                for member, published in (descriptor.get("members") or {}).items():
                    if member in names:
                        actual = hashlib.sha256(
                            archive.read(member)).hexdigest().upper()
                        if actual != published:
                            problems.append(f"{member} inside the bundle is not "
                                            "what BUNDLE.json says it is")
    except zipfile.BadZipFile:
        problems.append("the bundle is not readable as an archive")
    return problems, seen


def check_replay(scored_path: Path, descriptor: dict) -> list[str]:
    """The numbers a replay of this bundle must produce, exactly."""
    if not scored_path.is_file():
        return [f"no replay output at {scored_path}"]
    scored = json.loads(scored_path.read_text(encoding="utf-8-sig"))
    metrics, run = scored.get("metrics", {}), scored.get("run", {})
    telemetry = run.get("telemetry") or {}
    rows = scored.get("rows") or []
    correct = sum(1 for r in rows
                  if r.get("predicted_label") == r.get("expected_label"))
    expected = (descriptor or {}).get("expected") or EXPECTED

    problems: list[str] = []

    def compare(name, actual):
        if name in expected and actual != expected[name]:
            problems.append(f"{name}: the replay gives {actual} and the bundle "
                            f"declares {expected[name]}")

    compare("backend_requests", telemetry.get("backend_requests"))
    compare("correct_rows", correct)
    compare("total_rows", len(rows))
    compare("label_accuracy", metrics.get("label_accuracy"))
    compare("silent_release_count",
            (metrics.get("safety") or {}).get("silent_release_count"))
    compare("identity_wrong_released",
            ((metrics.get("target") or {}).get("gold") or {}).get(
                "identity_wrong_released"))
    compare("scope_wrong_released_count",
            (metrics.get("scope") or {}).get("scope_wrong_released_count"))

    hits = telemetry.get("cache_hits")
    total = (expected.get("extraction_cache_hits", 0)
             + expected.get("semantic_cache_hits", 0))
    if hits != total:
        problems.append(f"cache hits: the replay served {hits} and the bundle "
                        f"declares {total} ({expected.get('extraction_cache_hits')} "
                        f"extraction + {expected.get('semantic_cache_hits')} "
                        "semantic)")
    if telemetry.get("cache_misses"):
        problems.append(f"{telemetry['cache_misses']} cache miss(es): a replay "
                        "that misses is not replaying this recording")
    published = (descriptor or {}).get("expected_scored_sha256")
    if published and digest(scored_path) != published:
        problems.append("the replay output does not hash to what the bundle "
                        "declares, so something upstream of the numbers differs")
    return problems


def attest(bundle: Path, scored: Path, *, uri: str, seen: dict) -> dict:
    """The receipt another machine leaves behind. Hashed, so it cannot drift."""
    scored_body = json.loads(scored.read_text(encoding="utf-8-sig"))
    rows = scored_body.get("rows") or []
    body = {
        "bundle_uri": uri,
        "downloaded_sha256": seen.get("sha256"),
        "downloaded_size_bytes": seen.get("size_bytes"),
        "verified_at": subprocess.run(
            ["git", "log", "-1", "--format=%cI"], cwd=REPO, capture_output=True,
            text=True).stdout.strip(),
        "repo_commit": _commit(),
        "python": sys.version.split()[0],
        "replay_backend_requests": (
            (scored_body.get("run", {}).get("telemetry") or {}
             ).get("backend_requests")),
        "label_accuracy": scored_body.get("metrics", {}).get("label_accuracy"),
        "correct_rows": sum(1 for r in rows if r.get("predicted_label")
                            == r.get("expected_label")),
        "total_rows": len(rows),
        "output_sha256": digest(scored),
    }
    body["attestation_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return body


def attestation_is_intact(attestation: dict) -> bool:
    """An attestation that was edited after it was signed is not one."""
    body = {k: v for k, v in attestation.items() if k != "attestation_sha256"}
    return attestation.get("attestation_sha256") == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="verify_d1_7_bundle")
    ap.add_argument("--build", type=Path)
    ap.add_argument("--verify", type=Path)
    ap.add_argument("--scored", type=Path,
                    help="the replay output produced from this bundle")
    ap.add_argument("--uri", default="")
    ap.add_argument("--attest", type=Path)
    args = ap.parse_args(argv)

    if args.build:
        made = build(args.build)
        print(f"[bundle] {made['path']}")
        print(f"         sha256 {made['sha256']}")
        print(f"         {made['size_bytes']} bytes")
        print("\nBuilding is not publishing. Where this goes, and whether it may "
              "be published at all, is not decided here.")
        return 0

    if not args.verify:
        ap.error("one of --build or --verify is required")

    problems, seen = verify_bundle(args.verify, uri=args.uri)
    descriptor = seen.get("descriptor") or {}
    if args.scored:
        problems.extend(check_replay(args.scored, descriptor))
    else:
        problems.append("no --scored replay output given, so nothing was "
                        "reproduced; a bundle that merely unpacks proves "
                        "nothing")

    for problem in problems:
        print(f"  {problem}")
    print("BUNDLE " + ("VERIFIED" if not problems else
                       f"{len(problems)} problem(s)"))

    if not problems and args.attest:
        receipt = attest(args.verify, args.scored, uri=args.uri, seen=seen)
        args.attest.write_text(json.dumps(receipt, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[attestation] {args.attest} {receipt['attestation_sha256'][:16]}")
        print("This receipt is what lets the manifest say "
              "independently_verified — and only if it was produced somewhere "
              "other than the machine that made the recording.")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
