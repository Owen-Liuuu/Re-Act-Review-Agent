"""Copy the recordings a run may reuse into a NEW cache, and hash both ends.

A recording run must not write into a published cache: the artifact a previous
phase is described by would then contain entries that phase never made. So the
run uses a new file, and this puts the entries it is allowed to replay into it.

Copied by KEY, never wholesale. The keys come from the preflight's expected
plan and trace, so what is seeded is exactly what the run will look up and
nothing else — a whole-file copy would carry recordings from other contracts
into a cache whose name says otherwise.

Both ends are hashed. The source hash says which published recording this came
from; the destination hash is what the recording manifest pins BEFORE the run,
so that what the run added can be told from what it started with.

    python eval/seed_extraction_cache.py \
        --from output/baselines/melanoma_checkpoint_2017/phase7_extraction_cache.json \
        --to   output/baselines/melanoma_checkpoint_2017/phase8_batch_extraction_cache.json \
        --keys-from output/d1_7_preflight.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_review.tools.extraction_cache import ExtractionCache  # noqa: E402


def file_digest(path: Path) -> str:
    return (hashlib.sha256(path.read_bytes()).hexdigest().upper()
            if path.is_file() else "")


def keys_to_seed(report: dict) -> list[str]:
    """Every key the run is allowed to replay: the arm-identity lookups.

    Not the batch slots. Those must be empty — the whole point of the recording
    is that the batch route is new, and seeding one would replay an answer into
    a run that believes it asked for it.
    """
    return [entry["cache_key"] for entry in report.get("v4_trace") or ()
            if entry.get("hit")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="seed_extraction_cache",
        description="Seed a new extraction cache with named keys only.")
    ap.add_argument("--from", dest="source", type=Path, required=True)
    ap.add_argument("--to", dest="target", type=Path, required=True)
    ap.add_argument("--keys-from", type=Path, required=True,
                    help="a preflight --json report; its v4 hits are the keys")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a target that already exists")
    args = ap.parse_args(argv)

    if args.target.exists() and not args.force:
        raise SystemExit(
            f"{args.target} already exists. A recording that starts from an "
            "unknown cache cannot say what it added; delete it deliberately or "
            "pass --force")

    report = json.loads(args.keys_from.read_text(encoding="utf-8-sig"))
    if report.get("preflight") != "pass":
        print("[warn] the preflight report says it did not pass; seeding what "
              "it found anyway, and the preflight must be re-run against the "
              "seeded cache before any recording")

    source = ExtractionCache(args.source)
    keys = keys_to_seed(report)
    missing = [k for k in keys if source.get(k) is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} key(s) named by the report are not in {args.source}. "
            "Seeding a partial set would produce a cache that looks complete "
            "and is not")

    args.target.parent.mkdir(parents=True, exist_ok=True)
    seeded = ExtractionCache(args.target)
    for key in keys:
        seeded.put(key, source.get(key), model_id=source.model_id)
    seeded.save()

    print(f"source    : {args.source}")
    print(f"            sha256 {file_digest(args.source)[:32]} "
          f"({len(source)} entries)")
    print(f"seeded    : {args.target}")
    print(f"            sha256 {file_digest(args.target)[:32]} "
          f"({len(seeded)} entries, from {len(set(keys))} distinct keys)")
    print(f"model_id  : {source.model_id}")
    print("\nRe-run the preflight against the SEEDED cache before recording. "
          "The keys were computed against the source, and a cache is only the "
          "right one if the run's own lookups say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
