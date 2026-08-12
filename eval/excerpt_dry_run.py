"""What a batched run WOULD send, computed by the code that would send it.

No model, no recording, no cost — and no approximation either. The first
coverage figure this project produced came from a script that guessed the
selector's inputs (`field_type.replace("_", " ")` as the target term), which
makes the number a property of the script rather than of the run. The selector's
terms come from the knowledge base's concept and variants, from the target
contract's raw field name, and from the review's own column label; get any of
them wrong and a different set of blocks wins.

So this builds the real review items the harness builds, the real Collector with
the real knowledge base and the real contract, groups the claims with the real
grouping, and calls the real selector — then stops, before anything would be
asked of a model.

    python eval/excerpt_dry_run.py --benchmark eval/benchmarks/melanoma_checkpoint_2017 \
        --benchmark-profile phase8_batch_v3_profile.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from react_review.agents.collector import _claim_id, _document_sha256  # noqa: E402
from react_review.csv_io import load_included_studies  # noqa: E402
from react_review.dkb import load_runtime_knowledge  # noqa: E402
from react_review.eval_excerpt import (  # noqa: E402
    assess,
    benchmark_reviews,
    dry_run_collector,
    load_gold,
    planned_batches,
)
from react_review.eval_profile import load_profile  # noqa: E402
from react_review.retrieval.local_pdf import _pdf_text  # noqa: E402
from react_review.schemas.batch import BatchReadingRecord, ExcerptProvenance  # noqa: E402
from react_review.tools.extract_source import (  # noqa: E402
    SELECTION_METHOD_ID,
    SELECTION_VERSION,
    select_excerpt,
)

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="excerpt_dry_run",
        description="Compute what a batched run would send, without sending it.")
    ap.add_argument("--benchmark", type=Path,
                    default=REPO / "eval/benchmarks/melanoma_checkpoint_2017")
    ap.add_argument("--benchmark-profile", default="phase8_batch_v3_profile.json")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    benchmark = args.benchmark
    rows = list(csv.DictReader(
        (benchmark / "audit_template.csv").open(encoding="utf-8-sig")))
    profile = load_profile(benchmark, args.benchmark_profile,
                           answer_key_ids=[r["audit_id"] for r in rows])
    contract = profile.run_contract
    if contract is None or not contract.batching:
        print(f"{args.benchmark_profile} does not route anything to a batch; "
              "there are no windows to judge")
        return 2

    studies = load_included_studies(benchmark / "selected_studies.csv")
    paths = {s.study_id: benchmark / s.source_pdf for s in studies if s.source_pdf}

    collector = dry_run_collector(contract, load_runtime_knowledge(
        REPO / "configs/knowledge.seed.json", REPO / "configs/ontology"))
    reviews = benchmark_reviews(rows, profile.targets)
    readings: list[BatchReadingRecord] = []
    texts: dict[str, str] = {}
    for study_id in sorted({r.study_id for r in reviews}):
        path = paths.get(study_id)
        if path is None or not path.is_file():
            print(f"[skip] {study_id}: no local source paper at {path}")
            continue
        text = texts.setdefault(study_id, _pdf_text(path))
        for group in planned_batches(
                collector, [r for r in reviews if r.study_id == study_id],
                contract.extraction_routes["value"]):
            field_type = group.key.field_type
            concept = collector._concept_for(field_type)
            variants = collector._concept_variants_for(field_type)
            target = concept or group.key.raw_field_name or field_type
            excerpt, spans = select_excerpt(
                text, target=target, raw_label=group.key.raw_field_name,
                field_type=field_type, variants=variants)
            readings.append(BatchReadingRecord(
                execution_id=f"dry:{study_id}/{field_type}/{group.shape}",
                study_id=study_id, field_type=field_type,
                target_shape=group.shape,
                claim_ids=[_claim_id(c) for c in group.claims],
                excerpt_provenance=ExcerptProvenance(
                    windowed=len(excerpt) != len(text), source_chars=len(text),
                    excerpt_chars=len(excerpt), spans=spans,
                    selection_method_id=SELECTION_METHOD_ID,
                    selection_version=SELECTION_VERSION)))

    if profile.excerpt_gold_path is None:
        print("this profile publishes no excerpt gold; nothing can be judged")
        return 2

    report = assess(readings, load_gold(profile.excerpt_gold_path),
                    lambda s: texts.get(s) or None,
                    sha_for=lambda s: _document_sha256(texts.get(s) or ""))

    print(f"benchmark: {benchmark.name} [{args.benchmark_profile}]")
    print(f"contract : {contract.profile_id} "
          f"[value={contract.extraction_routes.get('value')}]")
    print(f"batches  : {len(readings)} computed without asking a model")
    if not report.assessable:
        print(f"excerpt  : NOT ASSESSABLE — {report.reason}")
        return 1
    counts = report.tally
    print(f"excerpt  : {counts.gold_covered_batches}/"
          f"{counts.gold_text_assessable_batches} gold-assessable batches "
          f"covered, {counts.gold_missing_batches} missing "
          f"({counts.windowed_batches} windowed)")
    for entry in report.batches:
        missed = [w for w in entry["witnesses"] if w["outcome"] != "covered"]
        state = "all covered" if not missed else json.dumps(missed)
        print(f"  {entry['batch_id']:46s} windowed={entry['windowed']!s:5s} "
              f"{entry['excerpt_chars']:>6d}/{entry['source_chars']:<6d} {state}")
    if report.unjudged_run_batches:
        print(f"  unjudged by the key: {list(report.unjudged_run_batches)}")
    if args.json:
        args.json.write_text(json.dumps(report.as_dict(), indent=2,
                                        ensure_ascii=False), encoding="utf-8")
        print(f"[json] {args.json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
