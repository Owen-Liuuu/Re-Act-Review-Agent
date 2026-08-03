"""Parser-stage accuracy: run the REAL ReviewParser on the review PDF and score
its long-table output against review_ground_truth.csv.

This fills the eval gap: run_full_accuracy skips the parser (uses answer-key
review values); this measures the parser itself. Slow (GLM, ~4 min) + costs
tokens, so drive it with --config.

    python eval/run_parser_accuracy.py --config configs/config.local.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import react_review.cli  # noqa: F401  prime the import graph
from react_review.audit import ToleranceTable, compare_values
from react_review.core.config import load_config
from react_review.core.enums import AuditLabel
from react_review.csv_io import load_included_studies
from react_review.dkb import FieldResolver, load_runtime_knowledge
from react_review.parser.review_parser import ReviewParser
from react_review.pipeline.factory import _create_llm_backend
from react_review.study_match import apply_modality_disambiguation, resolve_studies

BENCH = Path(__file__).resolve().parent / "benchmark"
ROOT = BENCH.parent.parent


def _key(sid: str, group: str, ft: str) -> tuple[str, str, str]:
    """The join key for scoring.

    ``-`` (study-level) and ``all`` (one combined cohort) are folded together
    because the ground truth writes one and the parser may write the other for
    the same cell. An EMPTY group is NOT folded in: it means the parser could not
    place the cohort, and hiding that here would conceal the very failure the
    cohort registry exists to surface — it is scored separately below.
    """
    g = (group or "-").strip().lower()
    if g in ("-", "all"):
        g = "all"
    return (sid, g, ft)


def _load_gt() -> list[dict[str, str]]:
    with open(BENCH / "review_ground_truth.csv", encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r.get("study_id")]


async def _run(args) -> None:
    backend = _create_llm_backend(load_config(args.config))
    kb = load_runtime_knowledge(
        ROOT / "configs" / "knowledge.seed.json", ROOT / "configs" / "ontology")
    parser = ReviewParser(backend, FieldResolver(kb, backend=backend))

    print(f"parsing review PDF (slow) … {args.pdf}")
    parsed = await parser.parse(args.pdf, research_context=args.context)
    studies = load_included_studies(BENCH / "included_studies.csv")
    items, sid_map = resolve_studies(parsed.items, studies)     # canonicalise study_ids
    items = apply_modality_disambiguation(
        items, sid_map, kb, parsed.field_resolutions)           # CT EAT → eat_volume
    print(f"parser produced {len(items)} items")

    gt = {_key(r["study_id"], r.get("group"), r["field_type"]): r for r in _load_gt()}
    got = {_key(it.study_id, it.group, it.field_type): it for it in items}

    gt_keys, got_keys = set(gt), set(got)
    tp, fn, fp = gt_keys & got_keys, gt_keys - got_keys, got_keys - gt_keys

    tol = ToleranceTable()
    vmatch = 0
    mismatched: list[dict] = []
    for k in tp:
        it, g = got[k], gt[k]
        lab = compare_values(
            field_type=it.field_type, review_value=it.value, source_value=g.get("value"),
            review_unit=it.unit, source_unit=g.get("unit") or "",
            rel_tolerance=tol.rel_tolerance(it.field_type),
            sd_rel_tolerance=tol.sd_rel_tolerance(it.field_type)).label
        # numeric match OR (for text fields the numeric compare can't judge) exact string
        text_eq = str(it.value).strip().lower() == str(g.get("value") or "").strip().lower()
        if lab == AuditLabel.MATCH or text_eq:
            vmatch += 1
        else:
            mismatched.append({"study": k[0], "group": k[1], "field": k[2],
                               "parser_value": it.value, "gt_value": g.get("value")})

    stats = {
        "n_gt": len(gt_keys), "n_parser": len(got_keys), "n_matched": len(tp),
        "recall": len(tp) / len(gt_keys) if gt_keys else 0.0,
        "precision": len(tp) / len(got_keys) if got_keys else 0.0,
        "value_match": vmatch / len(tp) if tp else 0.0, "value_matched": vmatch,
        "missed": dict(Counter(k[2] for k in fn)),
        "spurious": dict(Counter(k[2] for k in fp)),
        "mismatched_values": mismatched,
        "parser_rows": [{"study_id": it.study_id, "group": it.group,
                         "field_type": it.field_type, "raw_field_name": it.raw_field_name,
                         "value": it.value, "unit": it.unit} for it in items],
        # Cohort health — the field accuracy above cannot see any of this.
        "cohorts_discovered": sorted({it.cohort_label for it in items if it.cohort_label}),
        "cohort_status": dict(Counter(it.cohort_status for it in items)),
        "unknown_cohort_rows": [
            {"study_id": it.study_id, "label": it.cohort_label,
             "field_type": it.field_type or it.raw_field_name}
            for it in items if it.cohort_status in ("unknown", "ambiguous")],
        "provenance_missing": 0,   # source-side; populated by run_full_accuracy
    }

    print("\n================ PARSER accuracy vs review_ground_truth ================")
    print(f"ground-truth rows      : {stats['n_gt']}")
    print(f"parser rows            : {stats['n_parser']}")
    print(f"field coverage (recall): {stats['recall'] * 100:5.1f}%  ({len(tp)}/{len(gt_keys)} found)")
    print(f"field precision        : {stats['precision'] * 100:5.1f}%  ({len(fp)} spurious/extra)")
    print(f"value match (aligned)  : {stats['value_match'] * 100:5.1f}%  ({vmatch}/{len(tp)})")
    print(f"\nmissed field_types     : {stats['missed']}")
    print(f"spurious field_types   : {stats['spurious']}")
    print("-- cohorts (the field metrics above are blind to these) --")
    print(f"  discovered           : {stats['cohorts_discovered']}")
    print(f"  status               : {stats['cohort_status']}")
    unknown = stats["unknown_cohort_rows"]
    print(f"  unplaced cohorts     : {len(unknown)}"
          + (f"  ← must not silently become 'all'" if unknown else "  (none)"))
    for row in unknown[:8]:
        print(f"      {row['study_id']}/{row['label']!r}: {row['field_type']}")
    print("========================================================================")

    if args.html:
        from react_review.report import render_parser_report
        args.html.write_text(render_parser_report(stats), encoding="utf-8")
        print(f"[html] {args.html.resolve()}")
    if args.out:
        import json
        args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {args.out.resolve()}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Parser accuracy vs review_ground_truth.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pdf", type=Path, default=BENCH / "raw" / "EAT_T1DM_SRMA.pdf")
    ap.add_argument("--context", default="EAT thickness/volume in T1DM vs healthy controls")
    ap.add_argument("--html", type=Path, default=None, help="write an HTML parser report")
    ap.add_argument("--out", type=Path, default=None, help="write a JSON stats file")
    asyncio.run(_run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
