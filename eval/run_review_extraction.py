"""Live Review Extraction score vs benchmark_3 gold.

Default gold is ``eval/benchmark_3/review_ground_truth.csv``. Reports go to
``eval/benchmark_3/output/``. This is not the frozen table-capture A/B gate.

    python eval/run_review_extraction.py --config configs/config.local.yaml
    python eval/run_review_extraction.py --items eval/benchmark_3/output/parser_items.json
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import react_review.cli  # noqa: F401  prime the import graph

from eval.review_extraction_score import (
    load_gold_csv,
    render_html,
    score_extraction,
)
from react_review.core.config import load_config
from react_review.csv_io import load_included_studies
from react_review.dkb import FieldResolver, load_runtime_knowledge
from react_review.hitl import AutoContinue, RunJournal, StepReporter
from react_review.llm.factory import create_llm_backend, create_vision_backend
from react_review.parser.review_parser import ReviewParser
from react_review.study_match import resolve_studies

BENCH = ROOT / "eval" / "benchmark_3"


def _studies_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    for name in ("studies_worksheet.csv", "included_studies.csv"):
        path = BENCH / name
        if path.is_file():
            return path
    return None


def _hits_from_journal(journal_dir: Path) -> tuple[list[dict], dict, list[str], list[dict]]:
    steps = journal_dir / "steps"
    hits: list[dict] = []
    lens: dict = {}
    dropped: list[str] = []
    captured: list[dict] = []
    if not steps.is_dir():
        return hits, lens, dropped, captured
    for path in sorted(steps.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stage = str(body.get("stage") or "")
        payload = body.get("payload") or {}
        if stage == "review_lens":
            lens = payload.get("lens") or {}
        elif stage == "evidence_localize":
            hits = list(payload.get("displays") or [])
        elif stage == "forest_ocr":
            for table in payload.get("tables") or []:
                captured.append({
                    "table_id": table.get("table_id") or table.get("id") or "",
                    "caption": table.get("caption") or "",
                    "outcome": table.get("outcome") or "",
                    "display_kind": table.get("display_kind") or "forest_plot",
                    "n_rows": len(table.get("rows") or []),
                    "rows": table.get("rows") or [],
                    "difficulties": list(table.get("difficulties") or []),
                    "checksum_failures": list(table.get("checksum_failures") or []),
                    "checksum_printed_values": list(
                        table.get("checksum_printed_values") or []),
                    "checksum_column_sums": dict(
                        table.get("checksum_column_sums") or {}),
                    "capture_path": table.get("capture_path") or "",
                    "image_bytes": int(table.get("image_bytes") or 0),
                    "image_page": int(table.get("image_page") or 0),
                    "image_xref": int(table.get("image_xref") or 0),
                    "review_required": bool(table.get("review_required") or False),
                })
        elif stage == "claim_origin":
            dropped = list(payload.get("dropped_non_source") or [])
    return hits, lens, dropped, captured


def _write_reports(out_dir: Path, stats: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "review_extraction.json"
    html_path = out_dir / "review_extraction.html"
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(stats), encoding="utf-8")
    items_path = out_dir / "parser_items.json"
    items_path.write_text(
        json.dumps(stats.get("parser_rows") or [], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[json] {json_path}")
    print(f"[html] {html_path}")
    print(f"[items] {items_path}")


def _print_summary(stats: dict) -> None:
    loc = stats.get("localize") or {}
    table = stats.get("table_text") or {}
    forest = stats.get("figure_ocr") or {}
    claims = stats.get("claims") or {}
    outcome = (stats.get("join_diagnosis") or claims.get("join_diagnosis") or {}).get(
        "extraction_outcome") or {}
    print("\n================ Review Extraction vs gold ================")
    print(f"gold cells              : {stats.get('n_gold')}")
    print(f"parser cells            : {stats.get('n_parser')}")
    print(f"localize recall         : {loc.get('recall', 0) * 100:5.1f}%"
          f"  ({len(loc.get('recalled') or [])}/{loc.get('n_expected', 0)} displays)")
    print(f"Table 1 recall          : {table.get('recall', 0) * 100:5.1f}%"
          f"  ({table.get('n_matched', 0)}/{table.get('n_gt', 0)})")
    print(f"Forest recall           : {forest.get('recall', 0) * 100:5.1f}%"
          f"  ({forest.get('n_matched', 0)}/{forest.get('n_gt', 0)})")
    integrity = stats.get("integrity") or forest.get("integrity") or {}
    print(f"raw_accuracy            : {integrity.get('raw_accuracy', 0)}/"
          f"{integrity.get('raw_accuracy_denom', 0)}")
    print(f"detected_error          : {integrity.get('detected_error', 0)}")
    print(f"released_wrong          : {integrity.get('released_wrong', 0)}"
          "  (ship gate is 0)")
    print(f"all-cell recall         : {claims.get('recall', 0) * 100:5.1f}%")
    print(f"precision               : {claims.get('precision', 0) * 100:5.1f}%")
    print(f"value match (aligned)   : {claims.get('value_match', 0) * 100:5.1f}%")
    print(f"checksum_failed         : {outcome.get('checksum_failed', 0)}")
    print(f"not_extractable         : {outcome.get('not_extractable', 0)}")
    print(f"fabricated              : {outcome.get('fabricated', 0)}")
    print(f"missing                 : {outcome.get('missing', 0)}")
    missed = loc.get("missed") or []
    if missed:
        print("missed displays         : "
              + ", ".join(m.get("source_location", "") for m in missed))
    pooled = loc.get("pooled_marked_on") or []
    if pooled:
        print("GRADE/pooled marked ON  : "
              + ", ".join((p.get("caption") or p.get("display_id") or "")[:60]
                          for p in pooled))
    print("==========================================================\n")
    forests = [
        c for c in (stats.get("captured_tables") or [])
        if (c.get("display_kind") or "") == "forest_plot"
    ]
    if forests:
        print("Forest capture path")
        for c in forests:
            print(
                f"  {c.get('table_id')}: path={c.get('capture_path') or '-'} "
                f"bytes={c.get('image_bytes', 0)} page={c.get('image_page', 0)} "
                f"xref={c.get('image_xref', 0)} "
                f"{(c.get('outcome') or c.get('caption') or '')[:70]}")
        print()
    print("Forest recall near 0 is expected until OCR returns a grid.")
    print("Table 1 is the honest text-only extraction score.")


async def _parse(args):
    config = load_config(args.config)
    backend = create_llm_backend(config)
    vision = create_vision_backend(config)
    kb = load_runtime_knowledge(
        ROOT / "configs" / "knowledge.seed.json", ROOT / "configs" / "ontology")
    run_id = datetime.now().strftime("rx_%Y%m%d_%H%M%S")
    journal_dir = args.out / "journal" / run_id
    reporter = StepReporter(
        run_id, gate=AutoContinue(), journal=RunJournal(journal_dir))
    parser = ReviewParser(
        backend,
        FieldResolver(kb, backend=backend, write_back=False),
        reporter=reporter,
        checklist=None,
        max_chars=args.max_chars,
        vision_backend=vision,
    )
    print(f"parsing review PDF (slow) … {args.pdf}")
    parsed = await parser.parse(args.pdf, research_context=args.context)
    studies_path = _studies_path(args.studies)
    items = parsed.items
    if studies_path is not None:
        studies = load_included_studies(studies_path)
        items, _ = resolve_studies(items, studies)
        print(f"canonicalised study ids from {studies_path.name}")
    print(f"parser produced {len(items)} items")
    captured = [
        {"table_id": t.table_id, "caption": t.caption, "outcome": t.outcome,
         "display_kind": t.display_kind, "n_rows": len(t.rows),
         "difficulties": list(t.difficulties),
         "checksum_failures": list(t.checksum_failures),
         "checksum_printed_values": list(t.checksum_printed_values),
         "checksum_column_sums": dict(t.checksum_column_sums),
         "capture_path": t.capture_path, "image_bytes": t.image_bytes,
         "image_page": t.image_page, "image_xref": t.image_xref,
         "review_required": t.review_required}
        for t in parsed.tables.tables
    ]
    hits, lens, dropped, _from_journal = _hits_from_journal(journal_dir)
    return items, journal_dir, captured, hits, lens, dropped


def _load_items(path: Path) -> list[dict]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(body, dict):
        body = body.get("parser_rows") or body.get("items") or []
    if not isinstance(body, list):
        raise SystemExit(f"{path} is not a list of parser items")
    return body


async def _run(args) -> None:
    gold = load_gold_csv(args.gold)
    captured: list[dict] = []
    hits: list[dict] = []
    lens: dict = {}
    dropped: list[str] = []
    if args.items:
        items = _load_items(args.items)
        print(f"scoring saved items … {args.items} ({len(items)} rows)")
        if args.journal:
            hits, lens, dropped, captured = _hits_from_journal(args.journal)
    else:
        if args.config is None:
            raise SystemExit("--config is required unless --items is set")
        items, journal_dir, captured, hits, lens, dropped = await _parse(args)
        print(f"[journal] {journal_dir}")
    stats = score_extraction(
        gold, items, hits=hits, captured=captured,
        dropped_non_source=dropped, lens=lens,
    )
    stats["gold"] = str(args.gold)
    stats["pdf"] = str(args.pdf)
    stats["captured_tables"] = captured
    _write_reports(args.out, stats)
    _print_summary(stats)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Score Review Extraction against review_ground_truth.csv.")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--pdf", type=Path, default=BENCH / "raw" / "doc05.pdf")
    ap.add_argument("--gold", type=Path, default=BENCH / "review_ground_truth.csv")
    ap.add_argument("--studies", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=BENCH / "output")
    ap.add_argument("--context", default="")
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument("--items", type=Path, default=None,
                    help="skip the LLM and score a previous parser_items.json")
    ap.add_argument("--journal", type=Path, default=None,
                    help="optional journal dir when scoring --items")
    asyncio.run(_run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
