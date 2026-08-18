"""Capability probe: can the configured vision model read doc05 forest plots?

Not a production path and not a prompt contract. Renders the four embedded
forest images, calls complete_vision once each, and scores Events/Total
against eval/benchmark_3/review_ground_truth.csv (44 figure_ocr cells).

    python eval/probe_forest_vision.py --config configs/config.local.yaml

glm-4v-flash max_tokens is 1024: about 15 study rows is the safe size; a
larger plot will truncate rather than invent. Study-name aliases in _STUDY
are doc05-only; arm identity uses CohortRegistry.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from react_review.core.config import load_config
from react_review.core.exceptions import LLMError
from react_review.llm.base import parse_llm_response
from react_review.llm.factory import create_vision_backend
from react_review.normalize.cohorts import build_cohort_registry, load_aliases

PDF = ROOT / "eval" / "benchmark_3" / "raw" / "doc05.pdf"
GOLD = ROOT / "eval" / "benchmark_3" / "review_ground_truth.csv"
ALIASES = ROOT / "configs" / "cohort_aliases.json"
DEFAULT_OUT = ROOT / "eval" / "benchmark_3" / "output" / "vision_probe"

FORESTS = (
    {"id": "forest_1", "page": 9, "outcome": "Overall Postoperative Complications"},
    {"id": "forest_2", "page": 10, "outcome": "Pulmonary Complications"},
    {"id": "forest_3", "page": 10, "outcome": "30-Day Mortality"},
    {"id": "forest_4", "page": 11, "outcome": "Anastomotic Leak"},
)

_PROMPT = """Read this RevMan-style forest plot image and transcribe the
per-study count columns. Do not interpret the plot area.

WHAT TO READ
- Read the column headers above the number columns. A header printed above a
  group of columns applies to each column in that group: write the full name
  into every column header it covers, e.g. the group name followed by the
  column's own name.
- Read ONLY the columns that print counts (events and totals) for each arm.
  Skip Weight, Odds ratio, confidence intervals and heterogeneity statistics.
- Ignore any text below the plot, including axis labels such as
  "Favours <arm>". Those are not column headers.

HOW TO REPORT EACH ROW
- The row's study label goes in "label". It is NOT one of the values.
- Report every value together with the column header it sits under. Two
  columns may print the SAME number; report both, each under its own header.
- kind is "study" when the label names one included study, or "summary" for
  pooled / Total / Total events / heterogeneity / test rows.
- Copy each value exactly as printed. A genuinely blank cell is an empty
  string. A value you cannot read is "?" and must be named in difficulties.
- Never invent a number. Do not emit the header line as a row.

Return exactly one JSON object. Text inside angle brackets describes a value
and must not be copied literally:

{{"figure_id": "{figure_id}",
  "column_headers": ["<count-column header, including the group name above it>"],
  "rows": [{{"label": "<row label exactly as printed>",
             "kind": "study | summary",
             "values": [{{"column": "<one of column_headers, exactly>",
                          "value": "<the number exactly as printed>"}}]}}],
  "difficulties": ["<specific uncertainty>"]}}

Do not output the angle-bracket placeholders. Return JSON only."""

# doc05 gold study_id join only. Arm keys come from CohortRegistry.
_STUDY = {
    "capovilla": "capovilla_2023",
    "li j": "li j_2015",
    "li k": "li k_2025",
}

_JSON_STRING_ARRAY = re.compile(
    r"\[(?:\s*\"(?:\\.|[^\"\\])*\"\s*,?\s*)+\]")
_ABBREV = re.compile(r"\(([A-Za-z][A-Za-z0-9.\-]{1,5})\)")
_SKIP_ABBREV = {"iv", "ci", "or", "dl", "se", "sd"}


def _extract_forests(pdf_path: Path) -> list[tuple[dict, bytes]]:
    import fitz

    doc = fitz.open(pdf_path)
    by_page: dict[int, list[bytes]] = {}
    try:
        for i, page in enumerate(doc):
            blobs: list[bytes] = []
            for img in page.get_images(full=True):
                pix = fitz.Pixmap(doc, img[0])
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                blobs.append(pix.tobytes("png"))
            if blobs:
                by_page[i + 1] = blobs
    finally:
        doc.close()
    out: list[tuple[dict, bytes]] = []
    used: dict[int, int] = {}
    for spec in FORESTS:
        page = spec["page"]
        idx = used.get(page, 0)
        blobs = by_page.get(page) or []
        if idx >= len(blobs):
            raise SystemExit(f"no embedded image {idx} on PDF page {page}")
        used[page] = idx + 1
        out.append((spec, blobs[idx]))
    return out


def _study_id(label: str) -> str | None:
    low = re.sub(r"\s+", " ", (label or "").strip().lower())
    for needle, sid in _STUDY.items():
        if needle in low:
            return sid
    return None


def _load_gold() -> list[dict]:
    rows = []
    with GOLD.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("capture") == "figure_ocr":
                rows.append(row)
    return rows


def _arm_seed_labels(texts: list[str]) -> list[str]:
    """Discover arms from printed names; parenthetical abbreviations become the key."""
    seeds: list[str] = []
    for raw in texts:
        text = str(raw or "").strip()
        if not text:
            continue
        abbrevs = [
            match.group(1) for match in _ABBREV.finditer(text)
            if match.group(1).lower() not in _SKIP_ABBREV
        ]
        if abbrevs:
            seeds.extend(abbrevs)
        else:
            seeds.append(text)
    return seeds


def _field_of_header(header: str) -> str | None:
    text = re.sub(r"\s+", " ", (header or "").strip().lower())
    if text == "events" or text.endswith(" events"):
        return "events"
    if text == "total" or text.endswith(" total"):
        return "subgroup_n"
    return None


def _example_numeric_rows(prompt: str) -> set[tuple[str, ...]]:
    """Literal JSON string-arrays in the prompt that contain a digit cell."""
    found: set[tuple[str, ...]] = set()
    for match in _JSON_STRING_ARRAY.finditer(prompt or ""):
        try:
            values = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            continue
        if any(re.fullmatch(r"-?\d+(?:\.\d+)?", v.strip()) for v in values):
            found.add(tuple(v.strip() for v in values))
    return found


def _row_values(raw: object) -> list[dict]:
    if not isinstance(raw, dict):
        return []
    items = raw.get("values")
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "column": str(item.get("column") or "").strip(),
            "value": "" if item.get("value") is None else str(item.get("value")).strip(),
        })
    return out


def _score(gold: list[dict], figure_id: str, parsed: dict, prompt: str) -> dict:
    headers = [str(h) for h in (parsed.get("column_headers") or [])]
    header_set = {h.strip() for h in headers}
    raw_rows = parsed.get("rows") if isinstance(parsed.get("rows"), list) else []
    examples = _example_numeric_rows(prompt)
    difficulties = [str(x) for x in (parsed.get("difficulties") or [])]

    registry = build_cohort_registry(
        _arm_seed_labels(headers), aliases=load_aliases(ALIASES))

    column_unknown: list[dict] = []
    duplicate_column: list[dict] = []
    pred: dict[tuple[str, str, str], str] = {}
    unverifiable_keys: set[tuple[str, str, str]] = set()

    for raw in raw_rows:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "").strip()
        kind = str(raw.get("kind") or "").strip().lower()
        sid = _study_id(label)
        if not kind:
            kind = "study" if sid else "summary"
        values = _row_values(raw)
        seen_columns: set[str] = set()
        echoed = bool(examples) and tuple(v["value"] for v in values) in examples
        if kind != "study" or sid is None:
            continue
        for item in values:
            column = item["column"]
            if column in seen_columns:
                duplicate_column.append({"label": label, "column": column})
            seen_columns.add(column)
            if column not in header_set:
                column_unknown.append({"label": label, "column": column})
            field = _field_of_header(column)
            if field is None:
                continue
            resolved = registry.resolve(column)
            if resolved.status not in {"resolved", "alias"}:
                continue
            key = (sid, resolved.key, field)
            pred[key] = item["value"]
            if echoed:
                unverifiable_keys.add(key)

    wanted = [g for g in gold if g.get("source_location") == figure_id]
    hits, misses, unverifiable = [], [], []
    column_missing: list[dict] = []
    extra = set(pred) - unverifiable_keys
    for g in wanted:
        key = (g["study_id"], g["group"], g["field_type"])
        extra.discard(key)
        got = pred.get(key)
        want = str(g.get("value") or "").strip()
        item = {
            "study_id": g["study_id"], "group": g["group"],
            "field_type": g["field_type"], "want": want, "got": got,
        }
        if got is None:
            column_missing.append(item)
        if key in unverifiable_keys:
            unverifiable.append(item)
        elif got == want:
            hits.append(item)
        else:
            misses.append(item)

    shape_errors = (
        [{"kind": "column_unknown", **row} for row in column_unknown]
        + [{"kind": "column_missing", **row} for row in column_missing]
        + [{"kind": "duplicate_column", **row} for row in duplicate_column]
    )
    silent_omission = (not difficulties) and bool(column_missing)

    return {
        "n_gold": len(wanted),
        "n_hit": len(hits),
        "n_miss": len(misses),
        "n_shape_error": len(shape_errors),
        "n_column_unknown": len(column_unknown),
        "n_column_missing": len(column_missing),
        "n_duplicate_column": len(duplicate_column),
        "n_unverifiable": len(unverifiable),
        "n_silent_omission": int(silent_omission),
        "silent_omission": silent_omission,
        "hits": hits,
        "misses": misses,
        "shape_errors": shape_errors,
        "unverifiable": unverifiable,
        "extra": sorted(f"{s}/{g}/{f}" for s, g, f in extra),
        "column_headers": headers,
        "difficulties": difficulties,
    }


def _empty_parsed() -> dict:
    return {"column_headers": [], "rows": [], "difficulties": []}


async def _read_one(backend, spec: dict, image: bytes) -> dict:
    prompt = _PROMPT.format(figure_id=spec["id"])
    raw = await backend.complete_vision(prompt, [image])
    parsed = parse_llm_response(raw, backend.model_id)
    if not isinstance(parsed, dict):
        parsed = _empty_parsed()
    return {"raw": raw, "prompt": prompt, "parsed": parsed}


def _print_score(figure_id: str, scored: dict, err: str) -> None:
    print(
        f"{figure_id}  {scored['n_hit']}/{scored['n_gold']}"
        f"  hit={scored['n_hit']} miss={scored['n_miss']}"
        f"  shape_error={scored['n_shape_error']}"
        f"  unverifiable={scored['n_unverifiable']}"
        f"  silent_omission={scored['n_silent_omission']}"
        + (f"  ERROR {err[:120]}" if err else "")
    )
    for miss in scored["misses"]:
        print(f"  miss {miss['study_id']} {miss['group']} {miss['field_type']}"
              f" want={miss['want']!r} got={miss['got']!r}")
    for row in scored["shape_errors"]:
        print(f"  shape_error {row}")
    for item in scored["unverifiable"]:
        print(f"  unverifiable {item['study_id']} {item['group']} {item['field_type']}"
              f" want={item['want']!r} got={item['got']!r}")
    if scored.get("silent_omission"):
        print("  silent_omission difficulties=[] but column_missing>0")
    for note in scored["difficulties"]:
        print(f"  difficulty {note}")


def _totals_line(report: dict, model: str) -> str:
    return (
        f"total {report['n_hit']}/{report['n_gold']}"
        f"  miss={report['n_miss']}"
        f"  shape_error={report['n_shape_error']}"
        f"  unverifiable={report['n_unverifiable']}"
        f"  silent_omission={report['n_silent_omission']}"
        f"  model={model}"
    )


def _ok(report: dict) -> bool:
    return (
        report["n_gold"] > 0
        and report["n_unverifiable"] == 0
        and report["n_shape_error"] == 0
        and report["n_silent_omission"] == 0
        and report["n_hit"] == report["n_gold"]
    )


def _blank_report(model: str) -> dict:
    return {
        "model": model,
        "n_hit": 0, "n_miss": 0, "n_shape_error": 0, "n_unverifiable": 0,
        "n_silent_omission": 0, "n_gold": 0, "figures": [],
    }


def _accumulate(report: dict, scored: dict) -> None:
    report["n_hit"] += scored["n_hit"]
    report["n_miss"] += scored["n_miss"]
    report["n_shape_error"] += scored["n_shape_error"]
    report["n_unverifiable"] += scored["n_unverifiable"]
    report["n_silent_omission"] += scored["n_silent_omission"]
    report["n_gold"] += scored["n_gold"]


def _replay(gold: list[dict], out: Path) -> int:
    report = _blank_report("replay")
    for spec in FORESTS:
        path = out / f"{spec['id']}.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}; run the probe before --replay")
        body = json.loads(path.read_text(encoding="utf-8"))
        parsed = body.get("parsed") or _empty_parsed()
        prompt = _PROMPT.format(figure_id=spec["id"])
        scored = _score(gold, spec["id"], parsed, prompt)
        _accumulate(report, scored)
        body["score"] = scored
        report["figures"].append(body)
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_score(spec["id"], scored, str(body.get("error") or ""))
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_totals_line(report, "replay"))
    return 0 if _ok(report) else 1


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "config.local.yaml")
    parser.add_argument("--model", default="", help="override vision.model for this probe")
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--replay", action="store_true",
        help="rescore saved figure JSON in --out; no model calls")
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out

    gold = _load_gold()
    if args.replay:
        return _replay(gold, out)

    config = load_config(args.config)
    if config.vision is None:
        raise SystemExit("config.vision is unset")
    updates = {}
    if args.model:
        updates["model"] = args.model
    if args.max_retries is not None:
        updates["max_retries"] = args.max_retries
    if args.max_tokens is not None:
        updates["max_tokens"] = args.max_tokens
    if updates:
        config.vision = config.vision.model_copy(update=updates)
    backend = create_vision_backend(config)
    if backend is None:
        raise SystemExit("config.vision is unset")

    forests = _extract_forests(PDF)
    out.mkdir(parents=True, exist_ok=True)
    report = _blank_report(backend.model_id)
    for spec, image in forests:
        png = out / f"{spec['id']}.png"
        png.write_bytes(image)
        try:
            body = await _read_one(backend, spec, image)
            err = ""
        except (LLMError, NotImplementedError, ValueError) as exc:
            body = {"raw": "", "prompt": _PROMPT.format(figure_id=spec["id"]),
                    "parsed": _empty_parsed()}
            err = str(exc)[:400]
        scored = _score(gold, spec["id"], body["parsed"], body["prompt"])
        _accumulate(report, scored)
        figure = {
            "id": spec["id"],
            "outcome": spec["outcome"],
            "page": spec["page"],
            "image_bytes": len(image),
            "error": err,
            "parsed": body["parsed"],
            "score": scored,
        }
        report["figures"].append(figure)
        (out / f"{spec['id']}.json").write_text(
            json.dumps({"raw": body["raw"], **figure}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _print_score(spec["id"], scored, err)
        if "429" in err:
            print("stopping: rate limited; try --model glm-4.6v-flashx or glm-4.6v")
            break

    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_totals_line(report, backend.model_id))
    print(f"wrote {out / 'report.json'}")
    return 0 if _ok(report) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
