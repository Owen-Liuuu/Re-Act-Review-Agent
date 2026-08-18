"""Score Review Extraction claims against a long-table gold CSV.

Join is ``study × group × field × display``. Gold and parser pass through the
same family functions. Display identity is a printed Figure/Table number when
the caption has one; otherwise the nth evidence-chain forest plot (``forest_1``).
Sequential localize ids (``figure_2``) are never the join key.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping

from react_review.audit import ToleranceTable, compare_values
from react_review.core.enums import AuditLabel
from react_review.normalize.numeric import primary_number
from react_review.parser.review_extraction.schemas import DisplayHit, ReviewClaim
from react_review.schemas.evidence import ReviewDataItem

ClaimLike = ReviewDataItem | Mapping[str, Any]

_POOLED = re.compile(
    r"\b(grade|prisma|summary of findings|overall effect|pooled odds|"
    r"pooled or|weight\s*%)\b",
    re.I,
)
_FIGURE = re.compile(r"(?:figure|fig\.?)\s*(\d+(?:\.\d+)+|\d+)", re.I)
_TABLE = re.compile(r"table\s*([\d.]+)", re.I)
_FOREST_ID = re.compile(r"^forest_(\d+)$", re.I)

# Both field_type and raw_field_name go through this table. Identity entries
# keep gold tokens stable; synonyms collapse parser names onto the same token.
_FIELD_CANON: dict[str, str] = {
    "events": "events",
    "event": "events",
    "event_count": "events",
    "n_events": "events",
    "subgroup_n": "subgroup_n",
    "total": "subgroup_n",
    "n": "subgroup_n",
    "n_mie": "subgroup_n",
    "n_oe": "subgroup_n",
    "publication_year": "publication_year",
    "year": "publication_year",
    "pub_year": "publication_year",
    "country": "country",
    "study_design": "study_design",
    "design": "study_design",
    "age": "age",
}


def load_gold_csv(path: Path | str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [row for row in csv.DictReader(fh) if (row.get("study_id") or "").strip()]


def capture_family(row: Mapping[str, Any], item: ClaimLike | None = None) -> str:
    raw = str(row.get("capture") or "").strip().lower()
    if raw in {"table_text", "figure_ocr"}:
        return raw
    kind = _get(item, "display_kind") if item is not None else str(row.get("display_kind") or "")
    loc = str(row.get("source_location") or "").lower()
    if kind == "forest_plot" or loc.startswith("forest_") or "figure" in loc:
        return "figure_ocr"
    return "table_text"


def _field_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").strip().lower()).strip("_")


def field_family(
    field_type: str = "",
    raw_field_name: str = "",
    *,
    unmapped: list[dict[str, str]] | None = None,
) -> str:
    """Map gold and parser field names through one table.

    Canonical hits are silent. Substring fallback is recorded on ``unmapped``.
    """
    for source, raw in (("field_type", field_type), ("raw_field_name", raw_field_name)):
        token = _field_token(raw)
        if token in _FIELD_CANON:
            return _FIELD_CANON[token]
    blob = f"{field_type} {raw_field_name}".strip().lower()
    fallback = ""
    if "event" in blob:
        fallback = "events"
    elif re.search(r"\btotal\b", blob) or re.match(r"n\s*(mie|oe|t1dm|control)?\b", blob):
        fallback = "subgroup_n"
    elif "country" in blob or blob in {"region", "location"}:
        fallback = "country"
    elif "design" in blob:
        fallback = "study_design"
    elif re.search(r"\bage\b", blob):
        fallback = "age"
    else:
        fallback = _field_token(field_type or raw_field_name)
    if unmapped is not None:
        unmapped.append({
            "field_type": field_type,
            "raw_field_name": raw_field_name,
            "mapped_to": fallback,
            "via": "substring_fallback",
        })
    return fallback


def _arm_from_text(text: str) -> str:
    blob = (text or "").lower().replace("_", " ")
    if re.search(r"\bmie\b|minimally\s+invasive", blob):
        return "mie"
    if re.search(r"\boe\b|open\s+esophag", blob):
        return "oe"
    if "t1dm" in blob or "type 1" in blob:
        return "t1dm"
    if re.search(r"\bcontrols?\b", blob):
        return "control"
    return ""


def group_family(
    group: str = "",
    cohort_label: str = "",
    cohort_status: str = "",
    raw_field_name: str = "",
) -> str:
    """Arm identity. Column headers are tried before folding ``all`` / ``-``.

    That recovers Table 1 ``N MIE`` / ``N OE`` rows whose group landed on
    ``all``. Forest ``Events`` headers carry no arm; those stay ``all``.
    """
    header_arm = _arm_from_text(raw_field_name)
    if header_arm:
        return header_arm
    g = (group or "-").strip().lower()
    if cohort_status == "not_applicable" or g in {"-", "all"}:
        return "all"
    if g in {"mie", "oe", "t1dm", "control"}:
        return g
    label_arm = _arm_from_text(f"{g} {cohort_label}")
    if label_arm:
        return label_arm
    blob = f"{g} {cohort_label}".lower().replace("_", " ")
    return re.sub(r"[^a-z0-9]+", "_", blob).strip("_") or "all"


def outcome_family(outcome: str = "", source_location: str = "", capture: str = "") -> str:
    if (capture or "").strip().lower() == "table_text":
        return "characteristics"
    blob = f"{outcome} {source_location}".lower()
    if "anastomotic" in blob or re.search(r"\bleak\b", blob):
        return "anastomotic_leak"
    if "mortal" in blob:
        return "30_day_mortality"
    if "pulmonary" in blob:
        return "pulmonary"
    if "overall" in blob or "postoperative complication" in blob:
        return "overall_complications"
    if "characteristic" in blob:
        return "characteristics"
    return re.sub(r"[^a-z0-9]+", "_", (outcome or "").lower()).strip("_")


def display_family(*parts: str) -> str:
    """Printed Figure/Table number, or an ordinal forest id (``forest_1``).

    Sequential localize ids (``figure_2``) are not passed in. A caption with
    no printed number returns empty so the caller can use the nth
    evidence-chain forest plot.
    """
    for part in parts:
        match = _FOREST_ID.match((part or "").strip())
        if match:
            return f"forest_{int(match.group(1))}"
    blob = " ".join(p for p in parts if p)
    blob = re.sub(r"\s+", " ", blob).strip()
    if not blob:
        return ""
    match = _FIGURE.search(blob)
    if match:
        return "figure " + match.group(1).strip(".")
    match = _TABLE.search(blob)
    if match:
        return "table " + match.group(1).strip(".")
    return ""


def caption_index(
    hits: Iterable[DisplayHit | Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """display_id / table_id → caption and page_hint from localize hits."""
    out: dict[str, dict[str, str]] = {}
    for hit in hits:
        display_id = str(_get(hit, "display_id") or _get(hit, "id") or "").strip()
        if not display_id:
            continue
        out[display_id] = {
            "caption": str(_get(hit, "caption") or ""),
            "page_hint": str(_get(hit, "page_hint") or ""),
        }
    return out


def gold_display_catalog(gold: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for row in gold:
        loc = (row.get("source_location") or "").strip()
        family = display_family(loc)
        if not family or family in seen:
            continue
        seen[family] = {
            "display_family": family,
            "source_location": loc,
            "outcome": str(row.get("outcome") or ""),
            "capture": capture_family(row),
        }
    return list(seen.values())


def forest_ordinal_map(
    hits: Iterable[DisplayHit | Mapping[str, Any]] = (),
    items: Iterable[ClaimLike] = (),
) -> dict[str, str]:
    """``display_id`` / ``table_id`` → ``forest_1``, ``forest_2``, … in order.

    Evidence-chain forest hits first. If none, first-seen forest ``table_id``s
    on parser items, in encounter order.
    """
    ordered: list[str] = []

    def _add(display_id: str) -> None:
        token = (display_id or "").strip()
        if token and token not in ordered:
            ordered.append(token)

    for hit in hits:
        if str(_get(hit, "kind") or "") != "forest_plot":
            continue
        if not bool(_get(hit, "evidence_chain")):
            continue
        _add(str(_get(hit, "display_id") or _get(hit, "id") or ""))
    if not ordered:
        for item in items:
            if str(_get(item, "display_kind") or "") != "forest_plot":
                continue
            _add(str(_get(item, "table_id") or ""))
    return {did: f"forest_{i}" for i, did in enumerate(ordered, 1)}


def resolve_display(
    *,
    source_location: str = "",
    table_id: str = "",
    outcome: str = "",
    captions: Mapping[str, Mapping[str, str]] | None = None,
    catalog: Iterable[Mapping[str, str]] = (),
    forest_map: Mapping[str, str] | None = None,
    unresolved: list[dict[str, str]] | None = None,
) -> str:
    """Printed number if present; else nth evidence-chain forest plot."""
    del catalog  # join is printed-or-ordinal; outcome-fuzzy is not a display id
    hit = (captions or {}).get(table_id) or {}
    caption = str(hit.get("caption") or "")
    page_hint = str(hit.get("page_hint") or "")
    printed = display_family(source_location, caption, outcome)
    if printed:
        return printed
    ordinal = (forest_map or {}).get(table_id) or ""
    if ordinal:
        return ordinal
    token = (caption or outcome or table_id or "unknown")[:40]
    family = f"unresolved:{token}"
    if unresolved is not None:
        unresolved.append({
            "table_id": table_id,
            "caption": caption[:80],
            "outcome": outcome,
            "page_hint": page_hint,
            "candidates": "",
            "assigned": family,
        })
    return family


def claim_key(
    row: Mapping[str, Any],
    *,
    item: ClaimLike | None = None,
    captions: Mapping[str, Mapping[str, str]] | None = None,
    catalog: Iterable[Mapping[str, str]] = (),
    forest_map: Mapping[str, str] | None = None,
    unmapped: list[dict[str, str]] | None = None,
    unresolved: list[dict[str, str]] | None = None,
) -> tuple[str, str, str, str]:
    src = item if item is not None else row
    study = str(_get(src, "study_id") or "").strip()
    raw_name = str(_get(src, "raw_field_name") or _get(src, "column_header") or "")
    group = group_family(
        str(_get(src, "group") or ""),
        str(_get(src, "cohort_label") or ""),
        str(_get(src, "cohort_status") or ""),
        raw_name,
    )
    field = field_family(
        str(_get(src, "field_type") or ""),
        raw_name,
        unmapped=unmapped,
    )
    disp = resolve_display(
        source_location=str(row.get("source_location") or _get(src, "source_location") or ""),
        table_id=str(_get(src, "table_id") or ""),
        outcome=str(_get(src, "outcome") or row.get("outcome") or ""),
        captions=captions,
        catalog=catalog,
        forest_map=forest_map,
        unresolved=unresolved,
    )
    if not disp:
        disp = outcome_family(
            str(_get(src, "outcome") or row.get("outcome") or ""),
            str(row.get("source_location") or ""),
            capture_family(row, item),
        )
    return (study, group, field, disp)


def expected_displays(gold: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    catalog = gold_display_catalog(gold)
    return [
        {
            "source_location": entry["source_location"],
            "display_family": entry["display_family"],
            "kind": "forest_plot" if entry.get("capture") == "figure_ocr" else "pdf_table",
            "capture": entry.get("capture") or "",
        }
        for entry in catalog
    ]


def hit_matches_location(
    hit: DisplayHit | Mapping[str, Any],
    source_location: str,
    *,
    forest_id: str = "",
) -> bool:
    want = display_family(source_location)
    if want.startswith("forest_") and forest_id:
        return want == forest_id
    have = display_family(str(_get(hit, "caption") or ""), str(_get(hit, "page_hint") or ""))
    if want and have and want == have:
        return True
    if want.startswith("forest_"):
        return False
    blob = f"{_get(hit, 'caption')} {_get(hit, 'page_hint')}".lower()
    loc = (source_location or "").strip().lower()
    return bool(loc) and loc in blob


def values_match(parser_value: object, gold_value: object, field_type: str = "") -> bool:
    left = "" if parser_value is None else str(parser_value).strip()
    right = "" if gold_value is None else str(gold_value).strip()
    if left.lower() == right.lower():
        return True
    if field_type:
        label = compare_values(
            field_type=field_type, review_value=parser_value, source_value=gold_value,
            rel_tolerance=ToleranceTable().rel_tolerance(field_type),
            sd_rel_tolerance=ToleranceTable().sd_rel_tolerance(field_type),
        ).label
        if label == AuditLabel.MATCH:
            return True
    a, b = primary_number(parser_value), primary_number(gold_value)
    return a is not None and b is not None and a == b


def score_localize(
    hits: Iterable[DisplayHit | Mapping[str, Any]],
    gold: Iterable[Mapping[str, str]],
    *,
    captured: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = expected_displays(gold)
    hit_list = list(hits)
    on_hits = [h for h in hit_list if bool(_get(h, "evidence_chain"))]
    forest_map = forest_ordinal_map(on_hits)
    recalled = []
    missed = []
    for want in expected:
        matched = [
            h for h in on_hits
            if hit_matches_location(
                h, want["source_location"],
                forest_id=forest_map.get(str(_get(h, "display_id") or ""), ""),
            )
        ]
        (recalled if matched else missed).append(want)
    pooled_on = [
        {
            "display_id": str(_get(h, "display_id") or ""),
            "caption": str(_get(h, "caption") or ""),
        }
        for h in on_hits
        if _POOLED.search(str(_get(h, "caption") or _get(h, "display_id") or ""))
    ]
    captured_list = list(captured or [])
    captured_pooled = [
        {
            "table_id": str(t.get("table_id") or ""),
            "caption": str(t.get("caption") or ""),
        }
        for t in captured_list
        if _POOLED.search(str(t.get("caption") or t.get("table_id") or ""))
    ]
    n = len(expected)
    return {
        "n_expected": n,
        "n_hits": len(hit_list),
        "n_on": len(on_hits),
        "recall": (len(recalled) / n) if n else 1.0,
        "recalled": recalled,
        "missed": missed,
        "pooled_marked_on": pooled_on,
        "captured_pooled": captured_pooled,
    }


def _abstained_forest_displays(
    captured: Iterable[Mapping[str, Any]],
    forest_map: Mapping[str, str],
) -> set[str]:
    """Displays whose forest tool returned an empty grid and named the gap."""
    families: set[str] = set()
    for table in captured:
        if str(table.get("display_kind") or "") != "forest_plot":
            continue
        n_rows = table.get("n_rows")
        if n_rows is None:
            n_rows = len(table.get("rows") or [])
        try:
            empty = int(n_rows) == 0
        except (TypeError, ValueError):
            empty = not (table.get("rows") or [])
        if not empty:
            continue
        blob = " ".join(str(x) for x in (table.get("difficulties") or [])).lower()
        if (
            "raster pixels" not in blob
            and "no per-study grid" not in blob
            and "not invented" not in blob
            and "429" not in blob
            and "rate limited" not in blob
        ):
            continue
        tid = str(table.get("table_id") or table.get("display_id") or table.get("id") or "")
        family = forest_map.get(tid) or display_family(str(table.get("caption") or ""))
        if family:
            families.add(family)
        if tid:
            families.add(tid)
    return families


def _extraction_kind_row(
    kind: str,
    key: tuple[str, str, str, str],
    body: Mapping[str, Any] | ClaimLike,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "review_data_id": _get(body, "review_data_id") or "",
        "study": key[0],
        "group": key[1],
        "field": key[2],
        "display": key[3],
        "value": _get(body, "value"),
        "raw_field_name": _get(body, "raw_field_name") or _get(body, "column_header"),
        "table_id": _get(body, "table_id") or "",
    }


def _checksum_index(
    captured: Iterable[Mapping[str, Any]],
    forest_map: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for table in captured:
        info = {
            "failures": [str(x) for x in (table.get("checksum_failures") or [])],
            "printed": list(table.get("checksum_printed_values") or []),
            "sums": {str(k): v for k, v in dict(table.get("checksum_column_sums") or {}).items()},
        }
        tid = str(table.get("table_id") or table.get("display_id") or table.get("id") or "")
        family = forest_map.get(tid) or display_family(str(table.get("caption") or ""))
        for key in (tid, family):
            if key:
                out[key] = info
    return out


def _header_covers_gold(header: str, gold: Mapping[str, Any]) -> bool:
    want_field = field_family(
        str(gold.get("field_type") or ""), str(gold.get("raw_field_name") or ""))
    want_group = group_family(
        str(gold.get("group") or ""), str(gold.get("cohort_label") or ""),
        "", str(gold.get("raw_field_name") or ""))
    return (
        field_family("", header) == want_field
        and group_family("", "", "", header) == want_group
    )


def _cell_checksum_audit(
    gold: Mapping[str, Any],
    display: str,
    index: Mapping[str, dict[str, Any]],
) -> dict[str, Any] | None:
    info = index.get(display)
    if not info or not info["failures"]:
        return None
    for header in info["failures"]:
        if _header_covers_gold(header, gold):
            return {
                "column": header,
                "study_sum": info["sums"].get(header),
                "printed_totals": info["printed"],
            }
    return None


def classify_extraction_gaps(
    fn_keys: Iterable[tuple[str, str, str, str]],
    gt: Mapping[tuple[str, str, str, str], Mapping[str, str]],
    parser_unmatched: Iterable[tuple[tuple[str, str, str, str], ClaimLike]],
    abstained: set[str],
    checksum_index: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Split forest misses: checksum, abstained, invented, or produced nothing.

    Priority is checksum_failed > not_extractable > fabricated > missing so a
    cell is counted once.
    """
    checksum_index = checksum_index or {}
    checksum_failed: list[dict[str, Any]] = []
    not_extractable: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    fabricated: list[dict[str, Any]] = []
    seen_parser: set[tuple[str, str, str, str]] = set()
    for key in sorted(fn_keys):
        row = gt[key]
        if capture_family(row) != "figure_ocr":
            continue
        rec = _extraction_kind_row("missing", key, row)
        audit = _cell_checksum_audit(row, key[3], checksum_index)
        if audit:
            rec["kind"] = "checksum_failed"
            rec["study_sum"] = audit["study_sum"]
            rec["printed_totals"] = audit["printed_totals"]
            rec["column"] = audit["column"]
            checksum_failed.append(rec)
        elif key[3] in abstained:
            rec["kind"] = "not_extractable"
            not_extractable.append(rec)
        else:
            missing.append(rec)
    for key, item in parser_unmatched:
        if key in seen_parser:
            continue
        seen_parser.add(key)
        src = _item_as_row(item) if not isinstance(item, Mapping) else item
        if capture_family(src, item) != "figure_ocr":
            continue
        value = _get(item, "value")
        if value is None or str(value).strip() == "":
            continue
        gold = gt.get(key)
        audit = _cell_checksum_audit(gold or src, key[3], checksum_index) if gold else None
        if audit:
            rec = _extraction_kind_row("checksum_failed", key, item)
            rec["study_sum"] = audit["study_sum"]
            rec["printed_totals"] = audit["printed_totals"]
            rec["column"] = audit["column"]
            checksum_failed.append(rec)
            continue
        fabricated.append(_extraction_kind_row("fabricated", key, item))
    return {
        "checksum_failed": checksum_failed,
        "not_extractable": not_extractable,
        "fabricated": fabricated,
        "missing": missing,
        "counts": {
            "checksum_failed": len(checksum_failed),
            "not_extractable": len(not_extractable),
            "fabricated": len(fabricated),
            "missing": len(missing),
        },
    }


def failing_axis(
    parser_key: tuple[str, str, str, str],
    gold_keys: Iterable[tuple[str, str, str, str]],
) -> str:
    gold = list(gold_keys)
    same_display = [k for k in gold if k[3] == parser_key[3]]
    pool = same_display or gold
    if not any(k[0] == parser_key[0] for k in pool):
        return "study"
    if not any(k[0] == parser_key[0] and k[1] == parser_key[1] for k in pool):
        return "group"
    if not any(k[:3] == parser_key[:3] for k in pool):
        return "field"
    if parser_key not in set(gold):
        return "display"
    return ""


def score_claims(
    gold: Iterable[Mapping[str, str]],
    items: Iterable[ClaimLike],
    *,
    slice: str = "all",
    hits: Iterable[DisplayHit | Mapping[str, Any]] = (),
    captured: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    gold_rows = [
        row for row in gold
        if slice == "all" or capture_family(row) == _slice_capture(slice)
    ]
    item_rows = [
        it for it in items
        if slice == "all" or capture_family({
            "capture": _get(it, "capture"),
            "display_kind": _get(it, "display_kind"),
            "source_location": _get(it, "source_location"),
        }, it) == _slice_capture(slice)
    ]
    captions = caption_index(hits)
    catalog = gold_display_catalog(gold_rows)
    forest_map = forest_ordinal_map(hits, item_rows)
    abstained = _abstained_forest_displays(captured, forest_map)
    unmapped: list[dict[str, str]] = []
    unresolved: list[dict[str, str]] = []

    gt: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in gold_rows:
        gt[claim_key(row, captions=captions, catalog=catalog, forest_map=forest_map)] = row

    got: dict[tuple[str, str, str, str], ClaimLike] = {}
    collisions: list[dict[str, Any]] = []
    parser_keys: list[tuple[tuple[str, str, str, str], ClaimLike]] = []
    for item in item_rows:
        key = claim_key(
            _item_as_row(item), item=item, captions=captions, catalog=catalog,
            forest_map=forest_map, unmapped=unmapped, unresolved=unresolved,
        )
        parser_keys.append((key, item))
        if key in got:
            collisions.append({
                "key": list(key),
                "kept_value": _get(got[key], "value"),
                "dropped_value": _get(item, "value"),
                "raw_field_name": _get(item, "raw_field_name"),
            })
        got[key] = item

    tp_keys = set(gt) & set(got)
    fn_keys = set(gt) - set(got)
    fp_keys = set(got) - set(gt)
    matched_values = 0
    mismatches: list[dict[str, Any]] = []
    for key in sorted(tp_keys):
        gold_row, item = gt[key], got[key]
        parser_value = _get(item, "value")
        if values_match(parser_value, gold_row.get("value"), gold_row.get("field_type") or ""):
            matched_values += 1
        else:
            mismatches.append({
                "review_data_id": gold_row.get("review_data_id", ""),
                "study": key[0], "group": key[1], "field": key[2],
                "display": key[3],
                "parser_value": parser_value,
                "gt_value": gold_row.get("value"),
            })

    def _rows(keys, source: Mapping, kind: str) -> list[dict[str, Any]]:
        out = []
        for key in sorted(keys):
            body = source[key]
            out.append({
                "status": kind,
                "review_data_id": _get(body, "review_data_id") if kind != "fp" else "",
                "study": key[0], "group": key[1], "field": key[2],
                "display": key[3],
                "value": _get(body, "value"),
                "raw_field_name": _get(body, "raw_field_name") or _get(body, "column_header"),
            })
        return out

    n_gt, n_got, n_tp = len(gt), len(got), len(tp_keys)
    diagnosis = []
    for key, item in parser_keys:
        if key in tp_keys:
            continue
        axis = failing_axis(key, gt)
        diagnosis.append({
            "review_data_id": _get(item, "review_data_id") or "",
            "failing_axis": axis,
            "parser_key": list(key),
            "value": _get(item, "value"),
            "raw_field_name": _get(item, "raw_field_name"),
            "table_id": _get(item, "table_id"),
            "group": _get(item, "group"),
            "cohort_label": _get(item, "cohort_label"),
            "field_type": _get(item, "field_type"),
        })

    checksum_index = _checksum_index(captured, forest_map)
    gaps = classify_extraction_gaps(fn_keys, gt, [
        (key, item) for key, item in parser_keys if key not in tp_keys
    ], abstained, checksum_index)
    released_wrong: list[dict[str, Any]] = []
    for row in mismatches:
        key = (row["study"], row["group"], row["field"], row["display"])
        gold_row = gt.get(key)
        if gold_row is None or capture_family(gold_row) != "figure_ocr":
            continue
        released_wrong.append(row)
        audit = _cell_checksum_audit(gold_row, key[3], checksum_index)
        if audit:
            rec = _extraction_kind_row("checksum_failed", key, gold_row)
            rec["study_sum"] = audit["study_sum"]
            rec["printed_totals"] = audit["printed_totals"]
            rec["column"] = audit["column"]
            rec["parser_value"] = row.get("parser_value")
            gaps["checksum_failed"].append(rec)
            gaps["counts"]["checksum_failed"] = len(gaps["checksum_failed"])
    n_gt_forest = sum(1 for row in gt.values() if capture_family(row) == "figure_ocr")
    n_matched_forest = sum(
        1 for key in tp_keys
        if capture_family(gt[key]) == "figure_ocr"
        and values_match(_get(got[key], "value"), gt[key].get("value"), gt[key].get("field_type") or "")
    )
    if slice in {"forest", "figure", "figure_ocr"}:
        raw_ok, raw_denom = matched_values, n_gt
    else:
        raw_ok, raw_denom = n_matched_forest, n_gt_forest
    integrity = {
        "raw_accuracy": raw_ok,
        "raw_accuracy_denom": raw_denom,
        "detected_error": gaps["counts"]["checksum_failed"],
        "released_wrong": len(released_wrong),
        "released_wrong_rows": released_wrong,
    }
    return {
        "slice": slice,
        "n_gt": n_gt,
        "n_parser": n_got,
        "n_parser_rows": len(item_rows),
        "n_matched": n_tp,
        "recall": (n_tp / n_gt) if n_gt else 0.0,
        "precision": (n_tp / n_got) if n_got else 0.0,
        "value_match": (matched_values / n_tp) if n_tp else 0.0,
        "value_matched": matched_values,
        "value_accuracy_over_gold": (matched_values / n_gt) if n_gt else 0.0,
        "n_unaligned": len(item_rows) - n_tp,
        "missed": dict(Counter(k[2] for k in fn_keys)),
        "spurious": dict(Counter(k[2] for k in fp_keys)),
        "mismatched_values": mismatches,
        "false_negatives": _rows(fn_keys, gt, "fn"),
        "false_positives": _rows(fp_keys, got, "fp"),
        "unaligned": _rows(fp_keys, got, "fp"),
        "join_diagnosis": {
            "parser_unmatched": diagnosis,
            "failing_axis_counts": dict(Counter(d["failing_axis"] for d in diagnosis)),
            "key_collisions": collisions,
            "unmapped_field": unmapped,
            "display_unresolved": unresolved,
            "extraction_outcome": gaps["counts"],
            "checksum_failed": gaps["checksum_failed"],
            "not_extractable": gaps["not_extractable"],
            "fabricated": gaps["fabricated"],
            "missing": gaps["missing"],
            "integrity": integrity,
        },
        "integrity": integrity,
    }


def score_extraction(
    gold: Iterable[Mapping[str, str]],
    items: Iterable[ClaimLike],
    *,
    hits: Iterable[DisplayHit | Mapping[str, Any]] = (),
    captured: Iterable[Mapping[str, Any]] = (),
    dropped_non_source: Iterable[str] = (),
    lens: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gold_rows = list(gold)
    item_rows = list(items)
    hit_list = list(hits)
    overall = score_claims(
        gold_rows, item_rows, slice="all", hits=hit_list, captured=captured)
    table = score_claims(
        gold_rows, item_rows, slice="table", hits=hit_list, captured=captured)
    forest = score_claims(
        gold_rows, item_rows, slice="forest", hits=hit_list, captured=captured)
    localize = score_localize(hit_list, gold_rows, captured=captured)
    return {
        "n_gold": len(gold_rows),
        "n_parser": len(item_rows),
        "localize": localize,
        "claims": overall,
        "table_text": table,
        "figure_ocr": forest,
        "join_diagnosis": overall.get("join_diagnosis") or {},
        "integrity": forest.get("integrity") or overall.get("integrity") or {},
        "dropped_non_source": list(dropped_non_source),
        "lens": dict(lens or {}),
        "parser_rows": [_item_public(it) for it in item_rows],
    }


def render_html(stats: Mapping[str, Any]) -> str:
    claims = stats.get("claims") or {}
    table = stats.get("table_text") or {}
    forest = stats.get("figure_ocr") or {}
    loc = stats.get("localize") or {}
    diag = stats.get("join_diagnosis") or claims.get("join_diagnosis") or {}

    def pct(value: object) -> str:
        try:
            return f"{float(value) * 100:.1f}%"
        except (TypeError, ValueError):
            return "—"

    def aligned_label(body: Mapping[str, Any]) -> str:
        n_tp = int(body.get("n_matched") or 0)
        return f"{pct(body.get('value_match'))} of {n_tp} aligned"

    integrity = stats.get("integrity") or forest.get("integrity") or {}
    tiles = [
        ("Table 1 recall", pct(table.get("recall")), "accent"),
        ("Forest recall", pct(forest.get("recall")), "muted"),
        ("Raw accuracy",
         f"{integrity.get('raw_accuracy', 0)}/{integrity.get('raw_accuracy_denom', 0)}",
         "accent"),
        ("Detected error", str(integrity.get("detected_error", 0)), "muted"),
        ("Released wrong", str(integrity.get("released_wrong", 0)), "good"),
    ]
    tiles_html = "".join(
        f'<div class="tile {cls}"><div class="tv">{escape(val)}</div>'
        f'<div class="tk">{escape(title)}</div></div>'
        for title, val, cls in tiles
    )
    note = (
        f'<p class="sub">Gold <b>{stats.get("n_gold", 0)}</b> cells; parser '
        f'<b>{stats.get("n_parser", 0)}</b>. '
        f'Value accuracy / gold is {claims.get("value_matched", 0)}/'
        f'{claims.get("n_gt", 0)} — same denominator as recall. '
        f'Aligned value match hides unjoined cells; unaligned parser rows: '
        f'<b>{claims.get("n_unaligned", 0)}</b>.</p>'
    )

    def slice_block(title: str, body: Mapping[str, Any]) -> str:
        return (
            f"<h2>{escape(title)}</h2>"
            f'<p>recall <b>{pct(body.get("recall"))}</b> '
            f'({body.get("n_matched", 0)}/{body.get("n_gt", 0)}) · '
            f'precision <b>{pct(body.get("precision"))}</b> · '
            f'value match {escape(aligned_label(body))} · '
            f'value accuracy / gold <b>{pct(body.get("value_accuracy_over_gold"))}</b></p>'
        )

    missed = loc.get("missed") or []
    missed_html = "".join(
        f"<li><code>{escape(m.get('source_location', ''))}</code></li>" for m in missed
    ) or "<li>(none)</li>"
    pooled = loc.get("pooled_marked_on") or []
    pooled_html = "".join(
        f"<li><code>{escape(p.get('caption') or p.get('display_id') or '')}</code></li>"
        for p in pooled
    ) or "<li>(none)</li>"

    fn_rows = (claims.get("false_negatives") or [])[:40]
    fn_html = "".join(
        f"<tr><td>{escape(r['review_data_id'])}</td>"
        f"<td>{escape(r['study'])}</td><td>{escape(r['group'])}</td>"
        f"<td>{escape(r['field'])}</td><td>{escape(r['display'])}</td>"
        f"<td>{escape(str(r.get('value') or ''))}</td></tr>"
        for r in fn_rows
    )
    mm = claims.get("mismatched_values") or []
    mm_html = "".join(
        f"<tr><td>{escape(m.get('review_data_id', ''))}</td>"
        f"<td>{escape(str(m.get('parser_value')))}</td>"
        f"<td>{escape(str(m.get('gt_value')))}</td></tr>"
        for m in mm[:30]
    )
    unaligned = claims.get("unaligned") or []
    ua_html = "".join(
        f"<tr><td>{escape(r['study'])}</td><td>{escape(r['group'])}</td>"
        f"<td>{escape(r['field'])}</td><td>{escape(r['display'])}</td>"
        f"<td>{escape(str(r.get('value') or ''))}</td></tr>"
        for r in unaligned[:40]
    )
    axis_counts = diag.get("failing_axis_counts") or {}
    axis_html = "".join(
        f"<li><code>{escape(k)}</code> ×{v}</li>" for k, v in sorted(axis_counts.items())
    ) or "<li>(none)</li>"
    outcome = diag.get("extraction_outcome") or {}
    integrity = stats.get("integrity") or forest.get("integrity") or {}
    outcome_html = (
        f'<p>Forest gaps: <code>checksum_failed</code> {outcome.get("checksum_failed", 0)} · '
        f'<code>not_extractable</code> {outcome.get("not_extractable", 0)} · '
        f'<code>fabricated</code> {outcome.get("fabricated", 0)} · '
        f'<code>missing</code> {outcome.get("missing", 0)}</p>'
        f'<p>Integrity: raw_accuracy <b>{integrity.get("raw_accuracy", 0)}/'
        f'{integrity.get("raw_accuracy_denom", 0)}</b> · '
        f'detected_error <b>{integrity.get("detected_error", 0)}</b> · '
        f'released_wrong <b>{integrity.get("released_wrong", 0)}</b> '
        f'(ship gate is released_wrong = 0)</p>'
    )
    lens = stats.get("lens") or {}
    lens_line = escape(str(lens.get("lens_one_line") or lens.get("domain") or "(no lens recorded)"))
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Review Extraction</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;color:#1a2026;background:#f4f6f8}}
.wrap{{max-width:960px;margin:0 auto}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}}
.tile{{background:#fff;border:1px solid #e3e7eb;border-radius:10px;padding:14px;text-align:center}}
.tv{{font-size:1.4rem;font-weight:700}} .tk{{font-size:12px;color:#5e6b77;margin-top:4px}}
table{{border-collapse:collapse;width:100%;background:#fff}} th,td{{border:1px solid #e3e7eb;padding:6px 8px;font-size:13px}}
.sub{{color:#5e6b77}} code{{font-size:12px}}
</style></head><body><div class="wrap">
<div class="sub">ReAct-Review · Review Extraction</div>
<h1>Review Extraction report</h1>
<div class="tiles">{tiles_html}</div>
{note}
<p>Lens: {lens_line}</p>
{slice_block("Table 1 (table_text)", table)}
{slice_block("Forest Events/Total (figure_ocr)", forest)}
{slice_block("All gold cells", claims)}
<h2>Join diagnosis (unmatched parser rows)</h2>
{outcome_html}
<ul>{axis_html}</ul>
<h2>Localize</h2>
<p>expected displays {loc.get("n_expected", 0)} · on-chain hits {loc.get("n_on", 0)} ·
recall <b>{pct(loc.get("recall"))}</b></p>
<p>Missed displays</p><ul>{missed_html}</ul>
<p>Pooled / GRADE marked evidence_chain=true (should be off)</p><ul>{pooled_html}</ul>
<h2>Missed gold cells (first 40)</h2>
<table><thead><tr><th>id</th><th>study</th><th>group</th><th>field</th><th>display</th><th>gold</th></tr></thead>
<tbody>{fn_html or '<tr><td colspan="6">(none)</td></tr>'}</tbody></table>
<h2>Unaligned parser rows (not in gold)</h2>
<table><thead><tr><th>study</th><th>group</th><th>field</th><th>display</th><th>parser</th></tr></thead>
<tbody>{ua_html or '<tr><td colspan="5">(none)</td></tr>'}</tbody></table>
<h2>Aligned value mismatches</h2>
<table><thead><tr><th>id</th><th>parser</th><th>gold</th></tr></thead>
<tbody>{mm_html or '<tr><td colspan="3">(none)</td></tr>'}</tbody></table>
</div></body></html>
"""


def _slice_capture(slice_name: str) -> str:
    if slice_name in {"table", "table_text"}:
        return "table_text"
    if slice_name in {"forest", "figure", "figure_ocr"}:
        return "figure_ocr"
    return "all"


def _get(obj: ClaimLike | DisplayHit, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _item_as_row(item: ClaimLike) -> dict[str, Any]:
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    return {
        "study_id": getattr(item, "study_id", ""),
        "group": getattr(item, "group", ""),
        "field_type": getattr(item, "field_type", ""),
        "raw_field_name": getattr(item, "raw_field_name", ""),
        "value": getattr(item, "value", None),
        "outcome": getattr(item, "outcome", ""),
        "display_kind": getattr(item, "display_kind", ""),
        "table_id": getattr(item, "table_id", ""),
        "source_location": getattr(item, "source_location", ""),
        "cohort_label": getattr(item, "cohort_label", ""),
        "cohort_status": getattr(item, "cohort_status", ""),
    }


def _item_public(item: ClaimLike) -> dict[str, Any]:
    row = _item_as_row(item)
    keep = (
        "review_data_id", "study_id", "group", "field_type", "raw_field_name",
        "value", "unit", "outcome", "display_kind", "value_source", "table_id",
        "cohort_label", "cohort_status", "source_location",
    )
    return {k: row.get(k) for k in keep if k in row}
