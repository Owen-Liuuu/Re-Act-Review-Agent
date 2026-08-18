"""One-shot builder for benchmark_3 gold files. Not a runtime dependency."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Previous extractor reads kept for human verification. Unmatched N is NOT
# copied into source_value: the review cell is the matched N.
_HINTS: dict[str, dict[str, str]] = {
    "R001": dict(
        seed_note="previous run: abstract / affiliation China",
        expected_match_mode="skip",
        notes="bibliographic / affiliation; confirm whether the paper states country as a results field",
    ),
    "R002": dict(
        expected_match_mode="skip",
        notes="publication year is bibliographic",
    ),
    "R003": dict(
        seed_note="abstract said retrospective pair-matched",
        expected_match_mode="semantic",
    ),
    "R004": dict(
        expected_match_mode="skip",
        notes="review Table 1 is inclusion age (>70), not mean age; confirm the quantity in the paper",
    ),
    "R005": dict(
        seed_source_value="89",
        seed_note="previous extractor compared matched N to unmatched 89",
        expected_match_mode="numeric",
        notes="fill matched MIE N from the paper; unmatched 89 is the wrong comparator for this cell",
    ),
    "R006": dict(
        seed_source_value="318",
        seed_note="previous extractor compared matched N to unmatched 318",
        expected_match_mode="numeric",
        notes="fill matched OE N from the paper; unmatched 318 is the wrong comparator for this cell",
    ),
    "R007": dict(
        seed_note="previous run had no Capovilla full text",
        expected_match_mode="skip",
        notes="Italy + Germany is printed in the review; confirm in the source paper",
    ),
    "R008": dict(expected_match_mode="skip", notes="publication year is bibliographic"),
    "R009": dict(expected_match_mode="semantic"),
    "R010": dict(
        expected_match_mode="skip",
        notes="review Table 1 is inclusion age (≥75), not mean age",
    ),
    "R011": dict(
        expected_match_mode="numeric",
        notes="footnote in the review: 58 vs 58 matched; confirm in the paper",
    ),
    "R012": dict(
        expected_match_mode="numeric",
        notes="review prints unmatched/matched 102 / 58*; Fig 3.3.1–3.3.2 use OE Total 102, Fig 3.3.4 uses 58 (IC01). Fill the quantity the paper reports for this cell.",
    ),
    "R013": dict(
        source_value="China",
        seed_source_value="China",
        seed_note="PMC full text",
        expected_match_mode="skip",
        notes="verify in the paper; country may still be bibliographic",
    ),
    "R014": dict(expected_match_mode="skip", notes="publication year is bibliographic"),
    "R015": dict(
        source_value="retrospective cohort study",
        seed_source_value="retrospective cohort study",
        seed_note="source may not say PSM in that sentence",
        expected_match_mode="semantic",
        notes="verify wording in the paper",
    ),
    "R016": dict(
        seed_note="previous note: source Table 1 has median 73 (70-88)",
        expected_match_mode="skip",
        notes="inclusion age vs reported median — confirm they are the same quantity before filling",
    ),
    "R017": dict(
        seed_source_value="358",
        seed_note="previous extractor compared matched N to unmatched 358",
        expected_match_mode="numeric",
        notes="fill matched MIE N from the paper; unmatched 358 is the wrong comparator for this cell",
    ),
    "R018": dict(
        seed_source_value="111",
        seed_note="previous extractor compared matched N to unmatched 111",
        expected_match_mode="numeric",
        notes="fill matched OE N from the paper; unmatched 111 is the wrong comparator for this cell",
    ),
    "R022": dict(
        notes="Capovilla OE Total 102 here vs 58 on Fig 3.3.4 (IC01); confirm which N the paper reports for overall complications",
    ),
    "R054": dict(
        notes="Capovilla OE Total 58 here vs 102 on Fig 3.3.1–3.3.2 (IC01); confirm which N the paper reports for leak",
    ),
}


def _w(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_gold_claims() -> None:
    """One source-side row per review_ground_truth cell. No per-study OR."""
    fields = [
        "review_data_id", "study_id", "group", "timepoint", "field_type",
        "raw_field_name", "review_value", "unit", "review_location", "outcome",
        "capture", "source_value", "source_quote", "source_location_in_paper",
        "source_unit", "expected_label", "expected_match_mode",
        "seed_source_value", "seed_note", "notes",
    ]
    gold = ROOT / "review_ground_truth.csv"
    with gold.open(encoding="utf-8-sig", newline="") as handle:
        review_rows = [row for row in csv.DictReader(handle) if row.get("review_data_id")]

    rows: list[dict[str, str]] = []
    for item in review_rows:
        rid = item["review_data_id"].strip()
        hint = _HINTS.get(rid, {})
        field_type = item.get("field_type") or ""
        capture = item.get("capture") or ""
        mode = hint.get("expected_match_mode") or (
            "numeric" if field_type in {"events", "subgroup_n"} else "")
        notes = hint.get("notes", "")
        if capture == "figure_ocr" and not notes:
            notes = ("fill Events / Total from the source paper; do not copy the "
                     "review forest. Per-study OR is out of scope.")
        rows.append({
            "review_data_id": rid,
            "study_id": item.get("study_id") or "",
            "group": item.get("group") or "",
            "timepoint": item.get("timepoint") or "",
            "field_type": field_type,
            "raw_field_name": item.get("raw_field_name") or "",
            "review_value": item.get("value") or "",
            "unit": item.get("unit") or "",
            "review_location": item.get("source_location") or "",
            "outcome": item.get("outcome") or "",
            "capture": capture,
            "source_value": hint.get("source_value", ""),
            "source_quote": "",
            "source_location_in_paper": "",
            "source_unit": "",
            "expected_label": "",
            "expected_match_mode": mode,
            "seed_source_value": hint.get("seed_source_value", ""),
            "seed_note": hint.get("seed_note", ""),
            "notes": notes,
        })

    target = ROOT / "gold_claims.csv"
    _w(target, fields, rows)
    print(f"wrote {target.name} {len(rows)}")


def build_internal() -> None:
    fields = [
        "ic_id", "quantity", "location_a", "value_a", "location_b", "value_b",
        "discrepancy_type", "likely_cause", "error_owner", "severity",
        "needs_human_review", "notes",
    ]
    rows = [
        dict(ic_id="IC01", quantity="Capovilla OE N",
             location_a="Table 1 N OE", value_a="102 / 58*",
             location_b="Figure 2 and Figure 3 OE Total", value_b="102",
             discrepancy_type="matched_vs_unmatched",
             likely_cause="Methods say only PSM-matched data were pooled; Fig 2/3 use unmatched OE 102",
             error_owner="review_author", severity="high", needs_human_review="yes",
             notes="Fig 5 leak uses OE Total 58 (matched). Same study, two denominators."),
        dict(ic_id="IC02", quantity="pooled OE N",
             location_a="Figure 2 / Figure 3 total OE", value_a="215",
             location_b="Figure 5 total OE", value_b="171",
             discrepancy_type="inconsistent_denominator",
             likely_cause="215 = 102+58+55 (Fig 2/3 Capovilla unmatched); 171 = 58+58+55 (Fig 5 matched)",
             error_owner="review_author", severity="high", needs_human_review="yes",
             notes=""),
        dict(ic_id="IC03", quantity="overall complications pooled OR",
             location_a="Table 2 / §3.3.1 prose", value_a="0.40 (0.27-0.60)",
             location_b="Figure 2 diamond", value_b="0.40 [0.27, 0.60]",
             discrepancy_type="none", likely_cause="same estimate, different bracket style",
             error_owner="none", severity="low", needs_human_review="no",
             notes="negative control: Table 2, prose, and figure diamond agree"),
        dict(ic_id="IC04", quantity="30-day mortality studies",
             location_a="Table 2 Studies (n)", value_a="2",
             location_b="Figure 4", value_b="Li J 2015 + Li K 2025 (Capovilla absent)",
             discrepancy_type="none", likely_cause="Capovilla did not report this outcome in the matched forest",
             error_owner="none", severity="low", needs_human_review="no",
             notes="Li K 0/92 vs 0/55 is in both Figure 4 and §3.3.3 prose"),
        dict(ic_id="IC05", quantity="methods vs forest denominators",
             location_a="§2.7", value_a="only data from propensity score-matched populations were used",
             location_b="Figure 2 Capovilla OE Total", value_b="102 unmatched",
             discrepancy_type="protocol_vs_analysis",
             likely_cause="analysis did not follow the matched-only rule for every outcome",
             error_owner="review_author", severity="high", needs_human_review="yes",
             notes=""),
    ]
    _w(ROOT / "internal_consistency.csv", fields, rows)
    print(f"internal_consistency {len(rows)}")


if __name__ == "__main__":
    build_gold_claims()
