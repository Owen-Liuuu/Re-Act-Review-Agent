"""Review Extraction gold join keeps forest plots apart and is domain-neutral."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.review_extraction_score import (
    claim_key,
    expected_displays,
    group_family,
    load_gold_csv,
    score_claims,
    score_extraction,
    score_localize,
)
from react_review.parser.review_extraction.schemas import DisplayHit, ReviewClaim

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "eval" / "benchmark_3" / "review_ground_truth.csv"


def test_gold_display_catalog_uses_forest_ordinals_not_section_numbers():
    from eval.review_extraction_score import gold_display_catalog
    families = {e["display_family"] for e in gold_display_catalog(load_gold_csv(GOLD))}
    assert families == {"table 1", "forest_1", "forest_2", "forest_3", "forest_4"}


def test_gold_has_62_unique_extraction_keys():
    rows = load_gold_csv(GOLD)
    keys = [claim_key(row) for row in rows]
    assert len(rows) == 62
    assert len(set(keys)) == 62


def test_forest_events_are_not_collapsed_across_figures():
    rows = [r for r in load_gold_csv(GOLD) if r["study_id"] == "li j_2015"
            and r["group"] == "mie" and r["field_type"] == "events"]
    assert len(rows) == 4
    assert len({claim_key(r) for r in rows}) == 4
    displays = {claim_key(r)[3] for r in rows}
    assert displays == {"forest_1", "forest_2", "forest_3", "forest_4"}


def test_year_and_event_count_aliases_go_through_the_same_table():
    from eval.review_extraction_score import field_family
    assert field_family("year", "Year") == field_family("publication_year", "Year")
    assert field_family("event_count", "Events") == field_family("events", "Events")
    assert field_family("total", "Total") == "subgroup_n"


def test_n_mie_header_sets_group_before_all_fold():
    assert group_family("all", "", "combined", "N MIE") == "mie"
    assert group_family("all", "", "combined", "N OE") == "oe"
    assert group_family("all", "", "combined", "Events") == "all"


def test_parser_item_joins_gold_via_ordinal_not_sequential_id():
    from eval.review_extraction_score import (
        caption_index, claim_key as ckey, forest_ordinal_map, gold_display_catalog,
        resolve_display,
    )
    gold = load_gold_csv(GOLD)
    catalog = gold_display_catalog(gold)
    hits = [DisplayHit(
        display_id="figure_2", kind="forest_plot",
        caption="Forest plot of overall postoperative complications (MIE vs. OE).",
        evidence_chain=True,
    )]
    captions = caption_index(hits)
    forest_map = forest_ordinal_map(hits)
    gold_row = next(r for r in gold if r["review_data_id"] == "R023")
    item = ReviewClaim(
        study_id="li j_2015",
        group="minimally_invasive_esophagectomy",
        field_type="event_count",
        raw_field_name="Minimally Invasive Esophagectomy (MIE) Events",
        value="22",
        cohort_label="Minimally Invasive Esophagectomy (MIE)",
        table_id="figure_2",
        outcome="overall complications",
        display_kind="forest_plot",
    )
    assert group_family(item.group, item.cohort_label, "", item.raw_field_name) == "mie"
    assert resolve_display(
        table_id="figure_2", outcome="overall complications",
        captions=captions, catalog=catalog, forest_map=forest_map,
    ) == "forest_1"
    assert resolve_display(table_id="figure_2", captions={}, catalog=[]) != "figure 2"
    assert ckey(gold_row, captions=captions, catalog=catalog, forest_map=forest_map) == ckey(
        {"study_id": item.study_id, "group": item.group,
         "field_type": item.field_type, "raw_field_name": item.raw_field_name,
         "table_id": item.table_id, "outcome": item.outcome,
         "display_kind": item.display_kind, "cohort_label": item.cohort_label},
        item=item, captions=captions, catalog=catalog, forest_map=forest_map,
    )


def test_printed_figure_number_beats_forest_ordinal():
    from eval.review_extraction_score import caption_index, forest_ordinal_map, resolve_display
    hits = [DisplayHit(
        display_id="figure_2", kind="forest_plot",
        caption="Figure 2 Forest plot of overall complications",
        evidence_chain=True,
    )]
    assert resolve_display(
        table_id="figure_2",
        captions=caption_index(hits),
        forest_map=forest_ordinal_map(hits),
    ) == "figure 2"


def test_table_slice_can_be_perfect_while_forests_are_empty():
    gold = load_gold_csv(GOLD)
    table_items = [
        ReviewClaim(
            study_id=row["study_id"],
            group=row["group"],
            field_type=row["field_type"],
            raw_field_name=row["raw_field_name"],
            value=row["value"],
            table_id="table_1",
            outcome="Table 1. Characteristics of included studies.",
            display_kind="pdf_table",
            cohort_label="MIE" if row["group"] == "mie"
            else ("OE" if row["group"] == "oe" else ""),
            cohort_status="not_applicable" if row["group"] in {"-", "all"} else "resolved",
        )
        for row in gold if row["capture"] == "table_text"
    ]
    stats = score_extraction(gold, table_items)
    assert stats["table_text"]["n_gt"] == 18
    assert stats["table_text"]["recall"] == 1.0
    assert stats["table_text"]["value_match"] == 1.0
    assert stats["figure_ocr"]["n_gt"] == 44
    assert stats["figure_ocr"]["recall"] == 0.0
    assert stats["claims"]["n_matched"] == 18
    outcome = stats["join_diagnosis"]["extraction_outcome"]
    assert outcome["not_extractable"] == 0
    assert outcome["fabricated"] == 0
    assert outcome["missing"] == 44


def test_localize_recalls_table_1_and_flags_grade():
    gold = load_gold_csv(GOLD)
    hits = [
        DisplayHit(display_id="table_1", kind="pdf_table",
                   caption="Table 1 Characteristics of included studies",
                   evidence_chain=True),
        DisplayHit(display_id="figure_2", kind="forest_plot",
                   caption="Forest plot of overall postoperative complications (MIE vs. OE).",
                   evidence_chain=True),
        DisplayHit(display_id="table_2", kind="pdf_table",
                   caption="Table 2 GRADE summary of findings",
                   evidence_chain=True),
    ]
    loc = score_localize(hits, gold)
    expected = {d["display_family"] for d in expected_displays(gold)}
    assert "table 1" in expected
    assert "forest_1" in expected
    assert loc["missed"]  # other three forests not listed
    recalled = {d["display_family"] for d in loc["recalled"]}
    assert "table 1" in recalled
    assert "forest_1" in recalled
    assert loc["pooled_marked_on"]
    assert "GRADE" in loc["pooled_marked_on"][0]["caption"]


def test_score_claims_reports_a_value_mismatch_on_aligned_key():
    gold = [{
        "review_data_id": "R005",
        "study_id": "li j_2015",
        "group": "mie",
        "field_type": "subgroup_n",
        "raw_field_name": "N MIE",
        "value": "58 (matched)",
        "source_location": "Table 1",
        "outcome": "characteristics",
        "capture": "table_text",
    }]
    item = ReviewClaim(
        study_id="li j_2015", group="mie", field_type="subgroup_n",
        raw_field_name="N MIE", value="99", table_id="table_1",
        display_kind="pdf_table", cohort_label="MIE",
        source_location="Table 1",
    )
    scored = score_claims(gold, [item])
    assert scored["n_matched"] == 1
    assert scored["value_match"] == 0.0
    assert scored["value_accuracy_over_gold"] == 0.0
    assert scored["mismatched_values"]


def test_join_diagnosis_names_group_when_forest_arm_is_all():
    gold = load_gold_csv(GOLD)
    item = ReviewClaim(
        study_id="li j_2015", group="all", field_type="event_count",
        raw_field_name="Events", value="23", table_id="figure_2",
        outcome="overall complications", display_kind="forest_plot",
        cohort_status="combined",
    )
    hits = [DisplayHit(
        display_id="figure_2", kind="forest_plot",
        caption="Forest plot of overall postoperative complications (MIE vs. OE).",
        evidence_chain=True,
    )]
    scored = score_claims(gold, [item], hits=hits)
    unmatched = scored["join_diagnosis"]["parser_unmatched"]
    assert unmatched
    assert unmatched[0]["failing_axis"] == "group"
    assert unmatched[0]["parser_key"][3] == "forest_1"
    outcome = scored["join_diagnosis"]["extraction_outcome"]
    assert outcome["fabricated"] == 1
    assert outcome["not_extractable"] == 0
    assert outcome["missing"] == 44


def test_table1_n_mie_joins_when_parser_group_is_all():
    gold = [r for r in load_gold_csv(GOLD) if r["review_data_id"] == "R005"]
    item = ReviewClaim(
        study_id="li j_2015", group="all", field_type="subgroup_n",
        raw_field_name="N MIE", value="58 (matched)", table_id="table_1",
        outcome="Table 1. Characteristics of included studies.",
        display_kind="pdf_table", cohort_status="combined",
    )
    scored = score_claims(gold, [item])
    assert scored["n_matched"] == 1
    assert scored["value_matched"] == 1
    assert scored["value_accuracy_over_gold"] == 1.0


def _forest_hits() -> list[DisplayHit]:
    captions = (
        "Forest plot of overall postoperative complications (MIE vs. OE).",
        "Forest plot of pulmonary complications (MIE vs. OE).",
        "Forest plot of 30-day mortality (MIE vs. OE).",
        "Forest plot of anastomotic leak (MIE vs. OE).",
    )
    return [
        DisplayHit(display_id=f"figure_{i + 2}", kind="forest_plot",
                   caption=cap, evidence_chain=True)
        for i, cap in enumerate(captions)
    ]


@pytest.mark.parametrize("difficulty", [
    "the figure has no per-study grid in the PDF text layer; "
    "its numbers exist only as raster pixels",
    "no raster OCR backend; forest-plot cells were not invented "
    "from the text layer",
])
def test_forest_abstention_is_not_extractable_not_fabricated(difficulty):
    gold = load_gold_csv(GOLD)
    hits = [
        DisplayHit(display_id="table_1", kind="pdf_table",
                   caption="Table 1 Characteristics of included studies",
                   evidence_chain=True),
        *_forest_hits(),
    ]
    captured = [
        {
            "table_id": f"figure_{i}",
            "display_kind": "forest_plot",
            "n_rows": 0,
            "difficulties": [difficulty],
        }
        for i in (2, 3, 4, 5)
    ]
    table_items = [
        ReviewClaim(
            study_id=row["study_id"],
            group=row["group"],
            field_type=row["field_type"],
            raw_field_name=row["raw_field_name"],
            value=row["value"],
            table_id="table_1",
            outcome="Table 1. Characteristics of included studies.",
            display_kind="pdf_table",
            cohort_label="MIE" if row["group"] == "mie"
            else ("OE" if row["group"] == "oe" else ""),
            cohort_status="not_applicable" if row["group"] in {"-", "all"} else "resolved",
        )
        for row in gold if row["capture"] == "table_text"
    ]
    stats = score_extraction(gold, table_items, hits=hits, captured=captured)
    assert stats["table_text"]["recall"] == 1.0
    assert stats["localize"]["recall"] == 1.0
    assert stats["figure_ocr"]["recall"] == 0.0
    assert stats["figure_ocr"]["n_gt"] == 44
    outcome = stats["join_diagnosis"]["extraction_outcome"]
    assert outcome["not_extractable"] == 44
    assert outcome["fabricated"] == 0
    assert outcome["missing"] == 0
    assert outcome.get("checksum_failed", 0) == 0


def test_checksum_failed_outranks_missing_and_released_wrong_is_zero():
    gold = [r for r in load_gold_csv(GOLD) if r["source_location"] == "forest_1"]
    hits = _forest_hits()[:1]
    captured = [{
        "table_id": "figure_2",
        "caption": hits[0].caption,
        "display_kind": "forest_plot",
        "n_rows": 5,
        "difficulties": [
            "column sums disagree with the figure's own total row: "
            "Open Esophagectomy (OE) Total; these cells were not released"],
        "checksum_failures": ["Open Esophagectomy (OE) Total"],
        "checksum_printed_values": [87, 136, 208, 215],
        "checksum_column_sums": {"Open Esophagectomy (OE) Total": 130},
    }]
    items = [
        ReviewClaim(
            study_id=row["study_id"], group=row["group"],
            field_type=row["field_type"], raw_field_name=row["raw_field_name"],
            value=row["value"], table_id="figure_2",
            outcome=row["outcome"], display_kind="forest_plot",
            cohort_label="Minimally Invasive Esophagectomy (MIE)"
            if row["group"] == "mie" else "Open Esophagectomy (OE)",
        )
        for row in gold
        if not (row["group"] == "oe" and row["field_type"] == "subgroup_n")
    ]
    stats = score_extraction(gold, items, hits=hits, captured=captured)
    outcome = stats["join_diagnosis"]["extraction_outcome"]
    integrity = stats["integrity"]
    assert outcome["checksum_failed"] == 3
    assert outcome["missing"] == 0
    assert outcome["fabricated"] == 0
    assert integrity["raw_accuracy"] == 9
    assert integrity["raw_accuracy_denom"] == 12
    assert integrity["detected_error"] == 3
    assert integrity["released_wrong"] == 0
    rec = stats["join_diagnosis"]["checksum_failed"][0]
    assert rec["study_sum"] == 130
    assert 208 in rec["printed_totals"]
