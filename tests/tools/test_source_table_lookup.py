"""Deterministic source-table locate: unique hit or refuse, never nearest."""
from __future__ import annotations

import pytest

from react_review.agents.collector import Collector
from react_review.core.enums import CollectionOutcome
from react_review.normalize.cohorts import CohortLabel, CohortRegistry
from react_review.schemas.table import CapturedTable
from react_review.steps.data_extraction.schemas import DocumentScope, PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.tools.base import Tool, ToolStage
from react_review.tools.extract import FetchFullTextTool
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
    SourceValueResult,
)
from react_review.tools.registry import ToolRegistry
from react_review.tools.source_table_lookup import (
    FALLBACK_TEXT,
    LOCATED_NOT_READ,
    NO_TABLES,
    NOT_IN_TABLE,
    TABLE_LOOKUP_CLASSES,
    classify_missing_source,
    interpret_row_map_response,
    locate_source_cell,
    locate_source_table,
    render_source_row_map_prompt,
    render_table_prompt,
    _field_needle_in,
    _needle_in,
)
from react_review.schemas.evidence import ReviewDataItem


def _table(
    table_id: str,
    caption: str,
    headers: list[list[str]],
    rows: list[list[str]],
) -> CapturedTable:
    return CapturedTable(
        table_id=table_id, caption=caption,
        header_rows=headers, rows=rows,
    )


AGE_TABLE = _table(
    "Tab1", "Table 1 Patient characteristics",
    [["", "MIE (n=58)", "OE (n=102)"]],
    [
        ["Age (years)", "77.2 ± 4.1", "77.8 ± 4.8"],
        ["Male sex", "41 (71)", "70 (69)"],
    ],
)

COMPLICATIONS = _table(
    "Tab2", "Table 2 Postoperative complications",
    [["", "MIE", "OE"]],
    [
        ["Overall complications", "23 (40)", "64 (63)"],
        ["Pulmonary complications", "19 (33)", "58 (57)"],
        ["Anastomotic leak", "7 (12)", "5 (9)"],
    ],
)

TREATMENT_ONLY = _table(
    "near", "Table 1 Baseline",
    [["Characteristic", "Treatment (n=20)"]],
    [["Age (years)", "55 ± 8"]],
)


def test_no_tables_is_its_own_class():
    result = classify_missing_source(
        [], group="mie", cohort_display="MIE",
        raw_field_name="Age", field_type="age",
    )
    assert result.klass == NO_TABLES
    assert result.klass in TABLE_LOOKUP_CLASSES


def test_unique_hit_requires_cohort_in_headers_and_field_in_rows():
    locate = locate_source_table(
        [AGE_TABLE, COMPLICATIONS],
        group="mie", cohort_display="MIE",
        cohorts={"mie": ["MIE", "Minimally Invasive Esophagectomy"]},
        raw_field_name="Age", field_type="age",
    )
    assert locate.unique
    assert locate.table is AGE_TABLE
    prompt = render_table_prompt(locate.table)
    assert "Age (years)" in prompt
    assert "Overall complications" not in prompt


def test_near_wrong_treatment_table_does_not_match_placebo():
    """The Treatment/Placebo accident: a close header is still a miss."""
    locate = locate_source_table(
        [TREATMENT_ONLY],
        group="placebo", cohort_display="Placebo",
        cohorts={"placebo": ["Placebo"]},
        raw_field_name="Age", field_type="age",
    )
    assert not locate.unique
    assert locate.status == "none"
    classified = classify_missing_source(
        [TREATMENT_ONLY],
        group="placebo", cohort_display="Placebo",
        cohorts={"placebo": ["Placebo"]},
        raw_field_name="Age", field_type="age",
    )
    assert classified.klass == NOT_IN_TABLE


def test_treatment_query_does_hit_the_treatment_table():
    locate = locate_source_table(
        [TREATMENT_ONLY],
        group="treatment", cohort_display="Treatment",
        cohorts={"treatment": ["Treatment"]},
        raw_field_name="Age", field_type="age",
    )
    assert locate.unique
    assert locate.table is TREATMENT_ONLY


def test_ambiguous_match_falls_back_instead_of_guessing():
    twin = _table(
        "Tab1b", "Table 1b more characteristics",
        [["", "MIE", "OE"]],
        [["Age (years)", "76", "78"]],
    )
    classified = classify_missing_source(
        [AGE_TABLE, twin],
        group="mie", cohort_display="MIE",
        cohorts={"mie": ["MIE"]},
        raw_field_name="Age", field_type="age",
    )
    assert classified.klass == FALLBACK_TEXT
    assert classified.locate.status == "ambiguous"


def test_combined_group_unique_field_match_is_located():
    classified = classify_missing_source(
        [AGE_TABLE, COMPLICATIONS],
        group="all", cohort_display="",
        raw_field_name="Age", field_type="age",
    )
    assert classified.klass == LOCATED_NOT_READ
    assert classified.locate.table is AGE_TABLE


def test_field_absent_from_every_table_is_not_in_table():
    classified = classify_missing_source(
        [AGE_TABLE, COMPLICATIONS],
        group="mie", cohort_display="MIE",
        cohorts={"mie": ["MIE"]},
        raw_field_name="Country", field_type="country",
    )
    assert classified.klass == NOT_IN_TABLE


def test_every_class_is_one_of_the_four_and_never_unknown():
    samples = [
        classify_missing_source([], raw_field_name="Age"),
        classify_missing_source(
            [AGE_TABLE], group="mie", cohort_display="MIE",
            cohorts={"mie": ["MIE"]}, raw_field_name="Age",
        ),
        classify_missing_source(
            [AGE_TABLE], group="mie", cohort_display="MIE",
            cohorts={"mie": ["MIE"]}, raw_field_name="Country",
        ),
        classify_missing_source(
            [AGE_TABLE, _table(
                "x", "Table x", [["", "MIE"]], [["Age", "1"]],
            )],
            group="mie", cohort_display="MIE",
            cohorts={"mie": ["MIE"]}, raw_field_name="Age",
        ),
    ]
    klasses = {item.klass for item in samples}
    assert klasses <= TABLE_LOOKUP_CLASSES
    assert "unknown" not in klasses


def test_last_run_missing_source_rows_are_all_classified():
    """Contract 13 P4: the previous run's PMC misses must not stay unknown.

    22 missing_source rows (Capovilla 15 + Li 2025 7). Li 2015 is
    unresolved_source and is out of this set. Tables exist for both PMC
    papers; the designed unique-hit almost never fires.
    """
    from pathlib import Path

    from react_review.steps.paper_verification.fulltext_retriever import (
        pmc_xml_to_tables,
    )

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "pmc_tables"
    tables = {
        "capovilla_2023": pmc_xml_to_tables(
            (fixtures / "capovilla_2023.xml").read_text(encoding="utf-8")),
        "li_2025": pmc_xml_to_tables(
            (fixtures / "li_2025.xml").read_text(encoding="utf-8")),
    }
    assert tables["capovilla_2023"] and tables["li_2025"]
    mie_e = "Minimally Invasive Esophagectomy (MIE) Events"
    mie_t = "Minimally Invasive Esophagectomy (MIE) Total"
    oe_e = "Open Esophagectomy (OE) Events"
    oe_t = "Open Esophagectomy (OE) Total"
    claims = [
        ("capovilla_2023", "Age", "age"),
        ("capovilla_2023", "N MIE", "subgroup_n"),
        ("capovilla_2023", "N OE", "subgroup_n"),
        ("capovilla_2023", mie_e, "events"),
        ("capovilla_2023", mie_t, "subgroup_n"),
        ("capovilla_2023", oe_e, "events"),
        ("capovilla_2023", oe_t, "subgroup_n"),
        ("capovilla_2023", mie_e, "events"),
        ("capovilla_2023", mie_t, "subgroup_n"),
        ("capovilla_2023", oe_e, "events"),
        ("capovilla_2023", oe_t, "subgroup_n"),
        ("capovilla_2023", mie_e, "events"),
        ("capovilla_2023", mie_t, "subgroup_n"),
        ("capovilla_2023", oe_e, "events"),
        ("capovilla_2023", oe_t, "subgroup_n"),
        ("li_2025", mie_e, "events"),
        ("li_2025", oe_e, "events"),
        ("li_2025", oe_t, "subgroup_n"),
        ("li_2025", mie_e, "events"),
        ("li_2025", oe_e, "events"),
        ("li_2025", mie_e, "events"),
        ("li_2025", mie_e, "events"),
    ]
    assert len(claims) == 22
    dist: dict[str, int] = {}
    for study_id, raw, field_type in claims:
        item = classify_missing_source(
            tables[study_id], group="all", raw_field_name=raw,
            field_type=field_type,
        )
        assert item.klass in TABLE_LOOKUP_CLASSES
        assert item.klass != "unknown"
        dist[item.klass] = dist.get(item.klass, 0) + 1
    assert NO_TABLES not in dist
    assert LOCATED_NOT_READ not in dist
    assert dist == {FALLBACK_TEXT: 2, NOT_IN_TABLE: 20}


@pytest.mark.asyncio
async def test_collector_classifies_missing_source_from_document_tables():
    quote = "Methods text with no numbers."
    captured = AGE_TABLE
    document = PaperDocument(
        paper_id="fixture",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text=quote,
        tables=[captured],
        document_scope=DocumentScope.FULL_TEXT,
    )

    class Retriever(PaperRetriever):
        captured_tables = [captured]

        async def retrieve(self, reference):
            return document.model_copy(update={"reference": reference})

    class Extract(Tool):
        name = "extract_source_value"
        stage = ToolStage.EXTRACT

        async def run(self, payload):
            assert payload.document.tables
            return SourceValueResult(found=False, not_found_reason="not printed")

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(Retriever()))
    registry.register(Extract())
    collector = Collector(
        registry,
        cohorts=CohortRegistry(labels=[
            CohortLabel(key="mie", display="MIE"),
            CohortLabel(key="oe", display="OE"),
        ]),
        max_attempts=1,
    )
    review = ReviewDataItem(
        review_data_id="A_01", study_id="fixture", group="mie",
        field_type="age", raw_field_name="Age", value="77",
        cohort_label="MIE",
    )
    result = await collector.collect(
        review, ReferenceEntry(title="t", doi="10.0/x"))
    item = result.source_item
    assert item.collection_outcome is CollectionOutcome.MISSING_SOURCE
    codes = {r.code for r in item.reasons}
    assert LOCATED_NOT_READ in codes
    assert "unknown" not in codes
    lookup = next(r for r in item.reasons if r.code == LOCATED_NOT_READ)
    assert lookup.source == "deterministic"


def _li_2025():
    from pathlib import Path

    from react_review.steps.paper_verification.fulltext_retriever import (
        FullTextRetriever,
        pmc_xml_to_tables,
    )

    xml = (Path(__file__).resolve().parents[1] / "fixtures" / "pmc_tables"
           / "li_2025.xml").read_text(encoding="utf-8")
    return pmc_xml_to_tables(xml), FullTextRetriever._pmc_xml_to_text(xml)


def test_c02_all_age_is_the_total_median_with_column_header():
    """[C_02] must name the Total column, not a flattened row of five ages."""
    tables, text = _li_2025()
    hit = locate_source_cell(
        tables, group="all", raw_field_name="Age", field_type="age",
        document_text=text,
    )
    assert hit.unique
    assert hit.row_label == "Age, years / median (range)"
    assert "Total" in hit.column_header
    assert "MIE" not in hit.column_header
    assert hit.quote == hit.value
    assert "0.265" not in hit.quote
    assert hit.quote != text
    from react_review.normalize.anchors import normalised_contains
    assert normalised_contains(text, hit.quote)


def test_two_mie_columns_refuse_rather_than_pick_before_or_after_psm():
    tables, text = _li_2025()
    hit = locate_source_cell(
        tables, group="mie", cohort_display="MIE",
        cohorts={"mie": ["MIE"]}, raw_field_name="Age", field_type="age",
        document_text=text,
    )
    assert not hit.unique
    assert hit.status == "ambiguous"


def test_empty_tables_leave_the_text_path_untouched():
    hit = locate_source_cell(
        [], group="all", raw_field_name="Age", field_type="age",
        document_text="Age 73",
    )
    assert hit.status == "no_tables"
    assert not hit.unique


def test_two_statistic_followers_are_not_a_guess():
    table = _table(
        "Tab1", "Table 1",
        [["Characteristic", "Total"]],
        [
            ["Age, years", ""],
            ["median (range)", "73"],
            ["mean ± SD", "71 ± 4"],
        ],
    )
    hit = locate_source_cell(
        [table], group="all", raw_field_name="Age", field_type="age",
        document_text="median (range) 73 mean 71",
    )
    assert not hit.unique


def test_direct_cell_does_not_look_at_the_next_row():
    table = _table(
        "Tab1", "Table 1",
        [["", "MIE"]],
        [["Age (years)", "77.2"], ["median (range)", "73"]],
    )
    hit = locate_source_cell(
        [table], group="mie", cohort_display="MIE",
        cohorts={"mie": ["MIE"]}, raw_field_name="Age", field_type="age",
        document_text="Age (years) 77.2 median (range) 73",
    )
    assert hit.unique
    assert hit.value == "77.2"
    assert hit.row_label == "Age (years)"


@pytest.mark.asyncio
async def test_extract_tool_returns_the_cell_without_calling_the_model():
    tables, text = _li_2025()
    document = PaperDocument(
        paper_id="li_2025",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text=text,
        tables=tables,
        document_scope=DocumentScope.FULL_TEXT,
    )

    class _MustNotRun:
        model_id = "unused"

        async def complete(self, prompt, seed=0):
            raise AssertionError("B1 unique cell must not call the model")

    tool = ExtractSourceValueTool(_MustNotRun())
    result = await tool.run(ExtractSourceValueInput(
        document=document, field_type="age", group="all",
        raw_field_name="Age", concept="age",
    ))
    assert result.found
    assert result.row_label == "Age, years / median (range)"
    assert "Total" in result.column_header
    assert result.quote == result.value
    assert result.table_caption
    assert "Age, years / median (range)" in result.location
    assert result.value_origin == "verbatim"


@pytest.mark.asyncio
async def test_extract_tool_falls_through_when_tables_are_empty():
    document = PaperDocument(
        paper_id="x",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text="Age was 55 years in the cohort.",
        document_scope=DocumentScope.FULL_TEXT,
    )

    class _Answers:
        model_id = "scripted"

        async def complete(self, prompt, seed=0):
            return ('{"found": true, "value": "55", "unit": "years", '
                    '"quote": "Age was 55 years in the cohort.", '
                    '"group_label_in_paper": "cohort"}')

    tool = ExtractSourceValueTool(_Answers())
    result = await tool.run(ExtractSourceValueInput(
        document=document, field_type="age", group="all",
        raw_field_name="Age",
    ))
    assert result.found
    assert result.value == "55"
    assert not result.row_label
    assert not result.column_header


def _capovilla():
    from pathlib import Path

    from react_review.steps.paper_verification.fulltext_retriever import (
        FullTextRetriever,
        pmc_xml_to_tables,
    )

    xml = (Path(__file__).resolve().parents[1] / "fixtures" / "pmc_tables"
           / "capovilla_2023.xml").read_text(encoding="utf-8")
    return pmc_xml_to_tables(xml), FullTextRetriever._pmc_xml_to_text(xml)


def _arm_cohorts():
    return {"mie": ["MIE"], "oe": ["OE"]}


def test_li_2025_mie_subgroup_n_is_after_psm_header_n():
    tables, text = _li_2025()
    hit = locate_source_cell(
        tables, group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), raw_field_name="N MIE",
        field_type="subgroup_n", document_text=text,
    )
    assert hit.unique
    assert hit.value == "92"
    folded = hit.quote.lower()
    assert "n" in folded and "92" in hit.quote
    assert "After PSM" in hit.column_header
    assert "MIE" in hit.column_header
    from react_review.normalize.anchors import normalised_contains
    assert normalised_contains(text, hit.quote)


def test_li_2025_oe_subgroup_n_is_after_psm_header_n():
    tables, text = _li_2025()
    hit = locate_source_cell(
        tables, group="oe", cohort_display="OE",
        cohorts=_arm_cohorts(), raw_field_name="N OE",
        field_type="subgroup_n", document_text=text,
    )
    assert hit.unique
    assert hit.value == "55"
    assert "n" in hit.quote.lower() and "55" in hit.quote
    assert "After PSM" in hit.column_header
    from react_review.normalize.anchors import normalised_contains
    assert normalised_contains(text, hit.quote)


def test_capovilla_subgroup_n_refuses_rather_than_inventing_n():
    tables, text = _capovilla()
    open_cohorts = {
        "mie": ["MIE", "Minimally Invasive Esophagectomy"],
        "oe": ["OE", "Open Esophagectomy"],
    }
    for group, display, raw in (
            ("mie", "MIE", "N MIE"), ("oe", "OE", "N OE")):
        hit = locate_source_cell(
            tables, group=group, cohort_display=display,
            cohorts=open_cohorts, raw_field_name=raw,
            field_type="subgroup_n", document_text=text,
        )
        assert not hit.unique, group


def test_two_n_equals_in_one_column_path_are_refused():
    table = _table(
        "Tab1", "Table 1",
        [["", "MIE (n=10) (n=20)"]],
        [["Age", "55"]],
    )
    hit = locate_source_cell(
        [table], group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), raw_field_name="N MIE",
        field_type="subgroup_n", document_text="MIE (n=10) (n=20) Age 55",
    )
    assert not hit.unique


def test_anastomotic_leak_hits_the_leakage_table_not_a_guessed_row():
    """L2 unique-tables Tab3; two leakage rows still refuse the cell."""
    tables, text = _li_2025()
    locate = locate_source_table(
        tables, group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), field_type="events",
        outcome="Anastomotic leak",
        raw_field_name="Minimally Invasive Esophagectomy (MIE) Events",
    )
    assert locate.unique
    assert locate.table is not None
    assert locate.table.table_id == "Tab3"
    assert any(_field_needle_in("Anastomotic leak", row[0])
               for row in locate.table.rows)
    hit = locate_source_cell(
        tables, group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), field_type="events",
        outcome="Anastomotic leak",
        raw_field_name="Minimally Invasive Esophagectomy (MIE) Events",
        document_text=text,
    )
    assert not hit.unique
    label = next(
        (row[0] or "").strip() for row in locate.table.rows
        if "leakage" in (row[0] or "").lower()
        and "suspected" not in (row[0] or "").lower()
    )
    mapped = locate_source_cell(
        tables, group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), field_type="events",
        outcome="Anastomotic leak",
        raw_field_name="Minimally Invasive Esophagectomy (MIE) Events",
        mapped_row=label, document_text=text,
    )
    assert mapped.unique
    assert mapped.value.startswith("13")
    assert mapped.row_label == label
    assert "After PSM" in mapped.column_header


def test_leakage_needle_does_not_hit_a_leak_row():
    table = _table(
        "Tab2", "Complications",
        [["", "MIE"]],
        [["Anastomotic leak", "7 (12)"]],
    )
    locate = locate_source_table(
        [table], group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), field_type="events",
        outcome="Anastomotic leakage",
    )
    assert not locate.unique
    assert locate.status == "none"


def test_cohort_axis_does_not_prefix_match_placeboid():
    """Placebo ⊂ Placeboid would be a silent arm swap if the cohort axis prefixed."""
    table = _table(
        "near", "Table 1 Baseline",
        [["Characteristic", "Placeboid (n=20)"]],
        [["Age (years)", "55 ± 8"]],
    )
    locate = locate_source_table(
        [table], group="placebo", cohort_display="Placebo",
        cohorts={"placebo": ["Placebo"]},
        raw_field_name="Age", field_type="age",
    )
    assert not locate.unique
    assert locate.status == "none"
    assert _needle_in("Placebo", "Placeboid (n=20)") is False
    assert _field_needle_in("leak", "anastomotic leakage") is True


def test_cohort_needle_in_has_no_prefix_matching():
    import inspect

    source = inspect.getsource(_needle_in)
    assert "startswith" not in source
    assert "tokens <= haystack" in source
    field_source = inspect.getsource(_field_needle_in)
    assert "startswith" in field_source


def test_short_field_token_does_not_prefix_match():
    """``inf`` must not hit ``infection``; min prefix length is 4."""
    assert _field_needle_in("inf", "Pulmonary infection") is False
    assert _field_needle_in("leak", "Anastomotic leakage") is True


def test_mapped_row_pulmonary_infection_takes_after_psm_cell():
    tables, text = _li_2025()
    tab3 = next(t for t in tables if t.table_id == "Tab3")
    mie = locate_source_cell(
        [tab3], group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), field_type="events",
        raw_field_name="Pulmonary", outcome="Pulmonary complications",
        mapped_row="Pulmonary infection", document_text=text,
    )
    oe = locate_source_cell(
        [tab3], group="oe", cohort_display="OE",
        cohorts=_arm_cohorts(), field_type="events",
        raw_field_name="Pulmonary", outcome="Pulmonary complications",
        mapped_row="Pulmonary infection", document_text=text,
    )
    assert mie.unique and oe.unique
    assert mie.value.startswith("22")
    assert oe.value.startswith("21")
    assert "After PSM" in mie.column_header and "MIE" in mie.column_header
    assert "After PSM" in oe.column_header and "OE" in oe.column_header
    assert mie.row_label == "Pulmonary infection"


def test_mapped_row_none_unknown_or_number_is_refused():
    tables, text = _li_2025()
    tab3 = next(t for t in tables if t.table_id == "Tab3")
    labels = [(row[0] or "").strip() for row in tab3.rows]
    assert interpret_row_map_response("NONE", labels) is None
    assert interpret_row_map_response("Not a row", labels) is None
    assert interpret_row_map_response("22", labels) is None
    assert interpret_row_map_response('{"value": "22"}', labels) is None
    for mapped in ("NONE", "Not a row", "22"):
        hit = locate_source_cell(
            [tab3], group="mie", cohort_display="MIE",
            cohorts=_arm_cohorts(), field_type="events",
            raw_field_name="Pulmonary", outcome="Pulmonary complications",
            mapped_row=mapped, document_text=text,
        )
        assert not hit.unique


def test_row_map_prompt_asks_for_a_label_not_a_value():
    prompt = render_source_row_map_prompt(
        outcome="Pulmonary complications",
        row_labels=["Anastomotic leakage", "Pulmonary infection"],
    )
    assert "Pulmonary complications" in prompt
    assert "Pulmonary infection" in prompt
    assert "NONE" in prompt
    assert "cell value" in prompt.lower() or "Do not report any number" in prompt


def test_source_row_map_contract_pins_the_rendered_prompt():
    import hashlib
    import json
    from pathlib import Path

    from react_review.tools.source_table_lookup import SOURCE_ROW_MAP_VERSION

    body = json.loads(
        (Path(__file__).resolve().parents[2]
         / "configs/prompt_contracts/source_row_map_v1.json"
         ).read_text(encoding="utf-8"))
    assert body["prompt_version"] == SOURCE_ROW_MAP_VERSION
    fixture = body["fixture_inputs"]
    prompt = render_source_row_map_prompt(
        outcome=fixture["outcome"], row_labels=list(fixture["row_labels"]))
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()
    assert digest == body["rendered_prompt_sha256"]


def test_source_row_map_request_forbids_value_and_review_fields():
    from pydantic import ValidationError

    from react_review.tools.extract_source import SourceRowMapRequest

    SourceRowMapRequest(outcome="Pulmonary complications",
                        row_labels=["Pulmonary infection"])
    for extra in (
            {"value": "22"},
            {"review_value": "22"},
            {"review_data_id": "C_09"},
    ):
        with pytest.raises(ValidationError):
            SourceRowMapRequest(
                outcome="x", row_labels=["y"], **extra)


def test_30_day_mortality_does_not_force_match_death_on_li_2025():
    tables, text = _li_2025()
    locate = locate_source_table(
        tables, group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), field_type="events",
        outcome="30-day mortality",
        raw_field_name="Minimally Invasive Esophagectomy (MIE) Events",
    )
    assert not locate.unique
    hit = locate_source_cell(
        tables, group="mie", cohort_display="MIE",
        cohorts=_arm_cohorts(), field_type="events",
        outcome="30-day mortality",
        raw_field_name="Minimally Invasive Esophagectomy (MIE) Events",
        document_text=text,
    )
    assert not hit.unique


@pytest.mark.asyncio
async def test_row_map_stub_pulmonary_infection_skips_the_extract_model():
    tables, text = _li_2025()
    tab3 = next(t for t in tables if t.table_id == "Tab3")
    label = next(
        (row[0] or "").strip() for row in tab3.rows
        if (row[0] or "").strip() == "Pulmonary infection"
    )
    document = PaperDocument(
        paper_id="li_2025",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text=text,
        tables=[tab3],
        document_scope=DocumentScope.FULL_TEXT,
    )

    class _MustNotExtract:
        model_id = "unused"

        async def complete(self, prompt, seed=0):
            raise AssertionError("L3 unique cell must not call extract")

    class _RowMap:
        model_id = "stub-row-map"

        async def complete(self, prompt, seed=0):
            assert "Pulmonary complications" in prompt
            assert "Do not report any number" in prompt
            return label

    tool = ExtractSourceValueTool(_MustNotExtract(), row_map_backend=_RowMap())
    mie = await tool.run(ExtractSourceValueInput(
        document=document, field_type="events", group="mie",
        raw_field_name="Pulmonary", outcome="Pulmonary complications",
        cohort_display="MIE", cohorts=_arm_cohorts(),
    ))
    oe = await tool.run(ExtractSourceValueInput(
        document=document, field_type="events", group="oe",
        raw_field_name="Pulmonary", outcome="Pulmonary complications",
        cohort_display="OE", cohorts=_arm_cohorts(),
    ))
    assert mie.found and oe.found
    assert mie.value.startswith("22")
    assert oe.value.startswith("21")
    assert "After PSM" in mie.column_header
    assert mie.row_label == label


@pytest.mark.asyncio
async def test_row_map_stub_none_or_number_falls_through():
    tables, text = _li_2025()
    tab3 = next(t for t in tables if t.table_id == "Tab3")
    document = PaperDocument(
        paper_id="li_2025",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text=text,
        tables=[tab3],
        document_scope=DocumentScope.FULL_TEXT,
    )

    class _Answers:
        model_id = "scripted"

        async def complete(self, prompt, seed=0):
            return ('{"found": false, "value": null, "quote": "", '
                    '"not_found_reason": "not in excerpt"}')

    class _NoneRow:
        model_id = "stub"

        async def complete(self, prompt, seed=0):
            return "NONE"

    tool = ExtractSourceValueTool(_Answers(), row_map_backend=_NoneRow())
    result = await tool.run(ExtractSourceValueInput(
        document=document, field_type="events", group="mie",
        raw_field_name="Pulmonary", outcome="Pulmonary complications",
        cohort_display="MIE", cohorts=_arm_cohorts(),
    ))
    assert not result.found
    assert not result.row_label


@pytest.mark.asyncio
async def test_row_map_stub_number_is_not_used_as_a_value():
    tables, text = _li_2025()
    tab3 = next(t for t in tables if t.table_id == "Tab3")
    document = PaperDocument(
        paper_id="li_2025",
        reference=ReferenceEntry(title="t", doi="10.0/x"),
        full_text=text,
        tables=[tab3],
        document_scope=DocumentScope.FULL_TEXT,
    )

    class _Answers:
        model_id = "scripted"
        prompts: list = []

        async def complete(self, prompt, seed=0):
            self.prompts.append(prompt)
            return ('{"found": false, "value": null, "quote": "", '
                    '"not_found_reason": "not in excerpt"}')

    class _Number:
        model_id = "stub"

        async def complete(self, prompt, seed=0):
            return "22"

    extract = _Answers()
    tool = ExtractSourceValueTool(extract, row_map_backend=_Number())
    result = await tool.run(ExtractSourceValueInput(
        document=document, field_type="events", group="mie",
        raw_field_name="Pulmonary", outcome="Pulmonary complications",
        cohort_display="MIE", cohorts=_arm_cohorts(),
    ))
    assert not result.found
    assert result.value != "22"
    assert extract.prompts, "must fall through to extract, not take the number"

