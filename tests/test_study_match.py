"""Tests for study-id resolution and the local PDF retriever."""
from __future__ import annotations

from pathlib import Path

import pytest

from react_review.csv_io import load_included_studies
from react_review.retrieval.local_pdf import LocalPdfRetriever
from react_review.schemas.evidence import ReviewDataItem
from react_review.steps.paper_verification.schemas import ReferenceEntry
from react_review.parser.review_parser import ParsedStudy
from react_review.study_match import (
    build_reference_resolver,
    build_reference_resolver_from_parsed,
    is_resolvable,
    resolve_studies,
    resolve_study,
)

BENCH = Path(__file__).resolve().parents[1] / "eval" / "benchmark_1"
STUDIES = load_included_studies(BENCH / "included_studies.csv")


@pytest.mark.parametrize(
    "parser_sid, canonical",
    [
        ("ahmad_2022", "ahmad_2022"),
        ("Ahmad et al. [2022]", "ahmad_2022"),       # table words, not a slug
        ("yaz_2011", "yazici_2011"),                 # non-ASCII slug prefix
        ("de_2018", "de_gonzalo_calvo_2018"),        # particle name, year disambiguates
        ("colom_2018", "colom_2018"),                # same year as de_… but surname differs
        ("iacobellis_2014", "iacobellis_2014"),
        ("el-baky_2023", "elbaky_2023"),             # hyphenated surname -> strip to match
    ],
)
def test_resolve_study(parser_sid, canonical):
    s = resolve_study(parser_sid, STUDIES)
    assert s is not None and s.study_id == canonical


def test_resolve_study_unknown_returns_none():
    assert resolve_study("wongabongo_1999", STUDIES) is None


def test_resolve_studies_relabels_and_maps():
    items = [
        ReviewDataItem(study_id="yaz_2011", group="t1dm", field_type="eat_thickness",
                       value="3.30", unit="mm"),
        ReviewDataItem(study_id="unknownstudy_2000", group="t1dm",
                       field_type="bmi", value="24"),
    ]
    relabelled, sid_map = resolve_studies(items, STUDIES)
    assert relabelled[0].study_id == "yazici_2011"      # relabelled
    assert relabelled[1].study_id == "unknownstudy_2000"  # unresolved kept as-is
    assert "yazici_2011" in sid_map

    ref = build_reference_resolver(sid_map)("yazici_2011")
    assert isinstance(ref, ReferenceEntry)
    assert ref.doi == "10.1007/s12020-011-9478-x"


def test_apply_modality_disambiguation():
    # DKB A2 fix: a CT study's eat_thickness becomes eat_volume; echo stays.
    from react_review.dkb import KnowledgeBase
    from react_review.schemas.resolution import FieldResolutionRecord, ResolutionCellRef
    from react_review.study_match import apply_modality_disambiguation

    kb = KnowledgeBase.from_json(BENCH.parent.parent / "configs" / "knowledge.seed.json")
    sid_map = {s.study_id: s for s in STUDIES}
    items = [
        ReviewDataItem(study_id="svanteson_2019", group="t1dm",   # modality ct
                       field_type="eat_thickness", value="40", table_id="table_1",
                       cell_ref=(0, 2), resolution_key="eat-resolution"),
        ReviewDataItem(study_id="ahmad_2022", group="t1dm",       # modality echo
                       field_type="eat_thickness", value="6.6", unit="mm",
                       table_id="table_1", cell_ref=(1, 2),
                       resolution_key="eat-resolution"),
    ]
    resolutions = [FieldResolutionRecord(
        resolution_key="eat-resolution", raw_field_name="EFT/ EAT",
        field_type="eat_thickness", status="authoritative", source="deterministic",
        field_types_seen=["eat_thickness"],
        affected_cells=[
            ResolutionCellRef(table_id="table_1", cell_ref=(0, 2),
                              study_id="svanteson_2019", field_type="eat_thickness"),
            ResolutionCellRef(table_id="table_1", cell_ref=(1, 2),
                              study_id="ahmad_2022", field_type="eat_thickness"),
        ],
    )]
    out = apply_modality_disambiguation(items, sid_map, kb, resolutions)
    assert out[0].field_type == "eat_volume"        # CT → volume
    assert out[1].field_type == "eat_thickness"     # echo → unchanged
    assert resolutions[0].status == "mixed"
    assert resolutions[0].field_type is None
    assert {c.field_type for c in resolutions[0].affected_cells} == {
        "eat_volume", "eat_thickness"}
    assert resolutions[0].reasons[0].code == "modality_disambiguation"


def test_build_reference_resolver_from_parsed():
    studies = [
        ParsedStudy(study_id="ahmad_2022",
                    citation="Ahmad A. Epicardial fat in T1DM. J Cardiol. 2022.", doi="10.1/x"),
        ParsedStudy(study_id="aslan_2015",
                    citation="Aslan B. EFT and endothelial dysfunction. 2015.", doi=""),
    ]
    resolve = build_reference_resolver_from_parsed(studies)
    ahmad = resolve("ahmad_2022")
    assert ahmad.title.startswith("Ahmad A.") and ahmad.doi == "10.1/x"
    aslan = resolve("aslan_2015")
    assert aslan.doi is None                       # no printed DOI → reconciled online later
    # No citation for this study: the entry is MARKED rather than given the study
    # id as a title, which would send the resolver hunting for a paper by that name.
    missing = resolve("not_a_study_2000")
    assert missing.doi is None
    assert not is_resolvable(missing)
    assert "not_a_study_2000" in missing.title


def test_resolver_copies_a_printed_pmid():
    studies = [ParsedStudy(
        study_id="capovilla_2023",
        citation="Capovilla G. Front Oncol. 2023;13:1104109. PMID: 36726501",
        doi="", pmid="36726501")]
    ref = build_reference_resolver_from_parsed(studies)("capovilla_2023")
    assert ref.pmid == "36726501" and ref.doi is None


def test_reference_resolver_from_parsed_fuzzy_matches_slug_variants():
    # The data-table slug and the reference-list slug rarely align byte-for-byte;
    # a year + surname-prefix match still pairs them.
    studies = [
        ParsedStudy(study_id="yazici_2011", citation="Yazıcı D et al. 2011.", doi="10.9/y"),
        ParsedStudy(study_id="ahmad_2022", citation="Ahmad A et al. 2022.", doi="10.1/x"),
    ]
    resolve = build_reference_resolver_from_parsed(studies)
    assert resolve("yaz_2011").doi == "10.9/y"          # table slug 'yaz_2011' → 'yazici_2011'
    assert resolve("ahmad_2022").doi == "10.1/x"        # exact still works


def test_reference_resolver_from_parsed_ambiguous_does_not_guess():
    # Two 2020 studies whose surnames both start with the query prefix → ambiguous.
    studies = [
        ParsedStudy(study_id="smith_2020", citation="Smith A. 2020.", doi="10.1/a"),
        ParsedStudy(study_id="smithson_2020", citation="Smithson B. 2020.", doi="10.2/b"),
    ]
    resolve = build_reference_resolver_from_parsed(studies)
    ref = resolve("smit_2020")                          # prefix of BOTH → ambiguous
    assert ref.doi is None and not is_resolvable(ref)   # marked, never a wrong guess


def test_resolver_pairs_verbatim_table_labels_and_attaches_the_printed_doi():
    # doc05: two Li papers share a first word; the table's year + initials
    # pick one citation, and that citation's DOI rides along.
    studies = [
        ParsedStudy(
            study_id="li_2015",
            citation=("Li J, Shen Y, Tan L, et al. Is minimally invasive "
                      "esophagectomy beneficial to elderly patients with "
                      "esophageal cancer? Surg Endosc. 2015;29(4):925-930."),
            doi=""),
        ParsedStudy(
            study_id="capovilla_2023",
            citation=("Capovilla G, Uzun E, Scarton A, et al. Minimally invasive "
                      "Ivor Lewis esophagectomy in the elderly patient. "
                      "Front Oncol. 2023;13:1104109."),
            doi=""),
        ParsedStudy(
            study_id="li_2025",
            citation=("Li K, Lu S, Li C, et al. Long-term outcomes of minimally "
                      "invasive esophagectomy vs. open esophagectomy. "
                      "Langenbecks Arch Surg. 2025;410(1):311."),
            doi="10.1007/s00423-025-03877-4"),
    ]
    resolve = build_reference_resolver_from_parsed(studies)
    assert resolve("Li J et al. 2015").title.startswith("Li J")
    assert resolve("Li J et al. 2015").doi is None
    li_k = resolve("Li K et al. 2025")
    assert li_k.doi == "10.1007/s00423-025-03877-4"
    assert resolve("Capovilla G et al. 2023").title.startswith("Capovilla")
    missing = resolve("Li et al.")
    assert not is_resolvable(missing)


def test_a_citation_without_a_doi_is_still_worth_resolving():
    # Most reference lists print no DOIs; refusing those would discard the review.
    # Only a PLACEHOLDER (no citation at all) is unresolvable.
    studies = [ParsedStudy(study_id="aslan_2015",
                           citation="Aslan B. EFT and endothelial dysfunction. 2015.",
                           doi="")]
    resolve = build_reference_resolver_from_parsed(studies)
    assert is_resolvable(resolve("aslan_2015"))
    assert not is_resolvable(resolve("never_cited_1999"))


# --- LocalPdfRetriever ---

@pytest.mark.asyncio
async def test_local_retriever_no_mapping_returns_none():
    r = LocalPdfRetriever({}, base_dir=BENCH)
    assert await r.retrieve(ReferenceEntry(title="x", doi="10.1/x")) is None


@pytest.mark.asyncio
async def test_local_retriever_missing_file_returns_none():
    r = LocalPdfRetriever({"10.1/x": "pdf/does_not_exist.pdf"}, base_dir=BENCH)
    assert await r.retrieve(ReferenceEntry(title="x", doi="10.1/x")) is None


@pytest.mark.asyncio
async def test_local_retriever_reads_real_pdf_if_present():
    ahmad = next(s for s in STUDIES if s.study_id == "ahmad_2022")
    pdf_path = BENCH / ahmad.source_pdf
    if not pdf_path.is_file():
        pytest.skip("benchmark source PDFs are git-ignored / not present locally")
    r = LocalPdfRetriever({ahmad.doi: ahmad.source_pdf}, base_dir=BENCH)
    doc = await r.retrieve(ReferenceEntry(title="Ahmad", doi=ahmad.doi))
    assert doc is not None
    assert len(doc.full_text) > 1000
    assert doc.metadata["source"] == "local_pdf"
