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
    resolve_studies,
    resolve_study,
)

BENCH = Path(__file__).resolve().parents[1] / "eval" / "benchmark"
STUDIES = load_included_studies(BENCH / "included_studies.csv")


@pytest.mark.parametrize(
    "parser_sid, canonical",
    [
        ("ahmad_2022", "ahmad_2022"),
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
    from react_review.study_match import apply_modality_disambiguation

    kb = KnowledgeBase.from_json(BENCH.parent.parent / "configs" / "knowledge.seed.json")
    sid_map = {s.study_id: s for s in STUDIES}
    items = [
        ReviewDataItem(study_id="svanteson_2019", group="t1dm",   # modality ct
                       field_type="eat_thickness", value="40"),
        ReviewDataItem(study_id="ahmad_2022", group="t1dm",       # modality echo
                       field_type="eat_thickness", value="6.6", unit="mm"),
    ]
    out = apply_modality_disambiguation(items, sid_map, kb)
    assert out[0].field_type == "eat_volume"        # CT → volume
    assert out[1].field_type == "eat_thickness"     # echo → unchanged


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
    missing = resolve("not_a_study_2000")
    assert missing.title == "not_a_study_2000" and missing.doi is None


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
    assert ref.title == "smit_2020" and ref.doi is None  # minimal fallback, no wrong guess


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
