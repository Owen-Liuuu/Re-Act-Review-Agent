"""One derivation of a study id, shared by the parser and the matcher."""
from __future__ import annotations

import pytest

from react_review.normalize.study_key import (
    best_identity_match,
    identities_match,
    join_key,
    key_parts,
    study_key,
    surname_of,
)


@pytest.mark.parametrize("citation, expected", [
    ("Ahmad et al. [2022]", "ahmad_2022"),
    ("Keles et al. (2016)", "keles_2016"),
    ("El-Baky 2023", "elbaky_2023"),
    ("Smith J, Jones B. 2019. Lancet.", "smith_2019"),
])
def test_plain_citations(citation, expected):
    assert study_key(citation) == expected


def test_surname_particles_are_kept_so_distinct_papers_stay_distinct():
    # A first-word rule reduced both of these to "van_2020", after which every
    # claim from one paper was attributed to the other.
    assert study_key("van den Berg 2020") != study_key("van Rooij 2020")
    assert study_key("van den Berg 2020") == "vandenberg_2020"
    assert study_key("de Gonzalo-Calvo et al. (2018)") == "degonzalocalvo_2018"


@pytest.mark.parametrize("citation, expected", [
    ("Yazıcı et al. [2011]", "yazici_2011"),      # Turkish dotless i
    ("Grønbæk 2019", "gronbaek_2019"),            # Danish
    ("Wróblewski 2021", "wroblewski_2021"),       # Polish
    ("Müller 2015", "muller_2015"),               # combining umlaut
])
def test_names_that_ascii_folding_would_truncate(citation, expected):
    # These letters carry no combining accent, so NFKD alone leaves them and the
    # surname is cut short at the first one.
    assert study_key(citation) == expected


def test_a_real_collision_is_suffixed_not_silently_merged():
    taken = {"smith_2019"}
    assert study_key("Smith A. 2019", taken=taken) == "smith_2019_b"
    assert study_key("Smith A. 2019", taken=taken | {"smith_2019_b"}) == "smith_2019_c"


def test_citation_without_a_year_still_yields_a_key():
    assert study_key("Anonymous report") == "anonymous"
    assert study_key("") == "study"


def test_key_parts_round_trips_the_canonical_ids():
    # The matcher compares a parser key against included_studies ids; both sides
    # must reduce to the same (surname, year).
    assert key_parts("de_gonzalo_calvo_2018") == key_parts("degonzalocalvo_2018")
    assert key_parts("yazici_2011") == ("yazici", "2011")


def test_surname_stops_at_citation_boilerplate():
    assert surname_of("Ahmad et al.") == "ahmad"
    assert surname_of("Berg and Jones") == "berg"


def test_join_key_keeps_the_table_words_and_appends_year_only_when_missing():
    assert join_key("Li J et al.", "2015") == "Li J et al. 2015"
    assert join_key("  Li   J  et al.  ", "2015") == "Li J et al. 2015"
    # A year already in the cell is left where the review printed it.
    assert join_key("Ahmad et al. [2022]", "2019") == "Ahmad et al. [2022]"
    assert join_key("Ahmad 2022") == "Ahmad 2022"
    assert join_key("Capovilla G et al.", "2023") == "Capovilla G et al. 2023"


def test_identities_match_pairs_table_words_to_citation_slugs_by_year():
    assert identities_match("Li J et al. 2015", "li_2015")
    assert not identities_match("Li J et al. 2015", "li_2025")
    assert identities_match("Ahmad et al. [2022]", "ahmad_2022")
    assert identities_match("yaz_2011", "yazici_2011")
    assert identities_match("van den Berg 2020", "vandenberg_2020")
    assert not identities_match("van den Berg 2020", "vanrooij_2020")


def test_best_identity_match_keeps_same_surname_papers_distinct():
    li_papers = [
        ("li_2015", "Li J, Shen Y, Tan L, et al. Surg Endosc. 2015;29(4):925-930."),
        ("li_2025", "Li K, Lu S, Li C, et al. Langenbecks Arch Surg. 2025;410(1):311."),
        ("capovilla_2023", "Capovilla G, Uzun E, Scarton A, et al. Front Oncol. 2023;13:1104109."),
    ]
    assert best_identity_match("Li J et al. 2015", li_papers) == "li_2015"
    assert best_identity_match("Li K et al. 2025", li_papers) == "li_2025"
    assert best_identity_match("Capovilla G et al. 2023", li_papers) == "capovilla_2023"
    # No year, two Li papers → refuse rather than guess.
    assert best_identity_match("Li et al.", li_papers) is None
    # An outcome row is not a cited paper.
    assert best_identity_match("Overall Complications", li_papers) is None


def test_best_identity_match_prefers_smith_over_smithson():
    papers = [("smith_2020", "Smith A. 2020."), ("smithson_2020", "Smithson B. 2020.")]
    assert best_identity_match("Smith 2020", papers) == "smith_2020"
    assert best_identity_match("smit_2020", papers) is None
