"""One derivation of a study id, shared by the parser and the matcher."""
from __future__ import annotations

import pytest

from react_review.normalize.study_key import key_parts, study_key, surname_of


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
