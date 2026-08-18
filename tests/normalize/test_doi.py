"""Printed DOI / PMID are copied from the review, never invented."""
from __future__ import annotations

from react_review.normalize.doi import printed_doi, printed_pmid


def test_printed_doi_keeps_a_doi_that_the_reference_list_shows():
    refs = "1. Ahmad A. 2022. J Cardiol. doi:10.1007/s00246-021-02811-x"
    assert printed_doi(
        "https://doi.org/10.1007/s00246-021-02811-x",
        "Ahmad A. 2022. J Cardiol.",
        refs,
    ) == "10.1007/s00246-021-02811-x"


def test_printed_doi_drops_a_guessed_frontiers_doi():
    citation = (
        "Capovilla G, Uzun E, Scarton A, et al. Minimally invasive Ivor Lewis "
        "esophagectomy in the elderly patient. Front Oncol. 2023;13:1104109."
    )
    assert printed_doi("10.3389/fonc.2023.1104109", citation, citation) == ""


def test_printed_doi_copies_the_doi_off_the_citation_when_the_model_omits_it():
    citation = "Li K et al. Langenbecks Arch Surg. 2025. doi:10.1007/s00423-025-03877-4"
    assert printed_doi("", citation) == "10.1007/s00423-025-03877-4"


def test_printed_pmid_copies_digits_from_the_citation_line():
    assert printed_pmid("Li J et al. Surg Endosc. 2015. PMID: 25294532") == "25294532"
    assert printed_pmid("Front Oncol. 2023;13:1104109.") == ""
