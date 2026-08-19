"""Year and journal pulled from a printed Vancouver citation."""
from __future__ import annotations

from react_review.normalize.citation import citation_journal, citation_year

LI_J = "Li J, Shen Y, Tan L, et al. Surg Endosc. 2015;29(4):925-930."
CAPOVILLA = "Capovilla G, Uzun E, Scarton A, et al. Front Oncol. 2023;13:1104109."
LI_K = ("Li K, Lu S, Li C, et al. Langenbecks Arch Surg. 2025;410(1):311. "
        "doi:10.1007/s00423-025-03877-4")


def test_doc05_citations_yield_year_and_journal():
    assert citation_year(LI_J) == 2015
    assert citation_journal(LI_J) == "Surg Endosc"
    assert citation_year(CAPOVILLA) == 2023
    assert citation_journal(CAPOVILLA) == "Front Oncol"
    assert citation_year(LI_K) == 2025
    assert citation_journal(LI_K) == "Langenbecks Arch Surg"
