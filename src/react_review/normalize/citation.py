"""Pull year and journal out of a printed citation, deterministically.

Title-search resolvers need these to reject a high-scoring wrong paper. The
parser stores the verbatim citation line; it does not already split journal
and year onto ``ReferenceEntry``.
"""
from __future__ import annotations

import re

from react_review.normalize.study_key import year_of

# Vancouver-style: "Journal Name. 2023;13:1104109" or "Journal Name. 2023."
_JOURNAL_YEAR = re.compile(
    r"([A-Za-z][A-Za-z0-9][A-Za-z0-9\s.\-']*?)\.\s*((?:19|20)\d{2})\s*[.;:]",
)


def citation_year(citation: str) -> int | None:
    """The publication year printed on the citation, if any."""
    raw = year_of(citation or "")
    return int(raw) if raw else None


def citation_journal(citation: str) -> str:
    """The journal segment immediately before the year, or empty.

    Vancouver lines print ``et al. Journal. YEAR;``. Searching the whole
    string would keep ``et al`` as part of the journal. The tail after the
    last ``et al.`` is the journal-year pair; abbreviations like ``Front
    Oncol`` are kept as printed — matching a full name is the gate's job.
    """
    text = citation or ""
    tail = re.split(r"\bet al\.\s*", text, flags=re.I)[-1]
    last = ""
    for match in _JOURNAL_YEAR.finditer(tail):
        last = re.sub(r"\s+", " ", match.group(1)).strip(" .,;")
    return re.sub(r"(?i)^et al\.?\s*", "", last).strip(" .,;")
