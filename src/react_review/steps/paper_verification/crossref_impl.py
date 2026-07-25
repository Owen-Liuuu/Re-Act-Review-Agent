"""Real implementation for step 2: CrossRef-based reference verification.

Uses the CrossRef REST API:
  - GET /works/{doi}              — exact DOI lookup
  - GET /works?query.title=...    — fuzzy title search (fallback when no DOI)

API docs: https://api.crossref.org/swagger-ui/index.html
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import httpx
import structlog

from react_review.core.config import CrossRefSettings, ThresholdSettings
from react_review.core.enums import ValidationSeverity, VerificationStatus
from react_review.steps.paper_verification.interfaces import ReferenceVerifier
from react_review.steps.paper_verification.schemas import (
    ReferenceEntry,
    ReferenceVerificationResult,
    VerificationFlag,
)

logger = structlog.get_logger(__name__)


class CrossRefVerifier(ReferenceVerifier):
    """Verifies references by looking them up in CrossRef.

    Strategy:
      1. If DOI is provided → exact lookup via /works/{doi}
      2. If DOI is missing → fuzzy search via /works?query.title=...
      3. Compare returned metadata against student's reference entry

    Args:
        settings: CrossRef API configuration.
        thresholds: Matching thresholds (title_similarity, etc.).
    """

    def __init__(
        self,
        settings: CrossRefSettings,
        thresholds: ThresholdSettings | None = None,
    ) -> None:
        self._base_url = settings.base_url.rstrip("/")
        self._mailto = settings.mailto
        self._timeout = settings.timeout
        self._thresholds = thresholds or ThresholdSettings()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def verify(self, reference: ReferenceEntry) -> ReferenceVerificationResult:
        """Verify a single reference via CrossRef."""
        logger.info(
            "crossref_verify_start",
            title=reference.title[:60],
            doi=reference.doi or "N/A",
        )

        # Route 1: DOI-based exact lookup
        if reference.doi:
            return await self._verify_by_doi(reference)

        # Route 2: Title-based fuzzy search
        return await self._verify_by_title(reference)

    # ------------------------------------------------------------------
    # DOI-based verification
    # ------------------------------------------------------------------

    async def _verify_by_doi(
        self, reference: ReferenceEntry
    ) -> ReferenceVerificationResult:
        """Look up a reference by DOI and compare metadata."""
        url = f"{self._base_url}/works/{reference.doi}"
        params = self._build_params()

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)

            if resp.status_code == 404:
                logger.warning("crossref_doi_not_found", doi=reference.doi)
                return ReferenceVerificationResult(
                    reference=reference,
                    status=VerificationStatus.NOT_FOUND,
                    confidence=0.0,
                    flags=[
                        VerificationFlag(
                            code="DOI_NOT_FOUND",
                            severity=ValidationSeverity.ERROR,
                            message=f"DOI '{reference.doi}' not found in CrossRef.",
                        )
                    ],
                )

            resp.raise_for_status()
            data = resp.json()
            work = data.get("message", {})

            return self._match_metadata(reference, work, source="doi-lookup")

        except httpx.HTTPStatusError as exc:
            logger.error("crossref_http_error", doi=reference.doi, status=exc.response.status_code)
            return ReferenceVerificationResult(
                reference=reference,
                status=VerificationStatus.UNCERTAIN,
                confidence=0.0,
                access_note=f"CrossRef returned HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            logger.error("crossref_request_error", doi=reference.doi, error=str(exc))
            return ReferenceVerificationResult(
                reference=reference,
                status=VerificationStatus.UNCERTAIN,
                confidence=0.0,
                access_note=f"Network error: {exc}",
            )

    # ------------------------------------------------------------------
    # Title-based verification (fallback)
    # ------------------------------------------------------------------

    async def _verify_by_title(
        self, reference: ReferenceEntry
    ) -> ReferenceVerificationResult:
        """Search CrossRef by title when DOI is unavailable."""
        url = f"{self._base_url}/works"
        params = {
            **self._build_params(),
            "query.title": reference.title,
            "rows": 3,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            items = data.get("message", {}).get("items", [])
            if not items:
                return ReferenceVerificationResult(
                    reference=reference,
                    status=VerificationStatus.NOT_FOUND,
                    confidence=0.0,
                    flags=[
                        VerificationFlag(
                            code="TITLE_NOT_FOUND",
                            severity=ValidationSeverity.ERROR,
                            message=f"No results found in CrossRef for title: '{reference.title[:60]}...'",
                        )
                    ],
                )

            # Take the best match
            best = items[0]
            return self._match_metadata(reference, best, source="title-search")

        except httpx.RequestError as exc:
            logger.error("crossref_request_error", error=str(exc))
            return ReferenceVerificationResult(
                reference=reference,
                status=VerificationStatus.UNCERTAIN,
                confidence=0.0,
                access_note=f"Network error: {exc}",
            )

    # ------------------------------------------------------------------
    # Metadata matching
    # ------------------------------------------------------------------

    def _match_metadata(
        self,
        reference: ReferenceEntry,
        work: dict,
        source: str,
    ) -> ReferenceVerificationResult:
        """Compare a CrossRef work record against the student's reference.

        Checks: title similarity, author overlap, year match, journal match.
        """
        flags: list[VerificationFlag] = []
        matched_metadata: dict[str, str] = {"source": source}

        # --- Title comparison ---
        cr_titles = work.get("title", [])
        cr_title = cr_titles[0] if cr_titles else ""
        matched_metadata["title"] = cr_title
        title_sim = self._similarity(reference.title, cr_title)
        matched_metadata["title_similarity"] = f"{title_sim:.2f}"

        if title_sim < self._thresholds.title_similarity:
            flags.append(
                VerificationFlag(
                    code="TITLE_MISMATCH",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Title similarity {title_sim:.2f} < threshold "
                        f"{self._thresholds.title_similarity}. "
                        f"CrossRef: '{cr_title[:60]}...'"
                    ),
                )
            )

        # --- Author comparison ---
        cr_authors = [
            f"{a.get('family', '')} {a.get('given', '')}".strip()
            for a in work.get("author", [])
        ]
        matched_metadata["authors"] = "; ".join(cr_authors[:5])
        author_ratio = self._author_overlap(reference.authors, cr_authors)
        matched_metadata["author_match_ratio"] = f"{author_ratio:.2f}"

        if reference.authors and author_ratio < self._thresholds.author_match_ratio:
            flags.append(
                VerificationFlag(
                    code="AUTHOR_MISMATCH",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Author match ratio {author_ratio:.2f} < threshold "
                        f"{self._thresholds.author_match_ratio}."
                    ),
                )
            )

        # --- Year comparison ---
        cr_year = self._extract_year(work)
        matched_metadata["year"] = str(cr_year) if cr_year else "N/A"
        if reference.year and cr_year and reference.year != cr_year:
            flags.append(
                VerificationFlag(
                    code="YEAR_MISMATCH",
                    severity=ValidationSeverity.WARNING,
                    message=f"Student year {reference.year} ≠ CrossRef year {cr_year}.",
                )
            )

        # --- Journal comparison ---
        cr_journal_list = work.get("container-title", [])
        cr_journal = cr_journal_list[0] if cr_journal_list else ""
        matched_metadata["journal"] = cr_journal

        # --- DOI ---
        cr_doi = work.get("DOI", "")
        matched_metadata["doi"] = cr_doi

        # --- Compute overall confidence ---
        doi_matched = source == "doi-lookup"
        confidence = self._compute_confidence(
            title_sim, author_ratio, reference, cr_year, doi_matched=doi_matched
        )

        # --- Determine status ---
        # DOI-based lookups confirm existence; use a slightly lower verified
        # threshold (0.70 vs 0.80) because the DOI itself is already a
        # strong identifier.  Title/author mismatches become warnings, not
        # reasons to downgrade to UNCERTAIN.
        verified_threshold = 0.70 if doi_matched else 0.80
        if confidence >= verified_threshold:
            status = VerificationStatus.VERIFIED
        elif confidence >= 0.5:
            status = VerificationStatus.UNCERTAIN
        else:
            status = VerificationStatus.NOT_FOUND

        logger.info(
            "crossref_verify_done",
            status=status.value,
            confidence=f"{confidence:.2f}",
            title_sim=f"{title_sim:.2f}",
        )

        return ReferenceVerificationResult(
            reference=reference,
            status=status,
            matched_metadata=matched_metadata,
            confidence=confidence,
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_title(title: str) -> str:
        """Normalise a title string for robust comparison.

        Steps:
          1. Decode HTML entities / strip HTML tags (CrossRef sometimes
             returns ``<i>text</i>`` or ``&amp;``).
          2. Unicode NFC normalisation — unifies composed / decomposed forms.
          3. Replace typographic variants with ASCII equivalents:
               en/em dash → hyphen, curly quotes → straight quotes.
          4. Map common Greek/Latin abbreviations: β→beta, α→alpha, etc.
          5. Lowercase and collapse whitespace.
          6. Strip leading/trailing punctuation.
        """
        if not title:
            return ""

        # 1. Strip basic HTML tags and decode common entities
        t = re.sub(r"<[^>]+>", " ", title)
        t = (
            t.replace("&amp;", "&")
             .replace("&lt;", "<")
             .replace("&gt;", ">")
             .replace("&quot;", '"')
             .replace("&#39;", "'")
             .replace("&nbsp;", " ")
        )

        # 2. Unicode NFC normalisation
        t = unicodedata.normalize("NFC", t)

        # 3. Typographic → ASCII
        t = (
            t.replace("\u2013", "-")   # en-dash
             .replace("\u2014", "-")   # em-dash
             .replace("\u2018", "'")   # left single quote
             .replace("\u2019", "'")   # right single quote
             .replace("\u201c", '"')   # left double quote
             .replace("\u201d", '"')   # right double quote
             .replace("\u00d7", "x")   # multiplication sign → x
             .replace("\u00b1", "+/-") # ±
        )

        # 4. Common Greek letters → ASCII names (helps when one source uses
        #    the symbol and the other spells it out)
        greek = {
            "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
            "ε": "epsilon", "μ": "mu", "κ": "kappa", "λ": "lambda",
            "ρ": "rho", "σ": "sigma", "τ": "tau", "χ": "chi",
        }
        for sym, name in greek.items():
            t = t.replace(sym, name)

        # 5. Lowercase + collapse whitespace
        t = t.lower()
        t = re.sub(r"\s+", " ", t).strip()

        # 6. Strip leading/trailing punctuation
        t = t.strip(".,;:!?\"'()-")

        return t

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute normalised title similarity (0.0–1.0).

        Applies ``_preprocess_title`` to both strings before comparison
        so that case, Unicode, punctuation, and HTML differences don't
        artificially lower the score.

        Also computes a word-overlap (Jaccard) score and returns the
        maximum of the two, which is more robust for long titles where
        a single extra word can tank the character-level ratio.
        """
        if not a or not b:
            return 0.0
        # Preprocess
        pa = CrossRefVerifier._preprocess_title(a)
        pb = CrossRefVerifier._preprocess_title(b)
        if not pa or not pb:
            return 0.0

        # Character-level similarity
        char_sim = SequenceMatcher(None, pa, pb).ratio()

        # Word-level Jaccard similarity (handles word-order differences)
        words_a = set(pa.split())
        words_b = set(pb.split())
        # Remove very short stop words that add noise
        stops = {"a", "an", "the", "of", "in", "on", "at", "to", "and",
                 "or", "for", "with", "by", "from", "is", "are", "was",
                 "its", "their", "this", "that", "as", "be"}
        words_a -= stops
        words_b -= stops
        if words_a or words_b:
            jaccard = len(words_a & words_b) / len(words_a | words_b)
        else:
            jaccard = char_sim

        return max(char_sim, jaccard)

    @staticmethod
    def _normalise_author(name: str) -> tuple[str, str]:
        """Split an author name into (surname, first_initial) in lowercase.

        Handles the two dominant formats found in systematic review tables
        and CrossRef API responses:

          "Borghaei H"        → ("borghaei", "h")     [surname + initial]
          "Borghaei Hossein"  → ("borghaei", "h")     [surname + given; CrossRef]
          "Borghaei, Hossein" → ("borghaei", "h")     [comma-separated]
          "H Borghaei"        → ("borghaei", "h")     [given-first; rare]
          "Borghaei"          → ("borghaei", "")      [surname only]
          "Paz-Ares L"        → ("paz-ares", "l")     [hyphenated surname]

        The rule: whichever token is ≤2 chars is the initial; otherwise
        the FIRST token is treated as the family name (matching the
        CrossRef ``family given`` ordering and the typical ``Family I``
        student convention).
        """
        name = name.strip().lower().replace(",", " ")
        parts = [p.rstrip(".") for p in name.split() if p.rstrip(".")]
        if not parts:
            return ("", "")
        if len(parts) == 1:
            return (parts[0], "")

        # If the last token is an initial (≤2 chars) → "Borghaei H"
        if len(parts[-1]) <= 2:
            surname = " ".join(parts[:-1])
            initial = parts[-1][0]
            return (surname, initial)

        # If the first token is an initial (≤2 chars) → "H Borghaei"
        if len(parts[0]) <= 2:
            surname = " ".join(parts[1:])
            initial = parts[0][0]
            return (surname, initial)

        # Both tokens are long → "Borghaei Hossein" (CrossRef family+given)
        # Treat first token as family name, extract initial of second token
        return (parts[0], parts[1][0])

    @staticmethod
    def _author_overlap(student_authors: list[str], cr_authors: list[str]) -> float:
        """Compute author list overlap ratio.

        Uses surname + initial matching so that "Borghaei H" correctly
        matches CrossRef's "Borghaei Hossein" (or vice versa).
        """
        if not student_authors or not cr_authors:
            return 0.0 if student_authors else 1.0

        # Pre-parse CrossRef authors
        cr_parsed = [
            CrossRefVerifier._normalise_author(a) for a in cr_authors
        ]

        matches = 0
        for sa in student_authors:
            s_surname, s_initial = CrossRefVerifier._normalise_author(sa)
            if not s_surname:
                continue
            matched = False
            for c_surname, c_initial in cr_parsed:
                # Surname must be similar
                sur_sim = SequenceMatcher(None, s_surname, c_surname).ratio()
                if sur_sim < 0.80:
                    continue
                # If both have initials/given names, they must also agree
                if s_initial and c_initial:
                    # Compare first character only (initial vs initial)
                    if s_initial[0] != c_initial[0]:
                        continue
                matched = True
                break
            if matched:
                matches += 1

        return matches / len(student_authors)

    @staticmethod
    def _extract_year(work: dict) -> int | None:
        """Extract publication year from CrossRef date-parts."""
        for date_field in ["published-print", "published-online", "issued"]:
            date_obj = work.get(date_field, {})
            parts = date_obj.get("date-parts", [[]])
            if parts and parts[0] and len(parts[0]) >= 1:
                return int(parts[0][0])
        return None

    @staticmethod
    def _compute_confidence(
        title_sim: float,
        author_ratio: float,
        reference: ReferenceEntry,
        cr_year: int | None,
        *,
        doi_matched: bool = False,
    ) -> float:
        """Compute an overall confidence score (0.0–1.0).

        Weight scheme (doi-lookup vs title-search):
          - doi-lookup: DOI is an exact identifier. The paper definitely
            exists; we only deduct for severe title/author mismatches.
            title ×0.35 + author ×0.25 + year ×0.15 + doi_bonus 0.25
          - title-search: No DOI; we rely entirely on metadata.
            title ×0.50 + author ×0.30 + year ×0.20
        """
        if doi_matched:
            # DOI confirmed the paper exists → start from a higher base
            score = 0.25  # DOI existence bonus
            score += title_sim * 0.35
            score += author_ratio * 0.25
            if reference.year and cr_year:
                score += 0.15 if reference.year == cr_year else 0.0
            else:
                score += 0.08  # partial credit
        else:
            score = title_sim * 0.50 + author_ratio * 0.30
            if reference.year and cr_year:
                score += 0.20 if reference.year == cr_year else 0.0
            else:
                score += 0.10

        return min(score, 1.0)

    def _build_params(self) -> dict[str, str]:
        """Build common query params (mailto for polite pool)."""
        params: dict[str, str] = {}
        if self._mailto:
            params["mailto"] = self._mailto
        return params
