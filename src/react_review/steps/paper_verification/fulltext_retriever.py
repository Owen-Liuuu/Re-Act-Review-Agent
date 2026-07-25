"""Full-text paper retriever with a 4-tier retrieval chain.

Retrieval priority:
  1. PubMed Central (PMC) — free full-text XML for open-access papers
  2. Unpaywall — discovers OA PDF/HTML links for any DOI
  3. OpenAlex — broad coverage of non-English / regional journals;
     provides OA PDF links and an inverted-index abstract fallback
  4. PubMed abstract — last-resort abstract-only text
  → otherwise: metadata-only fallback document

This dramatically improves Step 3 data extraction quality because the
LLM receives full paper text (methods, results, tables) instead of just
a ~300-word abstract.

API references:
  PMC:       https://www.ncbi.nlm.nih.gov/pmc/tools/developers/
  Unpaywall: https://unpaywall.org/products/api  (email required)
  OpenAlex:  https://docs.openalex.org/           (no key; email → polite pool)
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET

import httpx
import structlog

from react_review.core.config import PubMedSettings
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.steps.paper_verification.schemas import ReferenceEntry

logger = structlog.get_logger(__name__)

# Max chars to keep from full text (avoid blowing up LLM context)
_MAX_FULLTEXT_CHARS = 60_000

# Minimum size (chars) that a section-filtered result must reach before we
# trust it. If filtering produces less than this, we fall back to returning
# the original text — better to send a bit of noise than to strip too hard
# and lose the actual data rows.
_MIN_FILTERED_CHARS = 1500

# ----------------------------------------------------------------------
# Section-title regex patterns (used by _extract_core_sections)
# ----------------------------------------------------------------------
# A "heading-like" line in plain-text PDFs: optional leading number
# ("2." or "2.1"), 1-8 words, starts with a capital letter, optional
# trailing colon, nothing else on the line. Captured as group "title".
_HEADING_LINE_RE = re.compile(
    r"^\s*"
    r"(?:\d+(?:\.\d+)*\.?\s+)?"                # optional "2." / "2.1 "
    r"(?P<title>[A-Z][A-Za-z][A-Za-z0-9\s\-&/,]{1,70})"
    r"\s*:?\s*$",
    re.MULTILINE,
)

# Titles we want to KEEP — these are where the data lives for extraction.
_KEEP_TITLE_RE = re.compile(
    r"(?i)\b(methods?|materials(?:\s+and\s+methods)?|procedures?|"
    r"experimental|patients?|participants|subjects|study\s+population|"
    r"results?|findings?|outcomes?|(?:study\s+)?characteristics|"
    r"baseline|data\s+extraction|table\s*\d*|figure\s*\d*)\b"
)

# Titles we want to DROP — low-signal for data extraction tasks.
_DROP_TITLE_RE = re.compile(
    r"(?i)\b(introduction|background|discussion|conclusions?|"
    r"references|bibliography|acknowledge?ments|funding|"
    r"author\s+contribution|conflict\s+of\s+interest|"
    r"declarations?|competing\s+interests?|"
    r"supplementary|appendix|ethics\s+statement)\b"
)


class FullTextRetriever(PaperRetriever):
    """Retrieves paper full text via PMC → Unpaywall → PubMed abstract.

    Args:
        pubmed_settings: PubMed/PMC API settings.
        unpaywall_email: Email for Unpaywall API (required by their TOS).
            If empty, Unpaywall tier is skipped.
    """

    def __init__(
        self,
        pubmed_settings: PubMedSettings,
        unpaywall_email: str = "",
    ) -> None:
        self._pubmed_base = pubmed_settings.base_url.rstrip("/")
        self._api_key = pubmed_settings.api_key
        self._unpaywall_email = unpaywall_email or pubmed_settings.email
        self._timeout = httpx.Timeout(60.0, connect=15.0)

    async def retrieve(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Try to get the fullest text available for a reference.

        Chain: PMC full text → Unpaywall OA → PubMed abstract → fallback.
        """
        paper_id = reference.doi or reference.title[:50]
        logger.info("fulltext_retrieve_start", paper_id=paper_id[:50])

        # --- Tier 1: PubMed Central full text ---
        doc = await self._try_pmc(reference)
        if doc:
            return doc

        # --- Tier 2: Unpaywall open-access PDF/HTML ---
        doc = await self._try_unpaywall(reference)
        if doc:
            return doc

        # --- Tier 3: OpenAlex (OA PDF or inverted-index abstract) ---
        doc = await self._try_openalex(reference)
        if doc:
            return doc

        # --- Tier 4: PubMed abstract (baseline) ---
        doc = await self._try_pubmed_abstract(reference)
        if doc:
            return doc

        # --- Fallback: metadata only ---
        logger.warning("fulltext_all_tiers_failed", title=reference.title[:50])
        return self._fallback_document(reference)

    # ==================================================================
    # Tier 1: PubMed Central
    # ==================================================================

    async def _try_pmc(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Try to get full text from PubMed Central.

        Steps:
          1. Find PMC ID via esearch (DOI → PMCID)
          2. Fetch full XML from PMC efetch
          3. Parse XML → plain text
        """
        try:
            pmc_id = await self._find_pmc_id(reference)
            if not pmc_id:
                return None

            full_text = await self._fetch_pmc_fulltext(pmc_id)
            if not full_text or len(full_text) < 200:
                return None

            # Keep only core sections (Methods/Results/Tables) to cut LLM
            # input size by ~60-70%. Falls back to original text if the
            # filter can't find enough core content.
            core_text = self._extract_core_sections(full_text)

            logger.info(
                "pmc_fulltext_retrieved",
                pmc_id=pmc_id,
                chars_original=len(full_text),
                chars_core=len(core_text),
                core_filtered=len(core_text) < len(full_text),
            )
            return PaperDocument(
                paper_id=reference.doi or f"pmc:{pmc_id}",
                reference=reference,
                full_text=core_text[:_MAX_FULLTEXT_CHARS],
                sections=self._split_sections(full_text),
                metadata={"source": "pmc", "pmc_id": pmc_id},
            )

        except Exception as exc:
            logger.debug("pmc_tier_failed", error=str(exc)[:120])
            return None

    async def _find_pmc_id(self, reference: ReferenceEntry) -> str | None:
        """Search for a PMC ID using the paper's DOI.

        IMPORTANT: Only searches by DOI, NOT by title. Title search in PMC
        is highly unreliable and frequently returns unrelated papers with
        similar keywords (e.g. other diabetes/EAT papers instead of the
        target paper). DOI is the only reliable identifier.
        """
        if not reference.doi:
            return None

        params: dict[str, str] = {
            "db": "pmc",
            "rettype": "json",
            "retmode": "json",
            "retmax": "3",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        params["term"] = f"{reference.doi}[doi]"
        return await self._esearch(params)

    async def _fetch_pmc_fulltext(self, pmc_id: str) -> str:
        """Fetch full text XML from PMC and convert to plain text."""
        url = f"{self._pubmed_base}/efetch.fcgi"
        params: dict[str, str] = {
            "db": "pmc",
            "id": pmc_id,
            "rettype": "xml",
            "retmode": "xml",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            xml_text = resp.text

        return self._pmc_xml_to_text(xml_text)

    @staticmethod
    def _pmc_xml_to_text(xml_text: str) -> str:
        """Parse PMC XML and extract readable text.

        Extracts: title, abstract, body sections, tables (as text).
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return ""

        parts: list[str] = []

        # Article title
        for title_el in root.iter("article-title"):
            if title_el.text:
                parts.append(f"TITLE: {_et_text(title_el)}\n")

        # Abstract
        for abstract_el in root.iter("abstract"):
            text = _et_text(abstract_el)
            if text:
                parts.append(f"ABSTRACT:\n{text}\n")

        # Body sections
        for body_el in root.iter("body"):
            for sec in body_el.iter("sec"):
                # Section title
                title_nodes = sec.findall("title")
                sec_title = _et_text(title_nodes[0]) if title_nodes else ""
                # Section paragraphs
                paragraphs = []
                for p in sec.findall("p"):
                    p_text = _et_text(p)
                    if p_text:
                        paragraphs.append(p_text)
                if paragraphs:
                    header = f"\n## {sec_title}\n" if sec_title else "\n"
                    parts.append(header + "\n".join(paragraphs) + "\n")

        # Tables (as text, for data extraction)
        for table_wrap in root.iter("table-wrap"):
            caption_el = table_wrap.find("caption")
            caption = _et_text(caption_el) if caption_el is not None else ""
            table_el = table_wrap.find(".//table")
            if table_el is not None:
                table_text = _table_to_text(table_el)
                parts.append(f"\nTABLE: {caption}\n{table_text}\n")

        return "\n".join(parts)

    # ==================================================================
    # Tier 2: Unpaywall
    # ==================================================================

    async def _try_unpaywall(self, reference: ReferenceEntry) -> PaperDocument | None:
        """Try to find and download open-access full text via Unpaywall.

        Unpaywall provides free OA locations for papers identified by DOI.
        It can return PDF or HTML links.
        """
        if not reference.doi or not self._unpaywall_email:
            return None

        try:
            oa_url, content_type = await self._unpaywall_find_oa(reference.doi)
            if not oa_url:
                return None

            if content_type == "pdf":
                full_text = await self._download_and_parse_pdf(oa_url)
            else:
                full_text = await self._download_html_text(oa_url)

            if not full_text or len(full_text) < 200:
                return None

            # Strip non-data sections (Intro/Discussion/References). Safe:
            # returns original text when the filter can't identify enough
            # core content.
            core_text = self._extract_core_sections(full_text)

            logger.info(
                "unpaywall_fulltext_retrieved",
                doi=reference.doi,
                content_type=content_type,
                chars_original=len(full_text),
                chars_core=len(core_text),
                core_filtered=len(core_text) < len(full_text),
            )
            return PaperDocument(
                paper_id=reference.doi,
                reference=reference,
                full_text=core_text[:_MAX_FULLTEXT_CHARS],
                sections=self._split_sections(full_text),
                metadata={
                    "source": "unpaywall",
                    "oa_url": oa_url,
                    "content_type": content_type,
                },
            )

        except Exception as exc:
            logger.debug("unpaywall_tier_failed", error=str(exc)[:120])
            return None

    async def _unpaywall_find_oa(self, doi: str) -> tuple[str, str]:
        """Query Unpaywall API for OA location.

        Returns (url, content_type) where content_type is 'pdf' or 'html'.
        Returns ('', '') if no OA version found.
        """
        url = f"https://api.unpaywall.org/v2/{doi}"
        params = {"email": self._unpaywall_email}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                logger.debug("unpaywall_not_found", doi=doi)
                return "", ""
            resp.raise_for_status()
            data = resp.json()

        # Check best_oa_location first, then oa_locations list
        best = data.get("best_oa_location") or {}
        if not best and data.get("oa_locations"):
            best = data["oa_locations"][0]

        if not best:
            return "", ""

        # Prefer PDF
        pdf_url = best.get("url_for_pdf", "")
        if pdf_url:
            return pdf_url, "pdf"

        # Fallback to landing page / HTML
        landing_url = best.get("url_for_landing_page", "") or best.get("url", "")
        if landing_url:
            return landing_url, "html"

        return "", ""

    async def _download_and_parse_pdf(self, url: str) -> str:
        """Download a PDF from a URL and extract text using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("pymupdf_not_installed", msg="pip install PyMuPDF")
            return ""

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=30.0),
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            pdf_bytes = resp.content

        # Parse PDF in memory
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts: list[str] = []
        for page in doc:
            text = page.get_text("text")
            if text:
                parts.append(text)
        doc.close()

        return "\n".join(parts)

    async def _download_html_text(self, url: str) -> str:
        """Download an HTML page and extract the main text content.

        Uses a simple approach: strip tags, keep text.
        """
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "react-review/1.0 (academic research tool; "
                    f"mailto:{self._unpaywall_email})"
                )
            },
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        # Basic HTML → text (strip tags)
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        # Decode HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
        return text.strip()

    # ==================================================================
    # Tier 3: OpenAlex
    # ==================================================================

    async def _try_openalex(
        self, reference: ReferenceEntry
    ) -> PaperDocument | None:
        """Tier 3 retriever — OpenAlex.

        Why this tier exists: Unpaywall often misses non-English /
        regional-journal papers (e.g. Turkish, Indian, Chinese OA titles
        that never registered with Unpaywall's sources). OpenAlex has
        much broader coverage (~250M works) and provides two useful
        data channels:

          (a) OA PDF URLs via ``open_access.oa_url`` and
              ``locations[*].pdf_url`` — when present, we download the
              PDF and run the same section filter as other tiers.
          (b) An ``abstract_inverted_index`` — a {word: [positions]}
              dict that we reconstruct into a flat abstract string.
              Better than a metadata-only fallback: the LLM can at least
              extract country / modality / broad outcome from the
              abstract.

        Requires a DOI (title-based OpenAlex queries are no more
        reliable than PMC's). Silent failure on any error — we fall
        through to Tier 4 (PubMed abstract).
        """
        if not reference.doi:
            return None

        try:
            work = await self._openalex_fetch_work(reference.doi)
            if not work:
                return None

            # --- 3a. OA PDF — best case ---
            pdf_url = self._openalex_best_pdf_url(work)
            if pdf_url:
                full_text = await self._download_and_parse_pdf(pdf_url)
                if full_text and len(full_text) >= 200:
                    core_text = self._extract_core_sections(full_text)
                    logger.info(
                        "openalex_fulltext_retrieved",
                        doi=reference.doi,
                        pdf_url=pdf_url[:80],
                        chars_original=len(full_text),
                        chars_core=len(core_text),
                    )
                    return PaperDocument(
                        paper_id=reference.doi,
                        reference=reference,
                        full_text=core_text[:_MAX_FULLTEXT_CHARS],
                        sections=self._split_sections(full_text),
                        metadata={
                            "source": "openalex_pdf",
                            "pdf_url": pdf_url,
                            "openalex_id": work.get("id", ""),
                        },
                    )

            # --- 3b. Inverted-index abstract — graceful fallback ---
            abstract = self._openalex_abstract_from_inverted_index(work)
            if abstract and len(abstract) >= 100:
                title = work.get("title") or reference.title
                text = f"TITLE: {title}\n\nABSTRACT:\n{abstract}"
                logger.info(
                    "openalex_abstract_retrieved",
                    doi=reference.doi,
                    chars=len(abstract),
                )
                return PaperDocument(
                    paper_id=reference.doi,
                    reference=reference,
                    full_text=text,
                    sections={"abstract": abstract},
                    metadata={
                        "source": "openalex_abstract",
                        "openalex_id": work.get("id", ""),
                    },
                )

            return None

        except Exception as exc:
            logger.debug("openalex_tier_failed", error=str(exc)[:120])
            return None

    async def _openalex_fetch_work(self, doi: str) -> dict | None:
        """Fetch an OpenAlex work record by DOI.

        Accepts DOI in any common form (bare, with URL prefix, with
        'doi:' prefix). Returns the parsed JSON dict, or None on 404.
        """
        d = doi.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if d.lower().startswith(prefix):
                d = d[len(prefix):]
                break

        url = f"https://api.openalex.org/works/doi:{d}"
        params: dict[str, str] = {}
        # Include the email in the polite-pool param for faster / more
        # reliable OpenAlex responses.
        if self._unpaywall_email:
            params["mailto"] = self._unpaywall_email

        async with httpx.AsyncClient(
            timeout=self._timeout,
            headers={"User-Agent": "react-review/1.0"},
        ) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                logger.debug("openalex_not_found", doi=doi)
                return None
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _openalex_best_pdf_url(work: dict) -> str:
        """Pick the best OA PDF URL from an OpenAlex work record.

        Priority: ``open_access.oa_url`` (if is_oa) → ``primary_location.
        pdf_url`` → any location with ``is_oa=true`` and ``pdf_url`` set.
        Returns '' if no PDF link is available.
        """
        oa = work.get("open_access") or {}
        if oa.get("is_oa") and oa.get("oa_url"):
            return oa["oa_url"]

        primary = work.get("primary_location") or {}
        if primary.get("pdf_url"):
            return primary["pdf_url"]

        for loc in work.get("locations") or []:
            if loc.get("is_oa") and loc.get("pdf_url"):
                return loc["pdf_url"]

        return ""

    @staticmethod
    def _openalex_abstract_from_inverted_index(work: dict) -> str:
        """Reconstruct a paper abstract from OpenAlex's inverted index.

        OpenAlex stores abstracts as ``{word: [positions, ...]}``. We
        reverse this into the original word-ordered text. Returns '' if
        the field is missing or empty.
        """
        idx = work.get("abstract_inverted_index") or {}
        if not idx:
            return ""

        pairs: list[tuple[int, str]] = []
        for word, positions in idx.items():
            if not isinstance(positions, list):
                continue
            for pos in positions:
                if isinstance(pos, int):
                    pairs.append((pos, word))
        if not pairs:
            return ""

        pairs.sort(key=lambda x: x[0])
        return " ".join(word for _, word in pairs)

    # ==================================================================
    # Tier 4: PubMed abstract (same as original retriever)
    # ==================================================================

    async def _try_pubmed_abstract(
        self, reference: ReferenceEntry
    ) -> PaperDocument | None:
        """Fallback: fetch just the abstract from PubMed."""
        try:
            pmid = await self._find_pubmed_pmid(reference)
            if not pmid:
                return None

            abstract = await self._fetch_abstract(pmid)
            if not abstract:
                return None

            logger.info(
                "pubmed_abstract_retrieved",
                pmid=pmid,
                chars=len(abstract),
            )
            return PaperDocument(
                paper_id=reference.doi or f"pmid:{pmid}",
                reference=reference,
                full_text=abstract,
                sections={"abstract": abstract},
                metadata={"source": "pubmed_abstract", "pmid": pmid},
            )

        except Exception as exc:
            logger.debug("pubmed_abstract_failed", error=str(exc)[:120])
            return None

    async def _find_pubmed_pmid(self, reference: ReferenceEntry) -> str | None:
        """Search PubMed to get PMID."""
        params: dict[str, str] = {
            "db": "pubmed",
            "rettype": "json",
            "retmode": "json",
            "retmax": "3",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        if reference.doi:
            params["term"] = f"{reference.doi}[doi]"
            pmid = await self._esearch(params)
            if pmid:
                return pmid

        if reference.title:
            params["term"] = f"{reference.title}[title]"
            pmid = await self._esearch(params)
            if pmid:
                return pmid

        return None

    async def _fetch_abstract(self, pmid: str) -> str:
        """Fetch abstract text from PubMed efetch."""
        url = f"{self._pubmed_base}/efetch.fcgi"
        params: dict[str, str] = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "abstract",
            "retmode": "text",
        }
        if self._api_key:
            params["api_key"] = self._api_key

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.text.strip()

    # ==================================================================
    # Shared helpers
    # ==================================================================

    async def _esearch(self, params: dict[str, str]) -> str | None:
        """Execute an NCBI esearch and return the first ID, or None."""
        url = f"{self._pubmed_base}/esearch.fcgi"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        id_list = data.get("esearchresult", {}).get("idlist", [])
        return id_list[0] if id_list else None

    # ------------------------------------------------------------------
    # Core-section extraction — reduces LLM input by ~60-70% on typical
    # papers by keeping only Methods/Results/Tables. Called before the
    # _MAX_FULLTEXT_CHARS truncation so the kept text is maximally useful.
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_core_sections(text: str) -> str:
        """Keep only sections that carry data (Methods, Results, Tables).

        Strategy (in order):
          A. If the text contains ``## Section`` markers (emitted by our
             PMC XML parser), split by those and keep sections whose
             heading matches ``_KEEP_TITLE_RE``. Also always preserve the
             TITLE/ABSTRACT preamble and any ``TABLE:`` blocks at the end.
          B. Otherwise (plain-text from PDF / HTML), scan for heading-like
             lines with ``_HEADING_LINE_RE`` and keep the segments whose
             heading matches the keep-list.
          C. If neither strategy produces at least ``_MIN_FILTERED_CHARS``
             of content, return the input unchanged. Being conservative
             here matters — the LLM can still work with noisier text, but
             it cannot recover data we stripped out by mistake.
        """
        if not text or len(text) < 3000:
            # Short texts (abstracts, short notes) — already focused.
            return text

        # --- Strategy A: PMC ## marker style ---
        if "\n## " in text or text.startswith("## "):
            filtered = FullTextRetriever._filter_marker_style(text)
            if filtered and len(filtered) >= _MIN_FILTERED_CHARS:
                return filtered

        # --- Strategy B: heading-line regex on plain text ---
        filtered = FullTextRetriever._filter_heading_regex(text)
        if filtered and len(filtered) >= _MIN_FILTERED_CHARS:
            return filtered

        # --- Strategy C: fallback ---
        return text

    @staticmethod
    def _filter_marker_style(text: str) -> str:
        """Keep only ``## Section`` blocks whose title matches the keep-list.

        Also preserves:
          - Everything before the first ``## `` (TITLE + ABSTRACT preamble)
          - Every ``TABLE: ...`` block (emitted at top level by _pmc_xml_to_text)
        """
        # Identify span of each ## section and each TABLE: block.
        lines = text.split("\n")
        kept_parts: list[str] = []

        # 1. Preamble: everything before the first "## " line.
        preamble: list[str] = []
        body_start = 0
        for i, line in enumerate(lines):
            if line.startswith("## "):
                body_start = i
                break
            preamble.append(line)
        else:
            # No ## markers at all (shouldn't happen given caller's check,
            # but handle gracefully).
            body_start = len(lines)

        if preamble:
            kept_parts.append("\n".join(preamble).rstrip() + "\n")

        # 2. Walk sections: each "## X" line starts a block that ends at
        #    the next "## " line (or end of text). Decide keep/drop by title.
        cur_title: str | None = None
        cur_block: list[str] = []

        def _flush() -> None:
            if cur_title is None or not cur_block:
                return
            # Keep if title matches keep-list AND does not match drop-list.
            if _KEEP_TITLE_RE.search(cur_title) and not _DROP_TITLE_RE.search(
                cur_title
            ):
                kept_parts.append(f"## {cur_title}\n" + "\n".join(cur_block).rstrip() + "\n")

        for line in lines[body_start:]:
            if line.startswith("## "):
                _flush()
                cur_title = line[3:].strip()
                cur_block = []
            else:
                cur_block.append(line)
        _flush()

        # 3. Make sure any TABLE: blocks that got swallowed by the last
        #    section (or appear standalone) are preserved — tables are
        #    where the extractable data often is.
        for m in re.finditer(
            r"^TABLE:.*?(?=^##\s|^TABLE:|\Z)",
            text,
            flags=re.MULTILINE | re.DOTALL,
        ):
            block = m.group(0).rstrip() + "\n"
            if block not in "\n".join(kept_parts):
                kept_parts.append("\n" + block)

        return "\n".join(kept_parts).strip()

    @staticmethod
    def _filter_heading_regex(text: str) -> str:
        """Scan plain text for heading-like lines; keep only wanted sections.

        For each detected heading, the section body runs until the next
        heading (or end of text). A section is kept iff its title matches
        ``_KEEP_TITLE_RE`` and not ``_DROP_TITLE_RE``.
        """
        # Collect candidate heading positions and their normalised titles.
        headings: list[tuple[int, int, str]] = []
        for m in _HEADING_LINE_RE.finditer(text):
            title = m.group("title").strip()
            # Skip very long "titles" — almost certainly a full sentence
            # that happened to start with a capital letter.
            if len(title.split()) > 8:
                continue
            # Skip very short noise like single letters.
            if len(title) < 4:
                continue
            headings.append((m.start(), m.end(), title))

        # Need at least 3 distinct headings to trust the segmentation.
        # (Most papers have >5 — Introduction/Methods/Results/Discussion/
        #  Conclusion/References.)
        if len(headings) < 3:
            return ""

        kept_parts: list[str] = []
        for i, (start, _end, title) in enumerate(headings):
            next_start = headings[i + 1][0] if i + 1 < len(headings) else len(text)
            segment = text[start:next_start]

            # Decide keep/drop.
            if _DROP_TITLE_RE.search(title):
                continue
            if _KEEP_TITLE_RE.search(title):
                kept_parts.append(segment.rstrip())

        return "\n\n".join(kept_parts).strip()

    @staticmethod
    def _split_sections(text: str) -> dict[str, str]:
        """Split full text into named sections by ## headers."""
        sections: dict[str, str] = {}
        current_name = "introduction"
        current_lines: list[str] = []

        for line in text.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections[current_name] = "\n".join(current_lines).strip()
                current_name = line[3:].strip().lower().replace(" ", "_")
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections[current_name] = "\n".join(current_lines).strip()

        return sections

    @staticmethod
    def _fallback_document(reference: ReferenceEntry) -> PaperDocument:
        """Create a metadata-only document when all retrieval tiers fail."""
        text = f"Title: {reference.title}\n"
        if reference.authors:
            text += f"Authors: {', '.join(reference.authors)}\n"
        if reference.journal:
            text += f"Journal: {reference.journal}\n"
        if reference.year:
            text += f"Year: {reference.year}\n"
        text += "\n[Full text not available. Only title and metadata provided.]"

        return PaperDocument(
            paper_id=reference.doi or f"fallback-{reference.title[:20]}",
            reference=reference,
            full_text=text,
            metadata={"source": "fallback-metadata-only"},
        )


# ======================================================================
# XML helper utilities
# ======================================================================


def _et_text(element: ET.Element | None) -> str:
    """Recursively extract all text from an ElementTree element."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _table_to_text(table_el: ET.Element) -> str:
    """Convert an XML <table> element to tab-separated text.

    Handles <thead>/<tbody>/<tr>/<th>/<td> structure.
    """
    rows: list[str] = []
    for tr in table_el.iter("tr"):
        cells: list[str] = []
        for cell in tr:
            if cell.tag in ("th", "td"):
                cells.append(_et_text(cell))
        if cells:
            rows.append("\t".join(cells))
    return "\n".join(rows)
