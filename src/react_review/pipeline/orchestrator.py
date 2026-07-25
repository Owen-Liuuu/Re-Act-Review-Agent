"""Pipeline orchestrator: runs the 4-step pipeline sequentially.

Flow after the Step 0 / Step 1 split:

  Step 0 (Ingestion, in pdf_parsing/parser.py):
          LLM understands the student submission — produces
          search_strategy (per-DB queries), selected_papers,
          submitted_tables, and the rich evidence_schema.

  Step 1 (this module):
          PURE-API multi-database identification count check.
          For each database the student claims to have searched, run
          the corresponding free API (PubMed / Europe PMC / OpenAlex)
          and compare counts. Subscription-only databases (Embase,
          CINAHL, Cochrane, Web of Science) are marked UNVERIFIED.
          Per project decision #5, Step 1 is informational only and
          does NOT contribute to the FAIL verdict.

  Step 2: Verify each selected paper exists via CrossRef.
  Step 3: Retrieve + extract data from selected papers using LLM,
          using the evidence_schema from Step 0.
  Step 4: Compare student's tables vs AI-generated tables → report.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from difflib import SequenceMatcher

import structlog

# Max number of papers processed concurrently in Step 2 (CrossRef verify)
# and Step 3 (full-text retrieval + LLM extraction).
#
# Lowered from 5 → 3 on 2026-05-10: with 5 papers in flight, Step 3 fired
# 5 near-simultaneous calls at each LLM backend, overwhelming stricter
# free-tier providers (GLM/Zhipu) with a 429 burst that even the retry
# loop could not fully absorb — the first batch of papers came back with
# MODEL N/A. 3 keeps a healthy speedup while shrinking the burst. The
# per-backend semaphore (LLMSettings.max_concurrency) still caps actual
# in-flight LLM requests independently; this just reduces how many papers
# pile up at once.
_PAPER_CONCURRENCY = 3  # concurrency cap

# Tolerance for identification count verdicts: AI count within 25% of
# student count is treated as MATCH, beyond that as WARN. Per decision
# #5 these never contribute to FAIL.
_COUNT_DELTA_MATCH_PCT = 25.0

from react_review.core.enums import (
    PipelineStep,
    ValidationSeverity,
    VerificationStatus,
)
from react_review.llm.base import LLMBackend
from react_review.steps.search_validation.interfaces import SearchProvider
from react_review.steps.search_validation.multi_db_count import (
    IdentificationCounter,
    UNVERIFIABLE_DATABASES,
    VERIFIABLE_DATABASES,
)
from react_review.steps.search_validation.schemas import (
    SearchStrategy as LegacySearchStrategy,
)
from react_review.steps.paper_verification.interfaces import (
    PaperRetriever,
    ReferenceVerifier,
)
from react_review.steps.data_extraction.interfaces import Extractor
from react_review.steps.table_comparison.interfaces import (
    ReportGenerator,
    TableComparator,
)
from react_review.pipeline.schemas import (
    DatabaseCountCheck,
    MultiDBIdentificationCheck,
    PipelineRunResult,
    StudentReviewInput,
    ValidationFlag,
)

logger = structlog.get_logger(__name__)


# Step 1's old LLM-driven review analysis has moved to Step 0 (Ingestion).
# The orchestrator no longer holds an analysis prompt — search_strategy and
# evidence_schema arrive pre-built on ``StudentReviewInput``.


class PipelineOrchestrator:
    """Coordinates the 4-step literature integrity checking pipeline.

    Args:
        search_provider: Step 1 single-DB reproducibility provider
            (PubMed). Used for the legacy ``papers_in_search`` check.
        counters: Identification-count providers used by Step 1's
            multi-database check. Keyed implicitly by ``counter.name``;
            typically one each for PubMed / Europe PMC / OpenAlex.
            Defaults to an empty list, which causes the multi-DB check
            to mark every database as UNVERIFIED.
        reference_verifier: Step 2 verifier implementation.
        paper_retriever: Step 2/3 retriever implementation.
        extractors: Step 3 extractor implementations (1 or more).
        table_comparator: Step 4 comparator implementation.
        report_generator: Step 4 report generator implementation.
        llm_backend: Optional LLM backend (kept for future internal use;
            no longer used by Step 1 — that work moved to Step 0).
        enabled_steps: Which steps to run. Defaults to all.
    """

    def __init__(
        self,
        search_provider: SearchProvider,
        reference_verifier: ReferenceVerifier,
        paper_retriever: PaperRetriever,
        extractors: list[Extractor],
        table_comparator: TableComparator,
        report_generator: ReportGenerator,
        counters: list[IdentificationCounter] | None = None,
        llm_backend: LLMBackend | None = None,
        enabled_steps: list[str] | None = None,
    ) -> None:
        self._search_provider = search_provider
        self._reference_verifier = reference_verifier
        self._paper_retriever = paper_retriever
        self._extractors = extractors
        self._table_comparator = table_comparator
        self._report_generator = report_generator
        self._llm = llm_backend
        self._enabled_steps = set(enabled_steps or [s.value for s in PipelineStep])
        # Index counters by their declared ``name`` so we can look them
        # up by database name when scanning the student's strategy.
        self._counters: dict[str, IdentificationCounter] = {
            c.name: c for c in (counters or []) if c.name
        }

    async def run(self, student_input: StudentReviewInput) -> PipelineRunResult:
        """Execute the pipeline on a student's submission."""
        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "pipeline_start",
            run_id=run_id,
            student_id=student_input.student_id,
            n_selected=len(student_input.selected_papers),
        )

        result = PipelineRunResult(run_id=run_id, student_input=student_input)

        # Step 1: Multi-database identification count check (informational).
        if PipelineStep.SEARCH_VALIDATION.value in self._enabled_steps:
            result = await self._run_search_validation(result, student_input)

        # Step 2: Paper Verification (CrossRef existence + metadata).
        if PipelineStep.PAPER_VERIFICATION.value in self._enabled_steps:
            result = await self._run_paper_verification(result, student_input)

        # Step 3: Data Extraction (driven by Step 0's evidence_schema).
        if PipelineStep.DATA_EXTRACTION.value in self._enabled_steps:
            result = await self._run_data_extraction(result, student_input)

        # Step 4: Table Comparison
        if PipelineStep.TABLE_COMPARISON.value in self._enabled_steps:
            result = await self._run_table_comparison(result)

        result.completed_at = datetime.now()
        logger.info("pipeline_complete", run_id=run_id, flags=len(result.all_flags))
        return result

    # ==================================================================
    # Step 1: Multi-database identification count check (informational)
    # ==================================================================
    #
    # Per project decision #5, Step 1 is informational only — it never
    # contributes to a FAIL verdict. It runs the queries Step 0 already
    # extracted from the review, against whichever free APIs we have
    # (PubMed / Europe PMC / OpenAlex), and produces a PRISMA-style
    # "records identified from databases" comparison row per database.
    #
    # No LLM is called in Step 1; that work moved to Step 0.

    async def _run_search_validation(
        self, result: PipelineRunResult, student_input: StudentReviewInput
    ) -> PipelineRunResult:
        """Step 1: Multi-DB count check + legacy single-PubMed reproducibility.

        Sub-steps:
          1a. Multi-database identification count check using the per-DB
              queries Step 0 produced. Subscription-only databases are
              marked UNVERIFIED. Cross-coverage references via Europe PMC
              and OpenAlex are added when the student didn't declare them.
          1b. Single-PubMed reproducibility check (legacy ``SearchProvider``)
              that produces the sample-result list used by ``papers_in_search``.
        """
        logger.info("step_start", step="search_validation")
        try:
            # Step 1a — multi-DB identification count check.
            result.multi_db_identification_check = (
                await self._run_multi_db_count_check(student_input)
            )
            for row in result.multi_db_identification_check.per_database:
                if row.verdict == "WARN":
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.SEARCH_VALIDATION,
                            severity=ValidationSeverity.WARNING,
                            code="DB_COUNT_MISMATCH",
                            message=(
                                f"{row.database}: student reported "
                                f"{row.student_reported}, reproduced count "
                                f"{row.ai_reproduced} (Δ {row.delta_pct:.0f}%)."
                            ),
                        )
                    )

            # Step 1b — legacy single-PubMed reproducibility check.
            # Pull the PubMed query out of the structured strategy. If none
            # was extracted (e.g. YAML input declared only Embase), skip the
            # legacy check rather than fabricating a query.
            pubmed_query = next(
                (
                    d.query
                    for d in student_input.search_strategy.extracted_per_database
                    if d.database == "PubMed" and d.query.strip()
                ),
                "",
            )
            if pubmed_query:
                legacy_strategy = LegacySearchStrategy(
                    database="PubMed",
                    raw_strategy_text=pubmed_query,
                    reported_result_count=(
                        student_input.search_strategy.reported_per_db_count.get("PubMed")
                    ),
                )
                result.search_result = await self._search_provider.validate_strategy(
                    legacy_strategy
                )
                for flag in result.search_result.flags:
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.SEARCH_VALIDATION,
                            severity=flag.severity,
                            code=flag.code,
                            message=flag.message,
                        )
                    )
                # Step 1c — cross-check selected papers in search results.
                result.papers_in_search = self._check_papers_in_search(
                    student_input, result
                )
            else:
                logger.info(
                    "step1_legacy_skip",
                    reason="no PubMed query in student's structured strategy",
                )

        except Exception as exc:
            logger.error("step_failed", step="search_validation", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.SEARCH_VALIDATION,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Search validation failed: {exc}",
                )
            )
        return result

    async def _run_multi_db_count_check(
        self, student_input: StudentReviewInput
    ) -> MultiDBIdentificationCheck:
        """Run the PRISMA-style identification count comparison.

        For each database the student declared in
        ``search_strategy.extracted_per_database``:

          * If we have a free counter for it → run the query, classify
            the result against the student's reported count.
          * If not (Embase / CINAHL / Cochrane / WoS) → mark UNVERIFIED.

        Cross-coverage references via Europe PMC + OpenAlex are added
        only when the student did not already declare those databases —
        these are run against the PubMed query as a coarse coverage
        sanity check.
        """
        strategy = student_input.search_strategy
        rows: list[DatabaseCountCheck] = []
        seen_databases: set[str] = set()

        async def _check_one(db_query) -> DatabaseCountCheck:
            db_name = db_query.database
            student_count = strategy.reported_per_db_count.get(db_name)
            counter = self._counters.get(db_name)

            # Subscription-only databases — no free API.
            if counter is None:
                return DatabaseCountCheck(
                    database=db_name,
                    student_reported=student_count,
                    ai_reproduced=None,
                    verdict="UNVERIFIED",
                    note=(
                        f"No free API for {db_name}; student count is shown "
                        "but not independently verified."
                    ),
                )

            try:
                ai_count = await counter.count_results(db_query.query)
            except Exception as exc:
                logger.warning(
                    "counter_failed", database=db_name, error=str(exc)
                )
                ai_count = None

            verdict = "WARN"
            delta_pct: float | None = None
            note = ""

            if ai_count is None:
                note = f"{db_name} API call failed; could not reproduce count."
            elif student_count is None:
                verdict = "REFERENCE"
                note = f"Student did not report a {db_name} count."
            elif student_count > 0:
                delta_pct = abs(ai_count - student_count) / student_count * 100.0
                verdict = (
                    "MATCH" if delta_pct <= _COUNT_DELTA_MATCH_PCT else "WARN"
                )
            elif student_count == 0 and (ai_count or 0) > 0:
                delta_pct = 100.0
                verdict = "WARN"
                note = "Student reported zero, but reproduced query returns results."
            else:
                # Student count present but not positive — just pass through.
                verdict = "MATCH" if ai_count == student_count else "WARN"

            return DatabaseCountCheck(
                database=db_name,
                student_reported=student_count,
                ai_reproduced=ai_count,
                delta_pct=delta_pct,
                verdict=verdict,
                note=note,
            )

        # Run per-DB checks in parallel.
        if strategy.extracted_per_database:
            rows = list(
                await asyncio.gather(
                    *(_check_one(q) for q in strategy.extracted_per_database)
                )
            )
            seen_databases = {q.database for q in strategy.extracted_per_database}

        # Cross-coverage references using the PubMed query syntax.
        pubmed_query = next(
            (
                d.query
                for d in strategy.extracted_per_database
                if d.database == "PubMed" and d.query.strip()
            ),
            "",
        )
        if pubmed_query:
            for ref_name in ("Europe PMC", "OpenAlex"):
                if ref_name in seen_databases:
                    continue
                counter = self._counters.get(ref_name)
                if counter is None:
                    continue
                try:
                    ref_count = await counter.count_results(pubmed_query)
                except Exception as exc:
                    logger.warning(
                        "counter_failed", database=ref_name, error=str(exc)
                    )
                    ref_count = None
                rows.append(
                    DatabaseCountCheck(
                        database=ref_name,
                        student_reported=None,
                        ai_reproduced=ref_count,
                        verdict="REFERENCE",
                        note=(
                            "Cross-coverage reference using the PubMed query "
                            "syntax. Not directly comparable to subscription "
                            "databases."
                        ),
                    )
                )

        # Sum reproducible counts when available; otherwise leave None.
        reproducible_counts = [
            r.ai_reproduced for r in rows
            if r.ai_reproduced is not None and r.verdict in ("MATCH", "WARN")
        ]
        ai_total = sum(reproducible_counts) if reproducible_counts else None

        return MultiDBIdentificationCheck(
            per_database=rows,
            student_reported_total=strategy.reported_total_count,
            ai_total_unique=ai_total,
            note=(
                "Identification counts are informational only and do not "
                "contribute to the FAIL verdict (project decision #5). "
                f"Verifiable databases: {', '.join(VERIFIABLE_DATABASES)}; "
                f"unverifiable (no free API): {', '.join(UNVERIFIABLE_DATABASES)}."
            ),
        )

    def _check_papers_in_search(
        self, student_input: StudentReviewInput, result: PipelineRunResult
    ) -> dict[str, bool]:
        """Check if each selected paper appears in the PubMed search results.

        Matches by DOI (exact) or title similarity (fuzzy).
        Returns a dict of paper_title -> found_in_search.
        """
        papers_found: dict[str, bool] = {}

        if not result.search_result or not result.search_result.sample_results:
            return papers_found

        # Collect DOIs and titles from search results
        search_dois = {
            s.doi.lower() for s in result.search_result.sample_results if s.doi
        }
        search_titles = [
            s.title.lower() for s in result.search_result.sample_results
        ]

        for paper in student_input.selected_papers:
            found = False
            reason = ""
            best_sim: float | None = None

            # Check by DOI
            if paper.doi and paper.doi.lower() in search_dois:
                found = True
                reason = "DOI exact match"

            # Check by title similarity (also track the best score for the
            # not-found case so the report can show "closest miss = 0.62")
            if not found:
                for st in search_titles:
                    sim = SequenceMatcher(None, paper.title.lower(), st).ratio()
                    if best_sim is None or sim > best_sim:
                        best_sim = sim
                    if sim > 0.85:
                        found = True
                        reason = f"title fuzzy match (similarity={sim:.2f})"
                        break

            if not found and not reason:
                if paper.doi:
                    if best_sim is not None:
                        reason = (
                            f"DOI not in top-20 PubMed results AND "
                            f"best title similarity={best_sim:.2f} (< 0.85 threshold)"
                        )
                    else:
                        reason = "DOI not in top-20 PubMed results (no titles to compare)"
                else:
                    if best_sim is not None:
                        reason = (
                            f"no DOI provided; best title similarity={best_sim:.2f} "
                            f"(< 0.85 threshold)"
                        )
                    else:
                        reason = "no DOI provided and no PubMed sample titles available"

            papers_found[paper.title] = found
            result.papers_in_search_detail[paper.title] = {
                "found": found,
                "reason": reason,
                "best_similarity": best_sim,
                "doi": paper.doi or "",
            }

            if not found:
                result.all_flags.append(
                    ValidationFlag(
                        step=PipelineStep.SEARCH_VALIDATION,
                        severity=ValidationSeverity.WARNING,
                        code="PAPER_NOT_IN_SEARCH",
                        message=(
                            f"Selected paper not found in reproduced search results: "
                            f"'{paper.title[:60]}...'"
                        ),
                        details=(
                            "This paper was not in the first 20 PubMed results. "
                            "It may appear later in the full result set, or "
                            "the search strategy may not capture it."
                        ),
                    )
                )
            else:
                logger.info(
                    "paper_in_search",
                    title=paper.title[:50],
                    reason=reason,
                )

        return papers_found

    # ==================================================================
    # Step 2: Paper Verification
    # ==================================================================

    async def _run_paper_verification(
        self, result: PipelineRunResult, student_input: StudentReviewInput
    ) -> PipelineRunResult:
        """Step 2: Verify each selected paper exists via CrossRef.

        Papers are verified concurrently (up to ``_PAPER_CONCURRENCY`` at a
        time) via ``asyncio.gather`` — CrossRef lookups are I/O-bound, so
        this typically gives a 3–5x speedup. Results are appended in the
        original input order (not completion order) so reports remain stable.
        """
        logger.info(
            "step_start",
            step="paper_verification",
            n_papers=len(student_input.selected_papers),
            concurrency=_PAPER_CONCURRENCY,
        )
        try:
            sem = asyncio.Semaphore(_PAPER_CONCURRENCY)

            async def _verify_one(ref):
                async with sem:
                    return await self._reference_verifier.verify(ref)

            # return_exceptions=True so one failing lookup doesn't abort the
            # entire step — we convert per-paper failures into flags below.
            verifications = await asyncio.gather(
                *(_verify_one(ref) for ref in student_input.selected_papers),
                return_exceptions=True,
            )

            for ref, verification in zip(
                student_input.selected_papers, verifications
            ):
                if isinstance(verification, Exception):
                    logger.error(
                        "paper_verify_failed",
                        title=ref.title[:50],
                        error=str(verification),
                    )
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.PAPER_VERIFICATION,
                            severity=ValidationSeverity.ERROR,
                            code="VERIFY_FAILED",
                            message=(
                                f"Verification raised an exception for "
                                f"'{ref.title[:60]}...': {verification}"
                            ),
                        )
                    )
                    continue

                result.verification_results.append(verification)
                # Populate the per-paper status dict that Step 3 uses as a
                # gate. Keyed by DOI when available (canonical) or by a
                # title slug otherwise, so Step 3 can look up the same key
                # from the bare ReferenceEntry.
                result.paper_verification_statuses[
                    self._paper_key(ref)
                ] = verification.status
                for flag in verification.flags:
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.PAPER_VERIFICATION,
                            severity=flag.severity,
                            code=flag.code,
                            message=flag.message,
                        )
                    )

                # -------------------------------------------------------
                # DOI backfill: if the student supplied no DOI but the
                # CrossRef title-search produced a VERIFIED match, write
                # the canonical DOI back into the reference so Step 3's
                # PMC / Unpaywall / OpenAlex lookups can actually run.
                # Guarded by VERIFIED (confidence ≥ 0.80 for title
                # matches) to avoid poisoning downstream retrieval with
                # a wrong DOI from an UNCERTAIN match.
                # -------------------------------------------------------
                if (
                    not ref.doi
                    and verification.status == VerificationStatus.VERIFIED
                ):
                    cr_doi = (verification.matched_metadata or {}).get(
                        "doi", ""
                    ).strip()
                    if cr_doi:
                        ref.doi = cr_doi
                        logger.info(
                            "doi_backfilled_from_crossref",
                            title=ref.title[:60],
                            doi=cr_doi,
                            source=(verification.matched_metadata or {}).get(
                                "source", "?"
                            ),
                            confidence=f"{verification.confidence:.2f}",
                        )
                        result.all_flags.append(
                            ValidationFlag(
                                step=PipelineStep.PAPER_VERIFICATION,
                                severity=ValidationSeverity.INFO,
                                code="DOI_BACKFILLED",
                                message=(
                                    f"DOI backfilled from CrossRef "
                                    f"('{cr_doi}') for '{ref.title[:60]}' — "
                                    "will be used for full-text retrieval."
                                ),
                            )
                        )
        except Exception as exc:
            logger.error("step_failed", step="paper_verification", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.PAPER_VERIFICATION,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Paper verification failed: {exc}",
                )
            )
        return result

    # ==================================================================
    # Step 3: Data Extraction (driven by Step 0's evidence_schema)
    # ==================================================================

    async def _run_data_extraction(
        self, result: PipelineRunResult, student_input: StudentReviewInput
    ) -> PipelineRunResult:
        """Step 3: Retrieve full text and extract data from selected papers.

        Field selection has been simplified: Step 0 (Ingestion) now produces
        a rich ``evidence_schema``, and Step 3 simply asks each extractor for
        the same set of student-named fields. As a fallback (e.g. for very
        old YAMLs that pre-date the schema migration), we also pull column
        headers from any submitted tables so the comparator can still align.

        Each entry of ``evidence_schema`` carries the student's verbatim
        field name (used as the join key in Step 4), the canonical concept
        (used by the extractor's prompt), the value type, and per-field
        tolerances. When ``evidence_schema`` is empty we synthesise minimal
        entries from any ``submitted_tables`` column headers so the
        pipeline still runs end-to-end on legacy inputs — the comparator
        will fall back to its default thresholds.
        """
        from react_review.pipeline.schemas import EvidenceFieldSchema

        # ----- Build the schema list passed to extractors -----
        seen: dict[str, EvidenceFieldSchema] = {}
        sources_used: list[str] = []

        def _add_schema(entries: list[EvidenceFieldSchema], source: str) -> None:
            added = False
            for entry in entries:
                key = entry.student_field_name.strip().lower()
                if key and key not in seen:
                    seen[key] = entry
                    added = True
            if added and source not in sources_used:
                sources_used.append(source)

        # Primary source: evidence_schema from Step 0.
        _add_schema(list(student_input.evidence_schema), "evidence_schema")

        # Fallback: synthesise minimal schema from submitted-table column
        # headers so very old YAML inputs (or Step 0 failures) still run.
        synth: list[EvidenceFieldSchema] = []
        for st in student_input.submitted_tables:
            for f in st.fields:
                if not f.field_name:
                    continue
                synth.append(
                    EvidenceFieldSchema(
                        student_field_name=f.field_name,
                        canonical_concept=f.field_name.lower().replace(" ", "_"),
                        type="text",  # safe default; comparator will probe numeric values
                    )
                )
        _add_schema(synth, "student_table_columns_fallback")

        schema_for_extraction = list(seen.values())
        fields = [s.student_field_name for s in schema_for_extraction]

        if not fields:
            logger.warning(
                "step_skip",
                step="data_extraction",
                reason="no extraction features from any source",
            )
            return result

        logger.info(
            "step_start",
            step="data_extraction",
            sources=sources_used,
            n_fields=len(fields),
            n_papers=len(student_input.selected_papers),
            n_extractors=len(self._extractors),
            fields=fields,
        )

        # Record extractor IDs upfront so the report always knows which
        # LLM columns to render even if some extractors fail.
        result.extractor_ids = [e.extractor_id for e in self._extractors]

        try:
            # Concurrency model:
            #   - Up to ``_PAPER_CONCURRENCY`` papers are processed in parallel.
            #   - Within a single paper, extractors still run sequentially,
            #     so the peak number of simultaneous LLM calls is capped at
            #     ``_PAPER_CONCURRENCY`` (not multiplied by n_extractors).
            #     This avoids tripping LLM-provider rate limits.
            #   - ``asyncio.gather(return_exceptions=True)`` ensures one
            #     failed paper doesn't abort the rest.
            sem = asyncio.Semaphore(_PAPER_CONCURRENCY)

            # Step 2 → Step 3 gate. When Step 2 ran, papers explicitly
            # confirmed NOT_FOUND in CrossRef are skipped here so we don't
            # waste full-text retrieval / LLM calls on phantom citations.
            # When the dict is empty (Step 2 disabled or all exceptions),
            # we fall back to processing every paper.
            from react_review.pipeline.schemas import (
                is_paper_eligible_for_extraction,
            )
            statuses = result.paper_verification_statuses
            skipped_keys: set[str] = set()
            if statuses:
                for ref in student_input.selected_papers:
                    key = self._paper_key(ref)
                    status = statuses.get(key)
                    if status is None:
                        continue
                    if not is_paper_eligible_for_extraction(status):
                        skipped_keys.add(key)
                        result.all_flags.append(
                            ValidationFlag(
                                step=PipelineStep.DATA_EXTRACTION,
                                severity=ValidationSeverity.WARNING,
                                code="PAPER_SKIPPED_NOT_VERIFIED",
                                message=(
                                    f"Skipped extraction for "
                                    f"'{ref.title[:60]}...' — Step 2 returned "
                                    f"{status.value}. The paper could not be "
                                    "verified to exist in CrossRef/PubMed."
                                ),
                            )
                        )

            async def _process_one(ref):
                """Retrieve full text for one paper and run all extractors.

                Returns a tuple ``(ref, doc, tables)``:
                  - ``doc`` is None if retrieval failed (flag raised upstream)
                    or if the paper was skipped by the Step 2 gate;
                  - ``tables`` is a list of ExtractedTable (one per extractor
                    that succeeded). Extractor exceptions are logged and
                    dropped so successful extractors' outputs still land.
                """
                async with sem:
                    # Step 2 → Step 3 gate.
                    if self._paper_key(ref) in skipped_keys:
                        return (ref, None, [])

                    doc = await self._paper_retriever.retrieve(ref)
                    if doc is None:
                        return (ref, None, [])

                    tables = []
                    for extractor in self._extractors:
                        try:
                            table = await extractor.extract(
                                doc,
                                schema_for_extraction,
                                research_context=student_input.research_context,
                            )
                            tables.append(table)
                            logger.info(
                                "extraction_done",
                                paper=ref.title[:40],
                                extractor=extractor.extractor_id,
                                fields=len(table.fields),
                            )
                        except Exception as ex_exc:
                            logger.error(
                                "extractor_failed",
                                paper=ref.title[:40],
                                extractor=extractor.extractor_id,
                                error=str(ex_exc),
                            )
                    return (ref, doc, tables)

            paper_results = await asyncio.gather(
                *(_process_one(ref) for ref in student_input.selected_papers),
                return_exceptions=True,
            )

            # Aggregate in original paper order so extracted_tables is
            # deterministic regardless of completion timing.
            for ref, outcome in zip(
                student_input.selected_papers, paper_results
            ):
                if isinstance(outcome, Exception):
                    logger.error(
                        "paper_extract_failed",
                        title=ref.title[:50],
                        error=str(outcome),
                    )
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.DATA_EXTRACTION,
                            severity=ValidationSeverity.ERROR,
                            code="PAPER_EXTRACT_FAILED",
                            message=(
                                f"Extraction raised an exception for "
                                f"'{ref.title[:60]}...': {outcome}"
                            ),
                        )
                    )
                    continue

                _ref, doc, tables = outcome
                if doc is None:
                    # Distinguish gate-skipped papers from genuine
                    # retrieval failures to avoid double-flagging.
                    if self._paper_key(ref) in skipped_keys:
                        continue
                    logger.warning(
                        "paper_not_retrievable",
                        title=ref.title[:50],
                    )
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.DATA_EXTRACTION,
                            severity=ValidationSeverity.WARNING,
                            code="PAPER_NOT_RETRIEVABLE",
                            message=f"Could not retrieve full text: '{ref.title[:60]}...'",
                        )
                    )
                    continue

                result.extracted_tables.extend(tables)

        except Exception as exc:
            logger.error("step_failed", step="data_extraction", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.DATA_EXTRACTION,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Data extraction failed: {exc}",
                )
            )
        return result

    # ==================================================================
    # Step 4: Table Comparison + Report
    # ==================================================================

    @staticmethod
    def _normalise_doi(doi: str) -> str:
        """Normalise a DOI for comparison: lowercase, strip URL prefix."""
        d = doi.strip().lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if d.startswith(prefix):
                d = d[len(prefix):]
        return d

    @classmethod
    def _paper_key(cls, ref) -> str:
        """Stable identifier for a paper used to index Step 2's status dict.

        Prefers the normalised DOI; falls back to a 60-char title slug
        when no DOI is available. The same function is used by Step 2
        (writing the status) and Step 3 (reading the gate), so both
        sides agree on the key.
        """
        if getattr(ref, "doi", None):
            normalised = cls._normalise_doi(ref.doi)
            if normalised:
                return f"doi:{normalised}"
        title = (getattr(ref, "title", "") or "").strip().lower()
        return f"title:{title[:60]}"

    def _find_matching_tables(
        self,
        paper_id: str,
        paper_title: str,
        tables: list,
    ) -> list:
        """Find tables whose paper_id matches by normalised DOI or title."""
        norm_id = self._normalise_doi(paper_id) if paper_id else ""
        matched = []
        for t in tables:
            t_norm = self._normalise_doi(t.paper_id) if t.paper_id else ""
            # Exact normalised DOI match
            if norm_id and t_norm and norm_id == t_norm:
                matched.append(t)
                continue
            # Title-based match (if paper_id contains the title substring)
            if paper_title and t.paper_id:
                t_lower = t.paper_id.lower()
                title_lower = paper_title.lower()[:40]
                if title_lower in t_lower or t_lower in title_lower:
                    matched.append(t)
        return matched

    async def _run_table_comparison(
        self, result: PipelineRunResult
    ) -> PipelineRunResult:
        """Step 4: Compare student's tables with AI-generated tables.

        Iterates over ALL selected papers (not just student-submitted tables)
        so every paper appears in the report. Uses DOI normalisation and
        title fallback to match student tables ↔ AI tables.
        """
        logger.info("step_start", step="table_comparison")
        try:
            comparison_results = []

            # Build the list of ALL papers to compare
            all_papers = result.student_input.selected_papers

            # Pre-assign orphaned student tables (paper_id = "") by index.
            # This handles the case where the PDF parser dropped DOIs.
            orphaned_student_tables = [
                t for t in result.student_input.submitted_tables if not t.paper_id.strip()
            ]
            orphan_idx = 0  # cursor into orphaned tables

            for ref in all_papers:
                paper_doi = ref.doi or ""
                paper_title = ref.title or ""

                # Find matching student table (by DOI or title)
                student_tables = self._find_matching_tables(
                    paper_doi, paper_title, result.student_input.submitted_tables
                )
                student_table = student_tables[0] if student_tables else None

                # Fallback: if no match and orphaned tables remain, assign next one
                if student_table is None and orphan_idx < len(orphaned_student_tables):
                    student_table = orphaned_student_tables[orphan_idx]
                    orphan_idx += 1
                    logger.info(
                        "student_table_orphan_assigned",
                        paper=paper_title[:50],
                        table_paper_id=student_table.paper_id,
                    )

                # Find matching AI-generated tables
                model_tables = self._find_matching_tables(
                    paper_doi, paper_title, result.extracted_tables
                )

                # Canonical paper_id for comparison: prefer validated DOI
                canonical_id = paper_doi or paper_title[:60]

                if student_table and model_tables:
                    # Ensure the student table uses the canonical paper_id
                    # so the comparison result's paper_id is always the DOI.
                    # IMPORTANT: we do NOT synthesise empty rows here —
                    # empty values on the student side must remain empty
                    # so the comparator can correctly assign MISSING_STUDENT.
                    from react_review.steps.data_extraction.schemas import (
                        ExtractedTable,
                    )
                    normalised_student = ExtractedTable(
                        paper_id=canonical_id,
                        fields=student_table.fields,
                        extractor_id=student_table.extractor_id,
                    )
                    comp = await self._table_comparator.compare(
                        normalised_student,
                        model_tables,
                        schema=result.student_input.evidence_schema or None,
                    )
                    comparison_results.append(comp)
                elif model_tables:
                    # AI extracted data but no student table. Feed a
                    # truly empty student table to the comparator so it
                    # can mark every field as MISSING_STUDENT — we no
                    # longer synthesise placeholder student rows.
                    from react_review.steps.data_extraction.schemas import (
                        ExtractedTable,
                    )
                    empty_student = ExtractedTable(
                        paper_id=canonical_id,
                        fields=[],
                        extractor_id="student",
                    )
                    comp = await self._table_comparator.compare(
                        empty_student,
                        model_tables,
                        schema=result.student_input.evidence_schema or None,
                    )
                    comparison_results.append(comp)
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.TABLE_COMPARISON,
                            severity=ValidationSeverity.INFO,
                            code="NO_STUDENT_TABLE",
                            message=(
                                f"No student table for '{paper_title[:60]}'. "
                                "Showing AI extraction only."
                            ),
                        )
                    )
                elif student_table:
                    # Student submitted a table but AI couldn't extract
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.TABLE_COMPARISON,
                            severity=ValidationSeverity.WARNING,
                            code="NO_MODEL_TABLE",
                            message=(
                                f"No AI-generated table for '{paper_title[:60]}'. "
                                "Cannot compare."
                            ),
                        )
                    )
                else:
                    # Neither side has data for this paper
                    result.all_flags.append(
                        ValidationFlag(
                            step=PipelineStep.TABLE_COMPARISON,
                            severity=ValidationSeverity.WARNING,
                            code="NO_TABLES",
                            message=(
                                f"No data for '{paper_title[:60]}' "
                                "from student or AI."
                            ),
                        )
                    )

            result.comparison_results = comparison_results

            # Generate final report
            result.report = await self._report_generator.generate(
                comparison_results, result.run_id
            )

            for flag in result.report.overall_flags:
                result.all_flags.append(
                    ValidationFlag(
                        step=PipelineStep.TABLE_COMPARISON,
                        severity=flag.severity,
                        code=flag.code,
                        message=flag.message,
                    )
                )
        except Exception as exc:
            logger.error("step_failed", step="table_comparison", error=str(exc))
            result.all_flags.append(
                ValidationFlag(
                    step=PipelineStep.TABLE_COMPARISON,
                    severity=ValidationSeverity.ERROR,
                    code="STEP_FAILED",
                    message=f"Table comparison failed: {exc}",
                )
            )
        return result
