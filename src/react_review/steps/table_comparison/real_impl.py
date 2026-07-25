"""Real implementations for step 4: table comparison and reporting.

Step 0 now produces a rich ``evidence_schema`` whose ``student_field_name``
is the same string the student wrote and the AI extractor returned —
so Step 4 can join the two sides directly by field name rather than
through an alias dictionary. ``EvidenceFieldSchema.type`` selects the
comparator (numeric / text / categorical / etc.) and the per-field
``threshold_match`` / ``threshold_partial`` values control the
classification bands.

Architectural changes from the previous implementation
------------------------------------------------------

  * Removed the ``_ALIAS_TO_BASE`` canonical-key dictionary and all
    group-tag heuristics — Step 0 carries that knowledge in the schema.
  * Removed ``_compare_authors`` — author / year / DOI metadata are
    now validated by Step 2 (CrossRef) and are skipped here via
    ``is_metadata=True``.
  * Per-field thresholds replace global 1% / 10% bands for numerics
    and 0.90 / 0.70 bands for text.
  * Tool synonyms are still applied (only for fields where the schema
    sets ``synonym_check=True``), since long-form vs abbreviated
    measurement-tool names are still common in real reports.

Verdict logic in :class:`RealReportGenerator` is unchanged from the
previous implementation:

  - ``skipped > 0`` AND ``compared == 0``  → INCOMPLETE
  - ``skipped > 0`` OR ``avg_coverage < 0.5`` → PARTIAL
  - ``avg_agreement < 0.6`` → FAIL
  - else PASS
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from react_review.core.enums import (
    ComparisonFlagCode,
    FieldStatus,
    ReportVerdict,
    ValidationSeverity,
)
from react_review.pipeline.schemas import EvidenceFieldSchema
from react_review.steps.data_extraction.schemas import ExtractedTable
from react_review.steps.table_comparison.interfaces import (
    ReportGenerator,
    TableComparator,
)
from react_review.steps.table_comparison.schemas import (
    ComparisonFlag,
    EvaluationReport,
    FieldDiff,
    TableComparisonResult,
)


# ======================================================================
# Default thresholds (used when schema is missing)
# ======================================================================

_DEFAULT_NUMERIC_THRESHOLDS: tuple[float, float] = (0.01, 0.10)
"""Numeric (relative-error) MATCH / PARTIAL_MATCH bounds."""

_DEFAULT_TEXT_THRESHOLDS: tuple[float, float] = (0.90, 0.70)
"""Text-similarity (SequenceMatcher) MATCH / PARTIAL_MATCH bounds."""


# ======================================================================
# Tool / measurement synonyms — applied when ``synonym_check=True``
# ======================================================================
#
# Kept tightly scoped to clinically common cardiology terms used in the
# project's demo dataset. Adding new synonym groups is intentionally
# manual: the LLM extractor already canonicalises most labels, and
# adding fuzzy synonyms blindly causes false matches.

_TOOL_SYNONYMS: dict[str, set[str]] = {
    "ccta": {
        "ccta", "cardiac cta",
        "cardiac computed tomography angiography",
        "coronary computed tomography angiography",
        "coronary ct angiography",
        "ct angiography", "computed tomography angiography",
        "cardiac ct", "coronary ct",
    },
    "cmr": {
        "cmr", "mri", "cardiac mri",
        "cardiovascular magnetic resonance",
        "cardiac magnetic resonance",
        "cardiac mr",
    },
    # Plain CT last — only used when nothing more specific matches.
    "ct": {"cardiac ct scan", "computed tomography"},
    "echo": {
        "echo", "echocardiography",
        "transthoracic echocardiography",
        "transthoracic echo",
    },
}


# ======================================================================
# Normalisation helpers
# ======================================================================


def _normalise_text(value: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation."""
    s = value.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,;:!?\"'()-_/[]")
    return s


def _normalise_tool(value: str) -> str:
    """Map a measurement-tool label to its canonical synonym key.

    Returns the input lowercased / cleaned when no synonym matches —
    so unknown tools still go through plain text comparison.
    """
    s = value.strip().lower()
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;")
    for canonical, syns in _TOOL_SYNONYMS.items():
        if s in syns:
            return canonical
        for syn in syns:
            if syn in s:
                return canonical
    return s


def _normalise_numeric(value: object) -> str | None:
    """Pull the first numeric token out of a value-string.

    Handles common medical-paper formats:
      "52.3 ± 8.1" → "52.3"
      "52.3+/-8.1" → "52.3"
      "52.3 (36.1-65.5) cm3" → "52.3"
      "52,3" → "52.3"   (European decimal comma)
      "52.3%" → "52.3"
      "52" → "52"
    Returns ``None`` if no number can be parsed.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = (
        s.replace(",", ".")
        .replace("±", "+-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("+/-", "+-")
    )
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    num = m.group(0)
    try:
        f = float(num)
    except ValueError:
        return num
    if f == int(f):
        return str(int(f))
    return f"{f:.6g}"


def _try_float(v: object) -> float | None:
    """Parse a value as float using the numeric normaliser; returns None on failure."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        n = _normalise_numeric(v)
        if n is None:
            return None
        try:
            return float(n)
        except ValueError:
            return None
    return None


# ======================================================================
# Pair-wise comparison primitives
# ======================================================================


def _compare_numeric(
    s_num: float, m_num: float, t_match: float, t_partial: float
) -> tuple[FieldStatus, str]:
    """Compare two numbers using a per-field relative-error band.

    ``t_match`` and ``t_partial`` are the MATCH / PARTIAL_MATCH
    upper-bounds on relative error. Defaults (1% / 10%) are applied by
    the caller when the schema does not specify them.
    """
    denom = max(abs(s_num), abs(m_num), 1.0)
    rel = abs(s_num - m_num) / denom
    if rel <= t_match:
        return FieldStatus.MATCH, f"numeric match ({s_num} ≈ {m_num})"
    if rel <= t_partial:
        return FieldStatus.PARTIAL_MATCH, (
            f"numeric close ({s_num} vs {m_num}, rel={rel:.2%})"
        )
    return FieldStatus.DIFF, f"numeric DIFF ({s_num} vs {m_num}, rel={rel:.2%})"


def _compare_text(
    s: str, m: str, t_match: float, t_partial: float
) -> tuple[FieldStatus, str]:
    """Compare two strings via SequenceMatcher similarity."""
    if not s and not m:
        return FieldStatus.NOT_COMPARABLE, "both empty"
    if s == m and s:
        return FieldStatus.MATCH, "exact match"
    sim = SequenceMatcher(None, s, m).ratio() if (s or m) else 0.0
    if sim >= t_match:
        return FieldStatus.MATCH, f"text match (sim={sim:.2f})"
    if sim >= t_partial:
        return FieldStatus.PARTIAL_MATCH, f"text close (sim={sim:.2f})"
    return FieldStatus.DIFF, f"text DIFF (sim={sim:.2f}): '{s}' vs '{m}'"


def _resolve_thresholds(
    schema_entry: EvidenceFieldSchema | None, fallback: tuple[float, float]
) -> tuple[float, float]:
    """Prefer per-field thresholds from the schema; fall back to defaults."""
    if (
        schema_entry is not None
        and schema_entry.threshold_match is not None
        and schema_entry.threshold_partial is not None
    ):
        return (schema_entry.threshold_match, schema_entry.threshold_partial)
    return fallback


def _normalise_for_compare(
    value: object, schema_entry: EvidenceFieldSchema | None
) -> object:
    """Normalise a value according to its schema entry (or sensible defaults).

    The comparator runs its dispatch on the **normalised** value so that
    "52.3 ± 8.1" and "52.3" agree on the numeric side, and "Coronary CT"
    and "CCTA" agree once tool synonyms have been applied.
    """
    if value is None:
        return None
    if isinstance(value, (bool, list)):
        return value

    s = str(value)

    # Schema-driven dispatch
    if schema_entry is not None:
        t = schema_entry.type
        if schema_entry.synonym_check:
            return _normalise_tool(s)
        if t in ("numeric", "year"):
            n = _normalise_numeric(s)
            return n if n is not None else _normalise_text(s)
        if t == "doi":
            return s.strip().lower()
        # text / categorical / author / unknown
        return _normalise_text(s)

    # No schema — probe by value type.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        n = _normalise_numeric(str(value))
        if n is not None:
            return n
    n = _normalise_numeric(s)
    if n is not None:
        return n
    return _normalise_text(s)


def _compare_pair(
    student_value: object,
    model_value: object,
    schema_entry: EvidenceFieldSchema | None,
) -> tuple[FieldStatus, str]:
    """Compare one (student, model) value pair and classify the outcome."""
    s_empty = student_value is None or (
        isinstance(student_value, str) and not student_value.strip()
    )
    m_empty = model_value is None or (
        isinstance(model_value, str) and not model_value.strip()
    )
    if s_empty and m_empty:
        return FieldStatus.NOT_COMPARABLE, "both empty"
    if s_empty:
        return FieldStatus.MISSING_STUDENT, "student empty, model has value"
    if m_empty:
        return FieldStatus.MISSING_MODEL, "model empty (extractor gap)"

    sn = _normalise_for_compare(student_value, schema_entry)
    mn = _normalise_for_compare(model_value, schema_entry)

    declared_type = schema_entry.type if schema_entry is not None else None

    # Numeric-typed fields use relative-error comparison whenever both
    # sides parse as numbers; otherwise we fall back to a text compare
    # (surfaces "category fell into a numeric column" mismatches).
    if declared_type in ("numeric", "year"):
        s_num = _try_float(sn)
        m_num = _try_float(mn)
        t_match, t_partial = _resolve_thresholds(
            schema_entry, _DEFAULT_NUMERIC_THRESHOLDS
        )
        if s_num is not None and m_num is not None:
            return _compare_numeric(s_num, m_num, t_match, t_partial)
        # Either side failed to parse — fall through to text similarity.
        text_match, text_partial = _resolve_thresholds(
            None, _DEFAULT_TEXT_THRESHOLDS
        )
        return _compare_text(
            str(sn) if sn is not None else "",
            str(mn) if mn is not None else "",
            text_match,
            text_partial,
        )

    # Schema declares text / categorical / author / doi → use similarity.
    if declared_type is not None:
        t_match, t_partial = _resolve_thresholds(
            schema_entry, _DEFAULT_TEXT_THRESHOLDS
        )
        return _compare_text(
            str(sn) if sn is not None else "",
            str(mn) if mn is not None else "",
            t_match,
            t_partial,
        )

    # No schema — probe by parseability.
    s_num = _try_float(sn)
    m_num = _try_float(mn)
    if s_num is not None and m_num is not None:
        return _compare_numeric(
            s_num, m_num, *_DEFAULT_NUMERIC_THRESHOLDS
        )
    return _compare_text(
        str(sn) if sn is not None else "",
        str(mn) if mn is not None else "",
        *_DEFAULT_TEXT_THRESHOLDS,
    )


# Best-status priority used when collapsing multiple model values down to
# one verdict. Order: MATCH > PARTIAL_MATCH > NEEDS_REVIEW > DIFF >
# MISSING_STUDENT/MISSING_MODEL > NOT_COMPARABLE. The dual-LLM design
# uses "best wins" so a single extractor's failure does not penalise
# the student.
_STATUS_RANK: dict[FieldStatus, int] = {
    FieldStatus.MATCH: 5,
    FieldStatus.PARTIAL_MATCH: 4,
    FieldStatus.NEEDS_REVIEW: 3,
    FieldStatus.DIFF: 2,
    FieldStatus.MISSING_STUDENT: 1,
    FieldStatus.MISSING_MODEL: 1,
    FieldStatus.NOT_COMPARABLE: 0,
}


# ======================================================================
# Comparator
# ======================================================================


def _key(name: str) -> str:
    """Canonical join key — verbatim student field name, lower-cased."""
    return (name or "").strip().lower()


class RealTableComparator(TableComparator):
    """Compare a student's table against AI-extracted tables.

    Joins by ``student_field_name`` (Step 0 ensures both sides agree on
    that string). Per-field type and tolerance come from the optional
    ``schema``; when absent, sensible defaults apply.
    """

    async def compare(
        self,
        student_table: ExtractedTable,
        model_tables: list[ExtractedTable],
        schema: list[EvidenceFieldSchema] | None = None,
    ) -> TableComparisonResult:
        flags: list[ComparisonFlag] = []
        schema_lookup: dict[str, EvidenceFieldSchema] = {
            _key(s.student_field_name): s for s in (schema or [])
        }

        # ----- 1. Bucket fields by student_field_name -----
        # ``buckets[key] = (student_field|None, [model_fields])``
        buckets: dict[str, tuple[Any | None, list[Any]]] = {}

        for sf in student_table.fields:
            key = _key(sf.field_name)
            if not key:
                continue
            student_field, model_fields = buckets.get(key, (None, []))
            # If multiple student fields collide, prefer one with a value.
            if student_field is None or (
                student_field.value is None and sf.value is not None
            ):
                student_field = sf
            buckets[key] = (student_field, model_fields)

        for mt in model_tables:
            for mf in mt.fields:
                key = _key(mf.field_name)
                if not key:
                    continue
                student_field, model_fields = buckets.get(key, (None, []))
                model_fields.append(mf)
                buckets[key] = (student_field, model_fields)

        if not buckets:
            return TableComparisonResult(
                paper_id=student_table.paper_id,
                field_diffs=[],
                agreement_rate=0.0,
                coverage_rate=0.0,
                compared_count=0,
                total_count=0,
                flags=[
                    ComparisonFlag(
                        code=ComparisonFlagCode.NO_TABLES.value,
                        severity=ValidationSeverity.WARNING,
                        message=(
                            f"Paper {student_table.paper_id}: "
                            "no student or model fields available."
                        ),
                    )
                ],
                skipped=True,
            )

        # ----- 2. Emit one FieldDiff per bucket -----
        diffs: list[FieldDiff] = []
        compared_count = 0
        agree_count = 0
        coverable_count = 0
        missing_model_fields: list[str] = []
        mismatch_fields: list[str] = []
        needs_review_fields: list[str] = []

        for key, (student_field, model_fields) in buckets.items():
            display_name = (
                student_field.field_name
                if student_field is not None and student_field.field_name
                else (model_fields[0].field_name if model_fields else key)
            )
            schema_entry = schema_lookup.get(key)

            # Skip metadata fields entirely — they're validated by Step 2
            # (author / year / DOI), so re-comparing them here would
            # double-flag the student.
            if schema_entry is not None and schema_entry.is_metadata:
                continue

            s_val = student_field.value if student_field is not None else None
            s_evidence = (
                student_field.evidence if student_field is not None else ""
            )
            model_vals = [mf.value for mf in model_fields]
            model_evidence = [mf.evidence or "" for mf in model_fields]
            any_extractor_failed = any(
                getattr(mf, "extractor_failed", False) for mf in model_fields
            )

            s_empty = s_val is None or (
                isinstance(s_val, str) and not s_val.strip()
            )
            model_non_empty = [
                v for v in model_vals
                if v is not None and (not isinstance(v, str) or v.strip())
            ]
            m_empty = not model_non_empty

            student_norm = _normalise_for_compare(s_val, schema_entry)
            model_norms = [
                _normalise_for_compare(v, schema_entry) for v in model_vals
            ]

            # ----- Decide status -----
            if s_empty and m_empty:
                status = FieldStatus.NOT_COMPARABLE
                explanation = "both sides empty"
            elif m_empty:
                status = FieldStatus.MISSING_MODEL
                explanation = (
                    "extractor failure — value not produced by model"
                    if any_extractor_failed
                    else "model has no value (extractor gap)"
                )
                missing_model_fields.append(display_name)
            elif s_empty:
                status = FieldStatus.MISSING_STUDENT
                explanation = "student has no value; model provided one"
            else:
                # Both sides have values — pick the best across extractors.
                best_status = FieldStatus.NOT_COMPARABLE
                best_expl = ""
                for mv in model_vals:
                    if mv is None or (isinstance(mv, str) and not mv.strip()):
                        continue
                    st, expl = _compare_pair(s_val, mv, schema_entry)
                    if _STATUS_RANK[st] > _STATUS_RANK.get(best_status, 0):
                        best_status = st
                        best_expl = expl
                    elif not best_expl:
                        best_expl = expl
                status = best_status
                explanation = best_expl
                coverable_count += 1
                compared_count += 1
                if status in (FieldStatus.MATCH, FieldStatus.PARTIAL_MATCH):
                    agree_count += 1
                elif status == FieldStatus.NEEDS_REVIEW:
                    needs_review_fields.append(display_name)
                else:
                    mismatch_fields.append(display_name)

            is_consistent = status in (FieldStatus.MATCH, FieldStatus.PARTIAL_MATCH)

            if student_field is not None and model_fields:
                source_type = "student+llm"
            elif model_fields:
                source_type = "llm-only"
            elif student_field is not None:
                source_type = "student-only"
            else:
                source_type = ""

            diffs.append(
                FieldDiff(
                    field_name=display_name,
                    canonical_concept=(
                        schema_entry.canonical_concept if schema_entry else ""
                    ),
                    student_raw_name=(
                        student_field.field_name if student_field is not None else ""
                    ),
                    model_raw_names=[mf.field_name for mf in model_fields],
                    student_value=s_val,
                    student_value_normalized=student_norm,
                    student_evidence=s_evidence,
                    model_values=model_vals,
                    model_values_normalized=model_norms,
                    model_evidence=model_evidence,
                    status=status,
                    is_consistent=is_consistent,
                    explanation=explanation,
                    source_type=source_type,
                )
            )

        # ----- 3. Roll up agreement / coverage and emit summary flags -----
        total = len(diffs)
        agreement_rate = (agree_count / compared_count) if compared_count > 0 else 0.0
        coverage_rate = (coverable_count / total) if total > 0 else 0.0

        if mismatch_fields:
            names = ", ".join(mismatch_fields[:5])
            flags.append(
                ComparisonFlag(
                    code=ComparisonFlagCode.FIELD_MISMATCH.value,
                    severity=(
                        ValidationSeverity.ERROR
                        if compared_count > 0 and agreement_rate < 0.5
                        else ValidationSeverity.WARNING
                    ),
                    message=(
                        f"Paper {student_table.paper_id}: "
                        f"{len(mismatch_fields)} field(s) with disagreements "
                        f"(e.g. {names})."
                    ),
                )
            )
        if missing_model_fields:
            names = ", ".join(missing_model_fields[:5])
            flags.append(
                ComparisonFlag(
                    code=ComparisonFlagCode.EXTRACTOR_GAP.value,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Paper {student_table.paper_id}: extractor produced "
                        f"no value for {len(missing_model_fields)} field(s) "
                        f"(e.g. {names}). Not counted against the student."
                    ),
                )
            )
        if needs_review_fields:
            names = ", ".join(needs_review_fields[:5])
            flags.append(
                ComparisonFlag(
                    code=ComparisonFlagCode.NEEDS_REVIEW.value,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Paper {student_table.paper_id}: "
                        f"{len(needs_review_fields)} field(s) need human review "
                        f"(e.g. {names})."
                    ),
                )
            )

        return TableComparisonResult(
            paper_id=student_table.paper_id,
            field_diffs=diffs,
            agreement_rate=agreement_rate,
            coverage_rate=coverage_rate,
            compared_count=compared_count,
            total_count=total,
            flags=flags,
        )


# ======================================================================
# Report generator (verdict thresholds preserved from the prior version)
# ======================================================================


class RealReportGenerator(ReportGenerator):
    """Generates a final evaluation report from comparison results.

    Verdict (applied top-down, first match wins):
      - skipped > 0 AND compared == 0 → INCOMPLETE
      - skipped > 0 OR avg_coverage < 0.5 → PARTIAL
      - avg_agreement < 0.6 → FAIL
      - else PASS
    """

    async def generate(
        self,
        comparison_results: list[TableComparisonResult],
        run_id: str,
    ) -> EvaluationReport:
        all_flags: list[ComparisonFlag] = []
        for cr in comparison_results:
            all_flags.extend(cr.flags)

        if not comparison_results:
            return EvaluationReport(
                run_id=run_id,
                comparison_results=[],
                overall_flags=[],
                summary="No papers to compare. Step 3 data not available.",
                verdict=ReportVerdict.INCOMPLETE,
            )

        compared_results = [cr for cr in comparison_results if not cr.skipped]
        skipped_count = len(comparison_results) - len(compared_results)
        total = len(comparison_results)

        avg_agreement = (
            sum(cr.agreement_rate for cr in compared_results) / len(compared_results)
            if compared_results else 0.0
        )
        avg_coverage = sum(cr.coverage_rate for cr in comparison_results) / total

        if skipped_count > 0 and len(compared_results) == 0:
            verdict = ReportVerdict.INCOMPLETE
            detail = (
                f"{skipped_count}/{total} paper(s) could not be compared "
                "(extractor produced nothing comparable)."
            )
        elif skipped_count > 0 or avg_coverage < 0.5:
            verdict = ReportVerdict.PARTIAL
            detail = (
                f"Coverage is limited (avg={avg_coverage:.0%}, "
                f"skipped={skipped_count}/{total}). Agreement where "
                f"comparable: {avg_agreement:.0%}."
            )
        elif avg_agreement < 0.6:
            verdict = ReportVerdict.FAIL
            detail = f"Average agreement {avg_agreement:.0%} is below 60%."
        else:
            verdict = ReportVerdict.PASS
            detail = (
                f"All comparable papers show acceptable agreement "
                f"(avg={avg_agreement:.0%}, coverage={avg_coverage:.0%})."
            )

        # Low-coverage health warning — keeps the summary honest even
        # when agreement looks high on a tiny comparable subset.
        low_coverage_note = ""
        if avg_coverage < 0.3:
            low_coverage_note = (
                " WARNING: average agreement reflects only the small "
                "comparable subset; overall coverage is very low "
                f"({avg_coverage:.0%}), so overall confidence is limited."
            )
        elif avg_coverage < 0.5:
            low_coverage_note = (
                " Note: average agreement is computed only over the "
                "comparable fields; coverage is below 50%."
            )

        summary = (
            f"[{verdict.value}] Reviewed {total} paper(s) "
            f"(compared={len(compared_results)}, skipped={skipped_count}). "
            f"Average agreement (comparable fields only): {avg_agreement:.0%}. "
            f"Average coverage: {avg_coverage:.0%}. {detail}{low_coverage_note}"
        )

        return EvaluationReport(
            run_id=run_id,
            comparison_results=comparison_results,
            overall_flags=all_flags,
            summary=summary,
            verdict=verdict,
            avg_agreement=avg_agreement,
            avg_coverage=avg_coverage,
            compared_papers=len(compared_results),
            skipped_papers=skipped_count,
        )
