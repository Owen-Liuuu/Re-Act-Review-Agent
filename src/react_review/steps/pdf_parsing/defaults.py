"""Default comparison thresholds for evidence-schema fields.

Per project decision #7C, this default table is the **primary** source of
truth for per-field comparison tolerances. The LLM may suggest thresholds
in the schema sub-task, but those are only used as a fallback when:

  1. No entry exists in this default table for the field's
     ``canonical_concept``, AND
  2. The LLM did suggest a value.

If neither default table nor LLM provides a value, the parser falls back
to type-level defaults (numeric: 1% / 10% relative-error; text-like: 0.90 /
0.70 SequenceMatcher similarity).

Threshold semantics
-------------------
The two threshold values mean different things by ``type``:

  * For ``"numeric"`` and ``"year"``:
      * ``threshold_match``   — relative-error **upper** bound for MATCH
        (smaller is stricter; e.g. 0.01 means values must be within 1%).
      * ``threshold_partial`` — relative-error upper bound for
        PARTIAL_MATCH (must be > ``threshold_match``).

  * For ``"text"``, ``"categorical"``, ``"author"``, ``"doi"``:
      * ``threshold_match``   — SequenceMatcher similarity **lower** bound
        for MATCH (bigger is stricter; e.g. 0.90 means strings must be
        ~90% similar).
      * ``threshold_partial`` — similarity lower bound for PARTIAL_MATCH
        (must be < ``threshold_match``).

The Step 4 comparator dispatches based on ``type`` and applies these
thresholds in the right direction.
"""
from __future__ import annotations


# ----------------------------------------------------------------------
# Type-level defaults (used when neither the per-concept table nor the
# LLM provides a value).
# ----------------------------------------------------------------------

_DEFAULT_NUMERIC: tuple[float, float] = (0.01, 0.10)        # rel-error MATCH=1%, PARTIAL=10%
_DEFAULT_TEXT:    tuple[float, float] = (0.90, 0.70)        # similarity MATCH=0.90, PARTIAL=0.70
_DEFAULT_DOI:     tuple[float, float] = (1.0, 1.0)          # DOIs must match exactly
_DEFAULT_YEAR:    tuple[float, float] = (0.0, 0.0)          # year must match exactly


# ----------------------------------------------------------------------
# Per-concept default thresholds.
# Keyed on ``canonical_concept`` (snake_case, lower-case). Values are
# (threshold_match, threshold_partial).
#
# Keep this table small and conservative — about 10-15 well-known
# biomedical fields. Per decision #8, this is the "fallback" we maintain;
# everything else relies on LLM-assigned thresholds or type defaults.
# ----------------------------------------------------------------------

_PER_CONCEPT_DEFAULTS: dict[str, tuple[float, float]] = {
    # Sample size: small absolute deviation is suspicious, so we use a
    # tight relative-error bound. PARTIAL=5% accommodates rounding for
    # very large studies; MATCH=0% requires exact agreement.
    "sample_size":      (0.0, 0.05),
    "n":                (0.0, 0.05),

    # Demographic numerics: rounding tolerated, 10% partial bound matches
    # typical reporting variability across reviews.
    "age":              (0.01, 0.10),
    "age_mean":         (0.01, 0.10),
    "bmi":              (0.01, 0.10),
    "weight":           (0.01, 0.10),
    "height":           (0.01, 0.10),

    # Imaging-derived measurements (e.g. EAT thickness/volume) are
    # noisier across labs, so a wider partial band is appropriate.
    "eat_or_eft":       (0.05, 0.20),
    "eat_thickness":    (0.05, 0.20),
    "eat_volume":       (0.05, 0.20),

    # Statistical outputs: tight match band; the partial band stays
    # narrow because crossing the 0.05 significance line is a meaningful
    # difference even when relative error is small.
    "p_value":          (0.001, 0.01),
    "hazard_ratio":     (0.05, 0.15),
    "odds_ratio":       (0.05, 0.15),
    "risk_ratio":       (0.05, 0.15),

    # Categorical / text fields. Use similarity-based thresholds.
    "country":          (1.0, 0.85),    # nearly exact preferred
    "study_design":     (0.95, 0.75),
    "measurement_tool": (0.95, 0.75),   # synonym table normalises before compare
    "intervention":     (0.90, 0.70),
    "outcome":          (0.90, 0.70),

    # Metadata-style fields (also flagged ``is_metadata=True`` so Step 4
    # skips them; thresholds shown here are used only if metadata flag
    # is False for some reason).
    "year":             _DEFAULT_YEAR,
    "publication_year": _DEFAULT_YEAR,
    "doi":              _DEFAULT_DOI,
    "author":           (0.90, 0.75),
    "first_author":     (0.90, 0.75),
}


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def resolve_field_thresholds(
    *,
    canonical_concept: str,
    type_: str,
    llm_match: float | None = None,
    llm_partial: float | None = None,
) -> tuple[float | None, float | None]:
    """Resolve per-field MATCH / PARTIAL_MATCH thresholds.

    Resolution order (per decision #7C):

      1. Default table for ``canonical_concept`` — if present, win.
      2. LLM-supplied thresholds — used only when default table has no
         entry and LLM provided both values.
      3. Type-level default — used when neither (1) nor (2) provides a
         value.

    Args:
        canonical_concept: Snake_case concept name (e.g. ``sample_size``).
        type_: One of ``numeric`` / ``text`` / ``categorical`` /
            ``author`` / ``year`` / ``doi``.
        llm_match: Optional LLM-suggested MATCH threshold.
        llm_partial: Optional LLM-suggested PARTIAL_MATCH threshold.

    Returns:
        ``(threshold_match, threshold_partial)`` — same units as the
        ``EvidenceFieldSchema`` interpretation rules.
    """
    key = (canonical_concept or "").strip().lower()
    if key in _PER_CONCEPT_DEFAULTS:
        return _PER_CONCEPT_DEFAULTS[key]

    if llm_match is not None and llm_partial is not None:
        return (llm_match, llm_partial)

    return _type_default(type_)


def _type_default(type_: str) -> tuple[float, float]:
    """Pick the type-level default when no concept-specific entry exists."""
    t = (type_ or "").strip().lower()
    if t in ("numeric",):
        return _DEFAULT_NUMERIC
    if t == "year":
        return _DEFAULT_YEAR
    if t == "doi":
        return _DEFAULT_DOI
    # text / categorical / author / unknown all use similarity defaults.
    return _DEFAULT_TEXT
