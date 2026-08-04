# Phase 7 contracts for `melanoma_checkpoint_2017`

These three files are **additive**. The Phase 6A freeze — `manifest.json`,
`audit_template.csv`, `review_ground_truth.csv`, the two PDFs, the recorded
Phase 6B/6E caches — is unchanged, and a run that does not pass
`--benchmark-profile` behaves exactly as it did before they existed.

| File | Authority |
| --- | --- |
| `phase7_profile.json` | selects the prompt profiles; pins the SHA-256 of the frozen inputs and of the two files below |
| `phase7_semantic_overlay.csv` | Phase 7 semantic **expectations** (relation, specificity direction, review-required) |
| `phase7_target_contract.csv` | evaluation **input**: the review-side facts needed to ask a well-formed question |

## `phase7_target_contract.csv` is not an answer key

It carries only what the review itself shows: the review's own column label, the
review's own words for the cohort, the timepoint, and the captured cell. It
carries no `expected_*` column, no source arm, no target-assignment result, no
correct quote, and no expected found/label/relation. The loader rejects any
column outside the declared set, so it cannot acquire one later.

Every `audit_id` must appear exactly once and the set must correspond
one-to-one with `audit_template.csv`; a missing, repeated or unknown row is a
hard load failure.

### How it was derived

Each answer-key row was matched to a `review_ground_truth.csv` row on
(`study_id`, `group`, `field_type`, `value`) and each match was unique. The
matched id is recorded per row in `review_data_source`.

**The row numbers do not line up, and must not be assumed to.** `MA004` comes
from `M006`, `MA005` from `M004`, `MA007` from `M005`. A contract built on the
apparent `MA0xx ↔ M0xx` coincidence would have mislabelled three of the fifteen
rows.

`cohort_label` is the review's own name for the arm, which the review states in
its treatment-arm cells rather than in a separate cohort column; the row that
supplied it is recorded in `cohort_label_source`. The four study-level rows
(`-`) and the four comparison rows carry no single-arm label, so the field is
empty for them: a comparison's two sides are derived from the group key at run
time rather than asserted here.

## `phase7_semantic_overlay.csv`

Phase 6B recorded `review_broader` for `MA005`/`MA007`, where the review states
`Ipilimumab (3 mg/kg) + placebo` and the source states `ipilimumab group`. Under
the semantic prompt's own definition — `review_broader` means the review is
*less* specific — the review is the more specific side, so the expectation
contradicted the definition it was written against. `MA003` has the same shape
and was recorded as `same`.

The overlay restates all four semantic rows under one consistent definition and
adds the explicit specificity direction. It is a **correction of the relation
contract, not a re-scoring**: both broader directions already resolve to
`MATCH + review_required`, so no label moves because of it. The Phase 6B
metrics, caches and answer key are left exactly as they were archived.
