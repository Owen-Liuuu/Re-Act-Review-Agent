# Frozen cross-domain benchmark: melanoma checkpoint inhibitors

This Phase 6A benchmark moves the audit from the EAT/T1DM domain to oncology.
It is frozen around Karlsson and Saleh's 2017 systematic review and the
three-arm CheckMate 067 trial reported by Larkin et al. (2015).

## Why this benchmark was selected

- The review includes seven randomized controlled trials and several trials
  with more than two arms.
- Larkin 2015 has three mutually exclusive arms (314, 315, and 316 patients),
  which sum deterministically to 945.
- The review reports textual study-design and treatment-arm descriptions that
  cannot be settled by numeric comparison alone.
- The review reports PFS, hazard ratios, confidence intervals, and inequality
  expressions. These exercise the structured numeric path.
- A natural discrepancy is present: the source reports 99.5% CIs for two
  prespecified comparisons while the review calls the same bounds 95% CIs.

## Frozen scope

The benchmark is deliberately selected, not exhaustive:

- `review_ground_truth.csv` contains 17 high-value claims from Table 1, the
  primary forest plot, and the surrounding results text.
- `audit_template.csv` contains 15 review-to-source comparisons for Larkin
  2015: four semantic, four numeric, and seven structured comparisons.
- `reference_universe.csv` records all seven included trials, but only Larkin
  2015 has a locally frozen source PDF for Phase 6B.
- `selected_studies.csv` is the local-source input for the selected Phase 6B
  audit scope.

This benchmark therefore tests whether the mechanisms run across domains. It
does **not** claim expert-validated accuracy for all melanoma evidence.

## Capture/unpivot preflight

The preflight used a live table capture, then replayed that approved capture
through only the unpivot operation. It stopped before cohort resolution, DKB
field resolution, reference extraction, source retrieval, or audit.

- Frozen table whitelist: `table_1`, `table_3`, `table_4`.
- Long rows produced: 85 (35 + 41 + 9).
- Genuine text values that deterministic numeric parsing cannot consume: 7
  study-design cells.
- Structured components found in captured table cells: 0.
- Structured claims frozen from visually checked narrative/forest-plot content:
  7.

The first capture also identified two risk-of-bias graphics as tables. One
contained shape errors and model-estimated percentages. Both are excluded from
the frozen whitelist; no inferred chart percentages enter the benchmark.

## Reproducibility and licensing

- The review PDF is tracked because the publisher marks it CC BY-NC 3.0. See
  `LICENSES.md` for attribution and restrictions.
- The Larkin source PDF is copyrighted and marked for personal use only. It is
  kept locally under `raw/sources/`, is ignored by Git, and is pinned by SHA-256
  in `manifest.json`.
- A fresh clone can reproduce the benchmark contract and review-side parser
  work. Source-side live/replay evaluation additionally requires the local
  Larkin PDF whose hash matches the manifest.
- Live model output is not treated as a deterministic regression baseline.

## Phase 6B entry gate

Phase 6B may start only if all of the following hold:

1. the tracked review PDF hash matches `manifest.json`;
2. the local Larkin PDF exists and its hash matches the manifest;
3. the table whitelist remains exactly `table_1`, `table_3`, `table_4`;
4. semantic comparison is expected on at least the four rows marked
   `expected_match_mode=semantic`;
5. known gaps are reported rather than adjusted away.

## Phase 6B result

The entry gate passed and the frozen 15 rows were run once live, once in record
mode, and once by deterministic replay of the recording. All four expected text
rows reached controlled semantic comparison; the observed mode counts were the
frozen 4 semantic, 4 numeric, and 7 structured rows.

The cross-domain accuracy gate did not pass. The recorded replay produced 60%
label accuracy, 20% strict discrepancy precision, and 33.3% strict discrepancy
recall. Safety remained visible: all three expected discrepancies either
received a non-MATCH verdict or a MATCH carrying `review_required`, so the
silent-release count was zero and review visibility was 100%.

The dominant failure was multi-arm directed extraction. Later treatment, PFS,
and hazard-ratio targets were sometimes assigned the first nearby arm or effect
estimate. Incomplete CI extraction also turned determinate structured checks
into partial `MATCH + review_required` results. One frozen confidence-level gap
was exposed; the second was masked by an incomplete or wrong extraction. The
quote-free metrics and private-artifact hashes are published in
`docs/baselines/melanoma_phase6b_metrics.json`; reports containing source quotes
and raw model responses remain under ignored `output/baselines/`.
