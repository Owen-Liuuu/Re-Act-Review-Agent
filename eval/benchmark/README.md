# Ground-truth benchmark: Sooragonda 2025 EAT/T1DM SRMA

Hand-labelled benchmark for evaluating ReAct-Review's Parser (review-side
extraction) and Auditor (review-vs-source cross-validation).

## Source review

- **Title:** Epicardial adipose tissue in type 1 diabetes mellitus: a systematic
  review and meta-analysis
- **Authors:** Sooragonda B.G., Karnwal V., Gunaseelan V., Joshi A., Biswas K.,
  Shenoy M.T., Anbazhagan R.
- **Journal:** The Egyptian Heart Journal (2025) 77:71
- **DOI:** 10.1186/s43044-025-00666-8
- **PDF:** `raw/EAT_T1DM_SRMA.pdf` (9 pages)
- **Included studies:** 9 (T1DM vs healthy control; EAT thickness by echo, EAT
  volume by CT)

## Files

| file | side | status | contents |
|---|---|---|---|
| `raw/EAT_T1DM_SRMA.pdf` | — | final | the review PDF |
| `pdf/*.pdf` | source | present (git-ignored) | the 9 source-paper PDFs — **copyrighted, kept local only**; not committed |
| `included_studies.csv` | — | **verify** | the 9 source papers: citation, country, N, modality, quality, **DOI**, review ref number, and `source_pdf` (path under `pdf/`) |
| `seeded_discrepancies.csv` | recall | ready | 10 deliberately-corrupted rows (8 positive + 2 negative controls) to measure Auditor recall and precision |
| `review_ground_truth.csv` | review | **verify** | long-format table of every value the review reports (`study × group × field_type × value`), with `source_location` (Table 1 / Fig 2 / Fig 3 / Fig 4) |
| `audit_template.csv` | source (main axis) | **filled** | one row per auditable value (57): `review_value` + `source_value` + `source_quote` + `source_location_in_paper` + `source_unit` + `expected_label`, hand-annotated from the source papers. Labels re-derived under the 3% tolerance policy below |
| `internal_consistency.csv` | review-internal (secondary axis) | **verify** | structured answer key (IC01–IC04) for the review contradicting itself (Table 1 vs forest plots vs reference list); each row `error_owner` + `needs_human_review` |
| `known_internal_discrepancies.md` | — | **verify** | human-readable narrative behind `internal_consistency.csv` |

## Column meaning (`review_ground_truth.csv`)

Aligned to the planned 4-table matching model (ReviewDataTable side):

- `review_data_id` — stable id (`R001`, …).
- `study_id` — e.g. `ahmad_2022` (join key to `included_studies.csv`).
- `group` — `t1dm` | `control` | `-` (study-level rows: N / country / tool /
  quality) | `all` (CT studies that report a single combined cohort with no
  T1DM/control split, i.e. de Gonzalo-Calvo 2018 and Colom 2018).
- `timepoint` — `single` (all included studies are cross-sectional).
- `field_type` — `sample_size` | `country` | `measurement_tool` |
  `overall_quality` | `age` | `bmi` | `eat_thickness` | `eat_volume` |
  `subgroup_n`.
- `raw_field_name` — the verbatim column header as printed (`N`, `Age`,
  `BMI Kg/m2`, `EFT/ EAT`, `Measurement tool`, `Overall quality`).
- `value` — verbatim value string (spread kept as-is, e.g. `6.60 ± 0.71`,
  `52.3 (36.1–65.5)`).
- `unit` — `mm` | `cm3` | `kg/m2` | `years` | `count` | ``.
- `source_location` — where in the PDF the value is printed. Values appearing
  in BOTH Table 1 and a forest plot get one row each, so table-vs-figure
  disagreements are captured as distinct rows.

## Tolerance policy, labels, encoding

- **Dual tolerance (MVP — may be revisited): mean 1% + SD 3%.** A `mean ± SD`
  value has its MEAN (primary value — first number in the cell) compared at ≤ 1%
  and, when BOTH sides report an SD, its SD compared at ≤ 3%. Both bands must
  hold for a MATCH; the SD band is looser because SDs are noisier and it is
  skipped when either side is a range / IQR / point value. Each row carries the
  computed `rel_error_pct` (mean) and `sd_rel_error_pct` (SD, when applicable).
- **Unit is a separate axis.** If the reported unit differs (e.g. `mm` vs `cm`,
  `kg/m2` vs `kg/m3`) the row is `unit_mismatch` regardless of how close the
  numbers are.
- `expected_label` taxonomy (MVP, may be revised): `match` | `mismatch` |
  `unit_mismatch`. **Distribution under the dual band: 52 match / 1 mismatch / 4
  unit_mismatch.** The single `mismatch` is **A003 Ahmad BMI** — mean identical
  (20.57) but SD **1.7 vs 1.77 = 3.95% > 3%**, a genuine SD transcription slip
  (1.77 does not round to 1.7) that the mean-only rule missed. The 4
  `unit_mismatch` are Keles cm↔mm (A022/A025, a review error) and Svanteson
  `kg/m3` (A010/A013, a source-paper typo). More numeric-mismatch positives come
  from `seeded_discrepancies.csv`.
- **Seeded discrepancies** (`seeded_discrepancies.csv`) provide the recall/precision
  positives the natural review lacks: 10 rows = 6 `mismatch` (gross shift; subtle
  just above 1% = S02 at 1.23%; group-swap; decimal error; digit transpose;
  fabrication) + 2 `unit_mismatch` (mm→cm) + 2 `match` negative controls (S03 a
  0.75% rounding just below 1% that must stay match, and S09 an exact match).
  S02/S03 straddle the 1% boundary to test tolerance calibration. `should_flag`
  marks the 8 positives. Each seed derives from a real benchmark value;
  `review_value` is the corrupted value, `source_value` is the truth.
- All CSVs are plain UTF-8 (no BOM). Re-editing in Excel may re-save as GBK or
  add a BOM — the loader should read with `utf-8-sig`; re-normalise before freeze.

## What this benchmark measures

There are two INDEPENDENT extraction ground-truths (does the system read each
side correctly) plus the audit judgement key (does it compare correctly):

| # | capability under test | ground truth | catches |
|---|---|---|---|
| 1 | **Review-side extraction** — Parser reads the review PDF (原文) | `review_ground_truth.csv` | Parser misreading the review's Table 1 / figures |
| 2 | **Source-side extraction** — Collector reads each source paper (源论文) | `source_value` column in `audit_template.csv`, run against the 9 PDFs in `pdf/` (mapped via `included_studies.csv` → `source_pdf`) | Collector misreading a source paper |
| 3 | **Audit judgement** — Auditor compares review value vs source value | `expected_label` column in `audit_template.csv` | wrong match / mismatch / unit_mismatch verdict |
| 4 | **Internal consistency** — Auditor checks the review against itself | `internal_consistency.csv` | review self-contradiction (Table vs figure, citation nos) |
| 5 | **Auditor recall + precision** | `seeded_discrepancies.csv` | failure to flag injected errors (recall) / over-flagging correct values (precision) |

Targets **1** and **2** are the two extraction benchmarks: #1 grades extraction
from the review itself, #2 grades extraction from the cited source papers. They
are separate because a wrong audit verdict can come from either a bad review
read, a bad source read, or a bad comparison — and this split tells them apart.
(Fully exercising #2 needs the 9 source PDFs; for now the hand-annotated
`source_value` is the ground truth.)

## Provenance / caveats

All review-side values were transcribed by reading the rendered PDF pages
(Table 1 on p5; Fig 2/4 thickness forest plots on p5/p6; Fig 3 volume forest
plot on p6; DOIs from the reference list on p9). **These transcriptions are
pending human verification against the PDF before the benchmark is frozen.**
