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
| `included_studies.csv` | — | **verify** | the 9 source papers: citation, country, N, modality, quality, **DOI**, review ref number |
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

- **Numeric tolerance = 3% (MVP).** Two numbers agree if the relative error of
  the **primary value** (mean / median / point estimate — the first number in a
  `mean ± SD` or `median (IQR)` cell) is ≤ 3%. SD / spread is secondary and does
  not by itself cause a mismatch in MVP.
- **Unit is a separate axis.** If the reported unit differs (e.g. `mm` vs `cm`,
  `kg/m2` vs `kg/m3`) the row is `unit_mismatch` regardless of how close the
  numbers are.
- `expected_label` taxonomy (MVP, may be revised): `match` | `mismatch` |
  `unit_mismatch`. **Distribution after applying 3% tolerance: 53 match / 0
  mismatch / 4 unit_mismatch.** The 4 unit_mismatch are Keles cm↔mm (A022/A025)
  and Svanteson `kg/m3` source typo (A010/A013).
- **Seeded discrepancies needed.** Because the review's Table-1 means all agree
  with the sources within 3% (0 real value-mismatches), the main axis alone
  cannot measure Auditor mismatch-recall. Add a small set of deliberately
  corrupted review values (Proposal §6 "Auditor recall on seeded discrepancies")
  before evaluating recall. The internal axis (IC01/IC02 Keles) provides a few
  real positive detections in the meantime.
- All CSVs are plain UTF-8 (no BOM). Re-editing in Excel may re-save as GBK or
  add a BOM — the loader should read with `utf-8-sig`; re-normalise before freeze.
- **Policy-sensitive rows** (flipped to `match` under 3%, override if a stricter
  policy is wanted): A003 Ahmad BMI (mean identical, SD 1.7 vs 1.77 — a real SD
  slip we ignore when auditing the mean); A049 Colom BMI (mean identical but the
  review cell string `27.0 ± 4.7/27.9` is malformed).

## How this is used

1. **Parser eval (review side).** Run the Parser on `raw/EAT_T1DM_SRMA.pdf` and
   compare its long-table output against `review_ground_truth.csv` (value-level
   accuracy + correct `study / group / field_type` assignment).
2. **Auditor eval (main axis, review vs source).** Run the Auditor over
   `audit_template.csv` (review_value vs source_value) and score its
   match/mismatch/unit_mismatch labels against `expected_label`.
3. **Auditor eval (secondary axis, review vs itself).** Score the Auditor
   against `internal_consistency.csv` (Table 1 vs forest / reference list).
4. **Auditor recall.** Add seeded discrepancies (see policy note) to measure
   mismatch-detection recall (Proposal §6).

## Provenance / caveats

All review-side values were transcribed by reading the rendered PDF pages
(Table 1 on p5; Fig 2/4 thickness forest plots on p5/p6; Fig 3 volume forest
plot on p6; DOIs from the reference list on p9). **These transcriptions are
pending human verification against the PDF before the benchmark is frozen.**
