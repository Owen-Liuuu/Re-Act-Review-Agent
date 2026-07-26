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
| `audit_template.csv` | source | **filled** | one row per auditable value (57): `review_value` + `source_value` + `source_quote` + `source_location_in_paper` + `source_unit` + `expected_label`, hand-annotated from the source papers |
| `known_internal_discrepancies.md` | — | **verify** | inconsistencies WITHIN the review (table vs figures); free audit targets |

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

## Labels and encoding

- `expected_label` taxonomy (MVP, may be revised): `match` | `mismatch` |
  `unit_mismatch` (value agrees but the reported unit differs). Current key
  distribution: 49 match / 4 mismatch / 4 unit_mismatch.
- All three CSVs are plain UTF-8 (no BOM). If re-edited in Excel they may be
  re-saved as GBK or with a BOM — the loader should read with `utf-8-sig` and
  files should be re-normalised before freeze.
- Open numeric-comparison policy (drives the tolerance rules, still to be
  fixed): whether rounding (7.01 vs 7.0180) counts as a mismatch, and whether
  numeric fields compare the mean only or `mean ± SD`. The current hand labels
  are strict (rounding and SD differences are marked `mismatch`).

## How this is used

1. **Parser eval (review side).** Run the Parser on `raw/EAT_T1DM_SRMA.pdf` and
   compare its long-table output against `review_ground_truth.csv` (value-level
   accuracy + correct `study / group / field_type` assignment).
2. **Auditor eval (review vs source).** Fill `audit_template.csv` `source_value`
   / `source_quote` from each cited paper, set `expected_label` to
   `Match` / `Mismatch`. Run the Auditor and score its labels against this key.
   `Auditor recall on seeded discrepancies` (Proposal §6) is measured here.

## Provenance / caveats

All review-side values were transcribed by reading the rendered PDF pages
(Table 1 on p5; Fig 2/4 thickness forest plots on p5/p6; Fig 3 volume forest
plot on p6; DOIs from the reference list on p9). **These transcriptions are
pending human verification against the PDF before the benchmark is frozen.**
