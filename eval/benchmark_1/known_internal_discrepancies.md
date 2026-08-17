# Internal-consistency axis (review vs itself)

Inconsistencies found WITHIN the Sooragonda 2025 review — Table 1 versus the
forest plots (Fig 2/3/4) and the reference list — as opposed to review-vs-source
discrepancies (the main axis, in `audit_template.csv`). This axis needs only the
review PDF and catches a different error class: the review contradicting itself.

The structured answer key for this axis is **`internal_consistency.csv`**
(rows IC01–IC04). This file is the human-readable narrative behind it. Every
row is flagged `needs_human_review` and attributed to `review_author` where the
review made the error.

## IC01 / IC02 — Keles 2016 EAT thickness: ~10x, author unit-conversion error (HIGH)

- Table 1 `EFT/ EAT`: T1DM **0.7 (0.6–0.9)**, Control **0.6 (0.5–0.7)**.
- Fig 2 / Fig 4 (forest): T1DM **7.33 ± 2.30**, Control **6.00 ± 1.55** (mm).
- Root cause: the source paper reports EFT in **cm** (0.7 cm). The review copied
  `0.7` into Table 1's `EFT/ EAT` column, which for every other study is in
  **mm**, without converting (0.7 cm ≈ 7.0 mm — consistent with the forest's
  7.33). This is an **author unit-conversion error**.
- Cross-referenced on the main axis as `A022` / `A025` (`unit_mismatch`).

## IC03 — Table-1 citation markers do not match the reference list (MEDIUM)

The in-table bracket numbers disagree with the numbered reference list (p9):

| Study (Table 1) | in-table marker | actual ref no. | DOI |
|---|---|---|---|
| Aslan 2015 | [17] | 22 | 10.1111/echo.12960 |
| Iacobellis 2014 | [22] | 27 | 10.1016/j.numecd.2013.11.001 |
| Yazici 2011 | [18] | 23 | 10.1007/s12020-011-9478-x |
| Colom 2018 | [19] | 24 | 10.1186/s12933-018-0794-9 |
| ElBaky 2023 | [20] | 25 | 10.5114/polp.2023.133530 |

(A `citation correctness` signal — Proposal §6.)

## IC04 — ElBaky 2023 thickness: rounding only (LOW, negative control)

- Table 1 T1DM **7.01 ± 1.85** vs forest **7.02 ± 1.86** (<0.2%). Within the 3%
  tolerance — should **not** be flagged. Kept as a negative control.

## Resolved (NOT inconsistencies)

- **Svanteson 2019 volume N (previously flagged):** Table 1 `N` = 148 vs Fig 3
  volume `Total` = 88. Resolved — the source reports T1DM n=88 + control n=60 =
  148 (see `audit_template.csv` A008). The forest `Total` is just the T1DM arm.
  Subgroup vs total, not an error.
- **Svanteson 2019 volume value:** Table 1 T1DM 52.3 (median, IQR) vs forest
  51.30 (mean). Different statistics (median vs mean), expected to differ
  slightly — not an error.
