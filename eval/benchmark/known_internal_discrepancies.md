# Known internal inconsistencies (review vs its own figures)

These are inconsistencies found WITHIN the Sooragonda 2025 review itself —
Table 1 versus the forest plots (Fig 2/3/4). They are distinct from
review-vs-source discrepancies (which need the source papers). They make good
"naturally-occurring" targets for a consistency-checking Auditor and for
testing the Parser's cross-location alignment.

**All items below are pending human verification against the PDF.**

## D1 — Keles 2016 EAT thickness: ~10x mismatch (HIGH signal)

- Table 1 `EFT/ EAT`: T1DM **0.7 (0.6–0.9)**, Control **0.6 (0.5–0.7)**.
- Fig 2 / Fig 4 (forest): T1DM **7.33 ± 2.30**, Control **6.00 ± 1.55** (mm).
- ~10x apart. Most likely a unit slip (cm in the table vs mm in the figure) or a
  transcription error in Table 1. Either way, the review is internally
  inconsistent for this study's headline value.

## D2 — Svanteson 2019 volume N mismatch

- Table 1 `N` = **148**.
- Fig 3 (volume forest) `Total` = **88**.
- The volume analysis appears to use a subset (88) while the table reports 148;
  the relationship is not explained.

## D3 — Table-1 citation markers do not match the reference list

The in-table bracket numbers disagree with the numbered reference list (p9):

| Study (Table 1) | in-table marker | actual ref no. | DOI |
|---|---|---|---|
| Aslan 2015 | [17] | 22 | 10.1111/echo.12960 |
| Iacobellis 2014 | [22] | 27 | 10.1016/j.numecd.2013.11.001 |
| Yazici 2011 | [18] | 23 | 10.1007/s12020-011-9478-x |
| Colom 2018 | [19] | 24 | 10.1186/s12933-018-0794-9 |
| ElBaky 2023 | [20] | 25 | 10.5114/polp.2023.133530 |

(This is a `citation correctness` signal — Proposal §6.)

## D4 — CT studies have no T1DM/control split in Table 1

- de Gonzalo-Calvo 2018 and Colom 2018 (both CT) report a single combined row in
  Table 1 (no T1DM vs control columns), yet the review compares T1DM vs control
  elsewhere. Their per-group contribution is ambiguous.

## D5 — Minor rounding (LOW signal, likely acceptable)

- ElBaky 2023 thickness: Table 1 T1DM **7.01 ± 1.85** vs forest **7.02 ± 1.86**.
  Within rounding; expected to be a MATCH, useful as a negative control.
