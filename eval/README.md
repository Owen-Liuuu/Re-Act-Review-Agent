# Eval datasets

Three on-disk folders. Logical ids inside frozen JSON are unchanged.

| Folder | Domain | Status |
|---|---|---|
| `eval/benchmark_1` | EAT/T1DM (Sooragonda 2025) | frozen answer key |
| `eval/benchmark_2` | melanoma checkpoint inhibitors (Karlsson & Saleh 2017) | frozen; logical id `melanoma_checkpoint_2017` |
| `eval/benchmark_3` | ESCC (doc05) | review extraction measured (below); source-audit answer key not yet filled; not frozen |

Review Extraction (lens → localize → v3 capture → forest OCR → origin) against `eval/benchmark_3/review_ground_truth.csv`:

```powershell
python eval/run_review_extraction.py --config configs/config.local.yaml
```

Reports land in `eval/benchmark_3/output/` (gitignored). This is not the frozen table-capture A/B gate.

Last measured — **single live run, 2026-08-18, not frozen** (`output/review_extraction.json`).
Live extraction varies between runs (see `docs/known-limitations.md` L13), so this is one
observation, not a regression baseline, and must not be re-run to pick a better number.

| Slice | recall | precision | value accuracy / gold |
|---|---|---|---|
| Table 1 (`table_text`) | 100.0% (18/18) | 100% | 94.4% (17/18) |
| forest plots (`figure_ocr`) | 93.2% (41/44) | 100% | 93.2% (41/44) |
| all gold cells | 95.2% (59/62) | 100% | 93.5% (58/62) |

Localize recall 100% (5/5 displays). Integrity: `detected_error 3`, `released_wrong 0`,
`fabricated 0` — the three missing forest cells were refused by a checksum, not misread.

**This is extraction accuracy, not audit accuracy.** It measures whether the review's own
tables were read correctly, and shares no denominator with the label accuracy reported for
`benchmark_1` / `benchmark_2`. The audit half of `benchmark_3` has no score at all: see
`benchmark_3/README.md`.

Historical run caches stay at `output/baselines/melanoma_checkpoint_2017/`.
