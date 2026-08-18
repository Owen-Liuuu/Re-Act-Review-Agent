# Eval datasets

Three on-disk folders. Logical ids inside frozen JSON are unchanged.

| Folder | Domain | Status |
|---|---|---|
| `eval/benchmark_1` | EAT/T1DM (Sooragonda 2025) | frozen answer key |
| `eval/benchmark_2` | melanoma checkpoint inhibitors (Karlsson & Saleh 2017) | frozen; logical id `melanoma_checkpoint_2017` |
| `eval/benchmark_3` | ESCC (doc05) | source-audit gold only; forest plots wait for OCR; not frozen |

Review Extraction (lens → localize → v3 capture → forest OCR → origin) against `eval/benchmark_3/review_ground_truth.csv`:

```powershell
python eval/run_review_extraction.py --config configs/config.local.yaml
```

Reports land in `eval/benchmark_3/output/` (gitignored). This is not the frozen table-capture A/B gate.

Historical run caches stay at `output/baselines/melanoma_checkpoint_2017/`.
