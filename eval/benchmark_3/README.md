# benchmark_3 — ESCC（doc05）源论文核对金标（未冻结）

本项目只核：**综述写到某一篇源论文头上的数，源论文里是不是这个数。**

不核正文叙述、不核 Table 2 合并 OR、不核 GRADE、不核森林图菱形。那些是综述自己的统计结论，源论文里没有。

## 要对源论文的格子

| 来源 | 内容 | 抽取 |
|---|---|---|
| Table 1 | 三篇的特征与匹配后 N | 现在就能测 |
| forest_1–forest_4 | 分研究 Events / Total | 等 OCR 出网格 |

图/表 **有印刷编号就用印刷编号**（本 PDF 的 Table 1）；**没有编号**时，按 evidence_chain 森林图出现顺序记 `forest_1`…`forest_4`。本文四张森林图 caption 没有 "Figure 3.3.1" 这类印刷号（3.3.1 是章节，不是图号），所以金标 `source_location` 用序数 id，不用章节号。

金标：`review_ground_truth.csv`（62 行，R001–R062）。Table 1 是 `capture=table_text`（18 行）；森林图 Events/Total 是 `capture=figure_ocr`（44 行）。没有分研究 OR、没有 Weight。

- `studies_worksheet.csv` — 三篇的引用串，用来把解析器的 `Li J et al. 2015` 对上金标 `li j_2015`
- `raw/doc05.pdf` — 综述
- `output/` — 本机评测报告（gitignore，勿提交）

`internal_consistency.csv` 记录综述自己打架（匹配/未匹配分母），**不是本项目主任务**。

## 怎么测 Review Extraction

需要本机 `configs/config.local.yaml`（LLM）。默认读这份金标，报告写到 `output/`：

```powershell
python eval/run_review_extraction.py --config configs/config.local.yaml
```

产出：

- `eval/benchmark_3/output/review_extraction.html`
- `eval/benchmark_3/output/review_extraction.json`
- `eval/benchmark_3/output/parser_items.json`
- `eval/benchmark_3/output/journal/<run_id>/` — lens / localize / origin 逐步件

看 **Table 1 recall** 才是当前诚实分数。Forest recall 在 OCR 还不能读图时会接近 0，这不是 localize 单独的失败。不要用 `eval/run_parser_accuracy.py`：它按 `(study, group, field_type)` 连接，会把四张森林图的 Events 压成一行。

已有一次运行时可以只复评、不打模型：

```powershell
python eval/run_review_extraction.py --items eval/benchmark_3/output/parser_items.json
```

这不是冻结的 TableCapture A/B 门（`eval/table_capture_ab_v1.json`，v1 vs v2 抄格子）。不要把 `table_capture_v3` 加进那份清单来冒充泛化。

## 2b 注意（森林图列头）

列头兜底只接受 `resolved` / `alias`，不要放宽到 `combined`。`Total` / `Overall` 在 cohort `_COMBINED` 里，`resolve("Total")` 得到 `key=all` `status=combined`。森林图列头正是 Events, Total, Events, Total；若 2b 复用 Table 1 这条路径，Total 会被当成明确的合并队列并吃掉臂身份——看起来解析成功、实际把 group 写成 all。

## 源论文填写

`gold_claims.csv` 与 `review_ground_truth.csv` 一行对一行（R001–R062）。`source_value` / `source_quote` / `source_location_in_paper` / `expected_label` 空着的格子由人对照源论文填。上一轮抽取读到的数留在 `seed_source_value`（未匹配 N 等），不要直接当金标。优先填 Events 和匹配后 N。分研究 OR 已去掉。
