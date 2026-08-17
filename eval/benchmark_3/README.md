# benchmarks_1 — 人工对答案键（未冻结）

种子运行：`output/runs/d00e2cedbc79`（2026-08-16，`doc05.pdf`，18 条 Table 1 声明，3 篇源论文）。
系统抄表结果只作**草稿**。答案键以你对照 PDF 后的填写为准，填完前不要当回归基线。

先填这两个表（UTF-8，不要用 Excel 另存成 GBK）：

1. `studies_worksheet.csv` — 三篇源论文能不能拿到、DOI/PMID、本地 PDF 路径
2. `gold_worksheet.csv` — 每一格：综述抄得对不对、源论文真值、期望判决

列含义和填法见各 CSV 表头下一行的注释列（`#` 开头的说明行已去掉，说明在下面）。

## 你要填的列

### studies_worksheet.csv

| 列 | 谁填 | 含义 |
|---|---|---|
| study_id | 已填 | 文献别名，不要改 |
| review_citation | 已填 | 综述参考文献原文 |
| printed_doi | 你填 | 参考文献里**印出来的** DOI，没有就空。不要猜 Frontiers |
| printed_pmid | 你填 | 参考文献里印出来的 PMID，没有就空 |
| source_pdf | 你填 | 本地全文 PDF 相对路径，如 `raw/sources/li_2015.pdf`；拿不到就空 |
| access | 你填 | `full_text` / `abstract_only` / `unavailable` |
| notes | 你填 | 任意 |

### gold_worksheet.csv

灰色列（`seed_*`）来自那次运行，**不要当标准答案**。只在抄表确实错了时改 `review_value_corrected`。

| 列 | 谁填 | 允许值 |
|---|---|---|
| review_ok | 你填 | `Y` = 和综述表格一致；`N` = 抄错了，同时改 `review_value_corrected` |
| review_value_corrected | 抄错才填 | 综述格子的正确原文 |
| source_value | 你填 | 源论文里对应的值，原文抄；源论文没有就空 |
| source_quote | 你填 | 能支撑该值的连续原文 |
| source_location | 你填 | 如 `Table 1; Methods; Abstract` |
| expected_label | 你填 | `match` / `mismatch` / `unit_mismatch` / `not_comparable` / `source_unavailable` |
| expected_match_mode | 你填 | `numeric` / `semantic` / `structured` / `skip` |
| notes | 你填 | 例如「综述是 PSM 后 N，摘要是匹配前 N」 |

判决怎么选：

- `match`：综述与源论文说的是同一件事（数字在容差内，或语义等价）
- `mismatch`：两边都有值，但不是同一件事（最常见：匹配前 vs 匹配后人数）
- `not_comparable`：源论文根本不报这个量（发表年、国家写在单位地址里等）
- `source_unavailable`：这篇论文这次拿不到全文/摘要，无法对

## 这次运行里已经能看出、但必须由你拍板的点

- Li J / Li K 的 `N MIE` / `N OE`：综述写的是 **(matched)**，系统从摘要抽到的是匹配前人数（89/318、358/111）。若源论文正文另有 PSM 后人数，以正文为准。
- Capovilla：这次没拿到 Frontiers 全文；报告里出现的 `10.1093/dote/doad052.248` 不是参考文献里的 `13:1104109`，不要写进 printed_doi。
- Table 2（汇总 OR / I² / GRADE）没有进这 18 条。若要一并做基准，在 `gold_worksheet.csv` 末尾按同样列加行。

填完告诉我，再拆成 `review_ground_truth.csv` + `audit_template.csv` 并冻结。
