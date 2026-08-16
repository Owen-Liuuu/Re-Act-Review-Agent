# 编号体系索引

这个项目里 `v1` / `v2` / `v3` 出现在**六套互不相干的编号体系**里。同一个数字在不同体系之间没有任何关系——`table_capture_v1` 和 `semantic_v1` 只是碰巧都叫 v1。

这一页只回答一个问题：**你看到的那个 v 号属于哪套体系，现在生产用的是哪个。**

---

## 先理解一条规则

> **v 号不表示「新版本更好」，而表示「这是另一份被冻结的东西，旧的那份必须继续可用」。**

原因在机制里：几乎所有 v 号都进了**缓存键**或**哈希登记表**。

- 抽取录音的键 = 模型 id + `prompt_version` + **提示词全文的 SHA-256**（`tools/extraction_cache.py:19-30`）
- 表格转录契约直接钉住渲染后的哈希（`configs/prompt_contracts/table_capture_v1.json`）

所以把提示词里一个字改掉，旧录音就全部变成 cache miss，而症状看起来像「录音丢了」，不像「提示词被改了」。项目因此规定**不改，只新增**——`docs/acceptance/gate_versions.md`：

```
1. A version file is never edited after it is written.
2. A correction is a new version naming what it supersedes and why.
3. Results are attributed to the gate version and hash that produced them.
4. Withdrawn results stay published as withdrawn, with the defect named.
```

由此产生两个反直觉的后果，看代码时最容易卡在这里：

- **名字里带 `legacy` 的可能是当前默认。** `legacy_v3` 是最早那份抽取契约，但它就是 `DEFAULT_PROFILE`。
- **数字大不代表更好。** `table_capture_v2` 比 v1 新，但实测更差，所以生产留在 v1。

---

## 一 · 提示词 profile（问模型的那句话）

这是四套**各自独立**的编号，分别管四种问题。

| 管什么 | 有哪些 | 生产用哪个 | 定义在 |
|---|---|---|---|
| 单条抽取 | `legacy_v3`、`targeted_v4`、`targeted_v6` | `targeted_v4`（臂身份）/ `targeted_v6`（v9 起） | `tools/extraction_profile.py` |
| 批量抽取 | `targeted_v5_batch` | 是，数值走这条 | `tools/batch_prompt.py` |
| 表格转录 | `table_capture_v1`、`table_capture_v2` | **`table_capture_v1`** | `parser/table_capture_contract.py:19` |
| 语义判定 | `semantic_v1`、`semantic_v2_specificity` | `semantic_v2_specificity` | `tools/semantic_compare.py:27-33` |

三点需要留意：

**代码默认 ≠ 生产实际用的。** 单条抽取的 `DEFAULT_PROFILE` 是 `legacy_v3`，语义判定的 `DEFAULT_SEMANTIC_PROFILE` 是 `semantic_v1`——但 run profile 会把它们覆盖成更新的那个。代码默认值代表「没人指定时的兜底」，不代表生产在跑什么。要知道生产在跑什么，看 run profile 文件。

**`targeted_v6` 与 `targeted_v4` 只差队列范例。** v4 用本领域的糖尿病队列举例，v6 改成 `<cohort A>` / `<cohort B>` 占位符，规则本体一字不差。`tests/tools/test_extraction_profile.py` 里有一条不变量测试断言两者「只在范例处不同、别处全同」。

**`table_capture_v2` 在生产里到不了。** 没有任何 run profile 指向它；唯一的切换开关 `TABLE_CAPTURE_PROMPT_PROFILE` 环境变量只被 `tests/parser/test_table_capture_live.py` 读取，`src/` 里没有代码看它。它只为跑 A/B 实验存在。

---

## 二 · run profile 的 `schema_version`——文件**格式**版本

`SUPPORTED_CONTRACT_VERSIONS = (1, 2, 3, 4)`，当前写新文件用 `4`（`run_profile.py:50-51`）。

- **v1**：只能写一个 `extraction_profile`，所有 claim 用同一个提示词
- **v2 起**：改为按 claim 种类分路由（`extraction_routes`），因为一次运行可以合理地「数值走批量、臂身份走单条」，而声明成单一 profile 的混合运行是没人能解释的录音
- v1 到 v4 都还能加载

---

## 三 · run profile 的 `profile_id`——**组合**版本

`configs/run_profiles/` 下的文件名：`legacy`、`phase8`、`phase8_batch`、`phase8_batch_v2` … `phase8_batch_v9`。

一份 run profile 就是一张**组合清单**：用哪个抽取提示词 + 哪个语义提示词 + 哪个表格转录提示词 + 哪张容差表 + 哪个证据闸门 + 哪个聚合策略。

**当前生产：`phase8_batch_v9.json`。**

这里有个最容易混的地方——**同一个文件里有两个不相干的数字**：

```json
"schema_version": 4,          // 体系二：文件格式
"profile_id": "phase8_batch_v9",   // 体系三：组合版本
```

另外注意：`phase8_batch_v6.json`（run profile）和 `targeted_v6`（抽取 profile）**完全无关**，只是两套体系各自数到了 6。

---

## 四 · 确定性评估器——三层同时带号

这一套最容易误读，因为看着像同一个东西的三个版本，其实是三个层次：

| 层次 | 例子 | 是什么 |
|---|---|---|
| 策略 | `safe_sum_v5`、`evidence_adequacy_v1` | 规则是什么 |
| 评估器 | `safe_aggregation_1.8.2`、`evidence_adequacy_1.0.0` | 实现代码的语义化版本 |
| 登记表 | `registry_v9`、`registry_v1` | 哪些组合被批准了 |

评估器用的是 `MAJOR.MINOR.PATCH`，而且升级判据是预先注册的：`1.8.1 → 1.8.2` 只算 PATCH，因为 26 个冻结行为向量一个都没移动（`gate_versions.md` 有完整说明）。

---

## 五 · 门（gate）版本

`configs/gates/` 下有三套各自编号的门，加上 `eval/` 下的 A/B 门。这是「用什么标准判定通过」的版本。

| 门 | 文件 | 管什么 |
|---|---|---|
| `d1_batch` | `d1_batch_v1/v2/v3.json` | 批量抽取路径的验收 |
| `cross_domain` | `cross_domain_v1/v2.json` | **跨领域准确率**——目前**未通过**，且从未被声明通过 |
| TableCapture A/B | `eval/table_capture_ab_v1.json` + `_result.json` | v1 对 v2 的配对比较 |

`cross_domain` 这一套值得单独知道：melanoma 的两份成绩单都拒绝声称它通过了。`melanoma_phase8_metrics.json` 写的是「the cross-domain accuracy gate remains unpassed and was not a target」；`melanoma_phase7_metrics.json` 写的是「The melanoma accuracy gate is NOT declared passed」，并给出理由——15 行样本里一行就值 6.7 个百分点，所以 80% 不能被当作跨领域准确率来引用。这是有意的诚实标注，不是遗漏。

`d1_batch` 从 v2 到 v3 的原因值得一读：v2 自称是预注册，但实际是在看到答案之后写的。v3 没有偷偷改 v2，而是新写一份，在文件自己的口径里把「这是回溯性诊断，不是预注册」说明白。

---

## 六 · Phase 编号——这个不是版本号

Phase 6 / 6B / 6E / 7 / 8 是**项目开发阶段**，不是任何东西的版本。

它会和上面的体系交叉引用：`legacy_v3` 是 Phase 6 录的，`targeted_v4` 是 Phase 7 的，`phase8_batch_*` 是 Phase 8 的。所以「Phase 6 的录音在 legacy_v3 下」这种说法里有两套编号。

---

## 你实际只需要记住三个

跑一次审计时真正会影响你的：

1. **`configs/run_profiles/phase8_batch_v9.json`** —— 传给 `--profile` 的那个文件，其余一切由它决定
2. **`targeted_v6`** —— v9 内部选的中立抽取提示词，不用手写
3. **`table_capture_v1`** —— 仍带本领域范例、且**故意**保留的那个

第 3 条会被问到，所以理由写在这里：它的领域中立候选 `table_capture_v2` 已经做过配对 A/B（`docs/acceptance/table_capture_ab_v1.md`），结果是 `regressed`——v2 在 EAT 上把 35 个视觉合并的续行单元格全填满，单元格准确率从 0.9085 掉到 0.6863，幻觉从 2 涨到 35；在 melanoma 上产生 ragged row 导致 schema 检查失败。同一次实验还发现 v1 提示词里那些 EAT/T1DM 范例词**并没有出现在 melanoma 的输出里**。

所以这个项目对提示词领域化的处理原则不是「中立措辞一定更好」，而是：**在中立措辞尚未被证明有代价的地方采用它，在已被证明有代价的地方保留原样，并且两种情况都留下可查的实测依据。**

其余体系（二到六）是内部治理用的，只在读 `gate_versions.md` 或写新 profile 时才会碰到。
