# ReAct-Review 导师汇报 PPT 框架

适用场景：约 10-12 分钟主讲，随后讨论。  
核心叙事：**ReAct-Review 不是对 Lit Inspector 的改名，而是把旧项目中“宽而松的流程检查”，收敛成“以原始证据为中心、确定性裁决、可复现评估”的审计系统。当前已完成单领域 MVP 闭环，也明确暴露了下一阶段的通用化瓶颈。**

---

## 一、整场汇报只回答四个问题

1. 为什么从 Lit Inspector 演进到 ReAct-Review？
2. 新系统最关键的架构变化是什么？
3. 当前 MVP 到底完成了什么，效果如何？
4. 哪些还没有完成，下一阶段优先解决什么？

建议只强调四个核心变化：

- 从“检查综述流程”转向“逐条核对综述数据与源论文证据”；
- 把 LLM 限定在语义理解和证据提取，把数值、单位、匹配、裁决留给确定性代码；
- 从整表盲抽取改为 review-driven directed extraction，并保存 evidence package；
- 从静态 vocabulary 演进为可检索、可追溯、可受控写回的 DKB。

---

## 二、主讲页结构

### Slide 1｜标题：ReAct-Review

副标题建议：

> From LLM-assisted checking to evidence-grounded systematic review auditing

本页只讲一句话：

> ReAct-Review 的目标，是把系统综述中的每一个报告值追溯到源论文证据，并用可复现规则判断是否一致，把需要专家判断的少数问题交给人。

建议画面：

- 标题；
- 一条很短的链路：`Review claim → Source evidence → Deterministic verdict → Human review`；
- 不放完整架构图。

时间：30 秒。

---

### Slide 2｜为什么要从 Lit Inspector 演进

标题建议：

> From workflow inspection to source-data audit

核心结论：

> Lit Inspector 证明了“搜索—论文验证—抽取—比较”的完整流程可以跑通；ReAct-Review 则重新收窄问题，把最有研究价值、也最可评估的一段——综述数据与源论文的逐条核对——做深。

建议用“保留 / 改变”两栏，不要把旧项目讲成失败：

| Lit Inspector 中保留的资产 | ReAct-Review 的改变 |
|---|---|
| 多级全文获取、CrossRef 验证 | 从广泛流程检查收敛为 source-data verification |
| 可插拔 LLM backend、重试机制 | 从整表抽取改为按 review claim 定向找证据 |
| 报告生成、模块化 pipeline | 新增 typed data contract、Evidence Package、Judge |
| 初步表格比较能力 | 比较和裁决改成确定性规则，并建立独立 benchmark |

口头说明：

- 旧项目的输出更接近“流程是否可疑”；
- 新项目的输出是“哪一条值、源论文写了什么、在哪里、为什么被标记”；
- 这是研究问题的收敛，不是简单重写。

时间：50 秒。

---

### Slide 3｜问题定义与 MVP 边界

标题建议：

> What exactly are we auditing?

输入：

- 一篇系统综述 PDF；
- 纳入研究及其源论文 PDF。

核心任务：

1. 从综述抽出 `(study, group, timepoint, field_type, value, unit)`；
2. 回到对应源论文，定向抽出 value、unit、quote、location；
3. 按同一 key 对齐；
4. 用容差和单位规则给出 `match / mismatch / unit_mismatch / not_comparable`；
5. 将非 clean match 项交给 human review。

输出：

- JSON Evidence Package；
- 人可核对的 HTML/PDF 报告；
- 每条结果都能回到源论文 quote 和 location。

MVP 范围必须明确：

- 当前绑定于一篇 EAT/T1DM 系统综述；
- 57 个 benchmark claims，9 篇源论文；
- 当前验证的是“数据轴”，不是森林图和综述结论轴。

时间：50 秒。

---

### Slide 4｜新系统架构

标题建议：

> Evidence-centred architecture

主图只用 System Architecture.pdf 上半部分的简化版本：

```text
Review PDF
   ↓
Parser
   ↓
Agent 1: Evidence Collector
   ↓
Evidence Package Store
   ↓
Agent 2: Evidence Auditor
   ↓
Agent 3: Judge / Arbiter
   ├─ Final verification → JSON / human-checkable report
   └─ Low confidence / unresolved → Human review
```

讲图时只强调三点：

- Orchestrator 控制流程，Agent 不能任意改写审计规则；
- Evidence Package 是共享的、持久化的审计记录；
- Judge 不是“再问一次 LLM”，而是把非 clean match 和缺证据项确定性路由给人。

注意：

- 不要把完整 Detailed Flowchart 直接缩到一页，字会不可读；
- 完整架构图放备份页。

时间：70 秒。

---

### Slide 5｜MVP 已经跑通的闭环

标题建议：

> A working, human-checkable audit loop

建议直接截取 Report.pdf 第 1 页上半部分，突出：

- 57 claims；
- 9 source papers；
- 52 match；
- 1 mismatch；
- 4 unit mismatch；
- 5 items flagged for human review。

选择一个具体例子讲：

> Ahmad 2022 的 BMI：综述为 `20.57 ± 1.70`，源论文为 `20.57 ± 1.77`。均值一致，但 SD 相对误差超过 3%，因此被标记为 mismatch。

再选一个单位例子：

> Keles 2016 的 EAT thickness：综述单位为 mm，源论文为 cm；数值表面相同，但单位轴不同，因此独立标记为 unit mismatch。

本页要传达：

> 系统的价值不是输出一个总分，而是把少数问题定位到具体研究、组别、字段和源证据。

时间：70 秒。

---

### Slide 6｜关键变化一：确定性边界

标题建议：

> Let the LLM interpret; let code decide

建议画成三层：

| 层 | 任务 | 实现方式 |
|---|---|---|
| Tier 1：语法归一 | `6.60 ± 0.71`、小数逗号、单位拼写、大小写/空格 | 确定性代码 |
| Tier 2：语义归一与提取 | 字段名/组别语义、源论文中同义表达 | DKB + LLM |
| Audit decision | key join、均值 1% 容差、SD 3% 容差、单位轴、最终标签 | 确定性代码 |

必须讲清楚：

- Tier 1 不使用 LLM，因为它必须便宜、稳定、可复现；
- LLM 只处理死规则覆盖不了的语义问题；
- LLM 不直接决定 match/mismatch；
- 单位不只是字符串清洗，而是独立审计轴。

这一页可用一句总结：

> The innovation is not a longer prompt; it is deciding what the model is not allowed to decide.

时间：70 秒。

---

### Slide 7｜关键变化二：从整表盲抽取到定向证据抽取

标题建议：

> Review-driven directed extraction

建议做 Before / After：

**Before：**

`Source PDF → extract every field → fuzzy align with review`

问题：

- 表格拍平后容易串列；
- 同义字段需要再次猜测；
- 输出过长，容易遇到 token 截断；
- 很难知道一个值为什么被抽出来。

**After：**

`Review claim → canonical concept → query one study/group/field → value + unit + quote + location`

例子：

```text
Query:
study = Ahmad 2022
group = T1DM
field_type = eat_thickness

Return:
6.60 ± 0.71 mm
quote = "Table 2..."
location = Table 2 / printed page 1004
```

核心价值：

- review 与 source 的匹配问题被缩小为一个明确查询；
- 每条审计结果带 provenance；
- 找不到与访问失败被区分为 `missing_source` 与 `source_access_failed`；
- group-confusion 时宁可降级给人，也不制造假 mismatch。

时间：70 秒。

---

### Slide 8｜关键变化三：DKB 解决领域歧义

标题建议：

> Domain knowledge as data, not hard-coded exceptions

用 EAT 案例讲，不要先讲 RAG 术语：

- `EAT` 可能指 thickness，也可能指 volume；
- Echo 更常测 thickness；
- CT / MRI 更常测 volume；
- 只靠静态 synonym 表，`EAT → eat_thickness` 会误分类 CT 论文。

建议展示精简条目：

```yaml
field_type: eat_volume
synonyms: [EAT, EFT, epicardial adipose tissue]
domain: cardiology/imaging
disambiguation:
  modality:
    ct: eat_volume
    echo: eat_thickness
default_unit: cm3
unit_equivalences: [mL, cc]
status: authoritative
provenance:
  source: curated
```

当前已经实现：

- KnowledgeEntry schema；
- synonym + unit + modality 的确定性解析；
- keyword top-k retrieval；
- KB miss 时的 grounded classification agent；
- 新概念以 `provisional` 写回，不会直接成为 authoritative；
- cache key 包含 `raw name + unit + research context`。

尚未完成：

- embedding/vector retrieval；
- provisional 的持久化晋级策略；
- human-confirm / repeated-agreement promotion；
- group knowledge 的完整通用化。

一句话总结：

> DKB 的意义不是“多存一些同义词”，而是让系统的领域判断有依据、可追溯、可修正。

时间：80 秒。

---

### Slide 9｜Agent 与可靠性工程

标题建议：

> Bounded autonomy, explicit recovery

建议分成三个工程组件：

1. **Bounded ReAct runtime**
   - Thought → Action → Observation；
   - 每个 agent 只暴露小工具子集；
   - `max_steps` 防止失控；
   - 坏工具名、坏参数、工具错误都记录为 observation，不让整条 pipeline 崩溃。

2. **Reflection Decider**
   - `accept / retry / escalate`；
   - 检索失败、双模型不一致、低置信度分别处理；
   - 路由规则由 Python 阈值控制。

3. **Evidence Package Store**
   - 每次运行保存完整 JSON；
   - 包含 review items、source items、审计报告、最终验证、处理轨迹；
   - 原子写入，便于复现和回归测试。

汇报口径要准确：

> 共享 bounded runtime、Reflection 和 Store 已实现并有测试；当前生产 MVP 为了降低风险，Collector 使用受控的定向抽取循环，Auditor 与 Judge 仍由确定性 orchestrator 驱动。三个角色完全接入同一 ReAct runtime 是下一步整合，而不是当前已经完成的事实。

时间：65 秒。

---

### Slide 10｜评估体系与当前结果

标题建议：

> Evaluate each failure source separately

不要只放一个“准确率”。建议分三层：

#### 1. 确定性审计核心

- 57/57 hand-labelled audit rows 复现正确；
- 输出分布：52 match / 1 mismatch / 4 unit mismatch；
- 10 个 seeded discrepancies：precision 100%，recall 100%；
- 说明：这证明规则实现与 answer key 一致，不代表 PDF/LLM 端到端已经 100%。

#### 2. Parser

- Ground truth：101 rows；
- Parser 输出：84 rows；
- 对齐 key：76；
- field coverage：75.2%；
- field precision：90.5%；
- aligned value match：84.2%。

#### 3. Collector + audit 端到端

- 57 rows；
- 本次报告 label accuracy：82.5%；
- discrepancy precision：71.4%；
- discrepancy recall：100%；
- F1：83.3%；
- found rate：89.5%；
- value match：82.5%；
- TP=5 / FP=2 / FN=0 / TN=50；
- 文档记录不同 LLM 运行约 82%-84%，说明仍有模型波动。

工程测试：

- 当前本地全量测试：247 passed。

本页的解释重点：

> 召回 100% 表示当前 benchmark 上没有漏掉应标记的问题；71.4% precision 表示仍有 false positives，需要减少人工负担。当前策略是 recall-first，因为审计系统宁可多交给人，也不能悄悄漏掉问题。

时间：90 秒。

---

### Slide 11｜已知局限：MVP 为什么还不能称为通用系统

标题建议：

> Known limits, not hidden failures

只讲最重要的四类：

1. **输入缺口**
   - 扫描页、图片、森林图读不到；
   - 文本抽取会把表格结构拍平。

2. **Parser 可扩展性**
   - Stage 2 当前一次输出整张长表，受 token 上限影响；
   - `subgroup_n` 仍是最大覆盖缺口之一。

3. **领域与上下文**
   - `research_context` 仍需手填；
   - group 仍绑定 T1DM / control / all；
   - 当前 benchmark 只有 EAT/T1DM。

4. **模型可靠性**
   - 相近数值的 group-confusion 仍会发生；
   - 双模型交叉抽取尚未接入当前 MVP。

建议说法：

> 这些限制决定了当前成果应该被称为“single-domain MVP with a measurable audit loop”，而不是通用临床审计产品。

时间：60 秒。

---

### Slide 12｜下一步路线与希望导师给的反馈

标题建议：

> Next: generalise before expanding

建议路线：

1. **DKB-3**
   - 持久化 provisional mapping；
   - human confirm / repeated agreement 晋级；
   - group 也进入 DKB；
   - 记录 KB version，保证复现。

2. **Parser / multimodal**
   - 按研究分块抽取；
   - 结构化表格识别；
   - OCR / 视觉模型读取扫描页与森林图。

3. **Reliability**
   - 双模型抽取与 disagreement routing；
   - 自动提取 PICO / research context。

4. **Evaluation**
   - 增加第 2、3 个不同领域 benchmark；
   - 报告 confidence interval 和按错误类型分解；
   - 再进入结论轴 / 森林图审计。

希望导师反馈的问题建议只问一个：

> 下一阶段是否应优先证明“跨领域可泛化”（DKB + 多领域 benchmark），再投入更大的多模态与结论审计工程？

结束句：

> 当前 MVP 已经证明了一件事：只要把语义理解、证据记录和确定性裁决分开，LLM 可以从一个不可控的抽取 demo，变成一个可评估、可追溯的审计组件。

时间：50 秒。

---

## 三、备份页

### Backup 1｜完整系统架构

- 放 System Architecture.pdf 的完整 Detailed Flowchart；
- 只在导师追问 agent 间如何反馈、re-audit / more evidence 路由时展示。

### Backup 2｜指标定义

- Parser coverage = aligned keys / ground-truth keys；
- Parser precision = aligned keys / parser keys；
- aligned value match 只在 key 对齐后统计；
- discrepancy positive = mismatch 或 unit_mismatch；
- found rate 与 value match rate 是两种不同指标。

### Backup 3｜DKB 完整数据结构

- concept；
- synonyms；
- domain / scope；
- modality disambiguation；
- unit equivalences；
- plausible range；
- provenance；
- authoritative / provisional。

### Backup 4｜确定性容差规则

- mean relative error ≤ 1%；
- 若两边都有 SD，SD relative error ≤ 3%；
- 单位轴独立判断；
- 任一无法比较则 not_comparable / missing-source 路由。

### Backup 5｜测试与可复现性

- 247 个测试通过；
- runtime、reflection、EvidencePackage、DKB、normalize、parser、audit、report 均有测试；
- benchmark 可作为 CI gate；
- Evidence Package 可作为回归 fixture。

### Backup 6｜旧项目资产复用

- full-text retrieval；
- CrossRef / PubMed 等外部验证；
- LLM retry / provider adapters；
- report renderer；
- 说明 ReAct-Review 是基于旧项目经验进行的架构收敛。

---

## 四、演示建议

如果导师允许 60-90 秒演示：

1. 打开审计报告首页；
2. 指出 `52 / 1 / 4 / 5`；
3. 展开 Ahmad BMI mismatch；
4. 展开 Keles unit mismatch；
5. 指向 quote + location；
6. 最后打开 package.json，说明报告不是临时生成的文本，而是由完整证据包确定性渲染。

不要现场演示：

- 实时 LLM 跑完整 57 条；
- 完整 Detailed Flowchart；
- 大段源码；
- prompt 文本。

---

## 五、汇报红线

- 不要把“确定性核心 57/57”说成“端到端 100%”；
- 不要把 Parser 的 84.2% value match 说成端到端 label accuracy；
- 不要说 DKB 已经是向量数据库，目前是 keyword retriever + 可替换接口；
- 不要说 provisional 已能自动晋级，目前 promotion 是 DKB-3；
- 不要说三类 agent 已全部接入同一 ReAct runtime；
- 不要把 `FAIL` 总结为“综述整体错误”，它表示当前规则下存在需要复核的条目；
- 不要回避单一 benchmark；主动说明能提升可信度。

---

## 六、30 秒开场与结束

开场：

> 上一个项目 Lit Inspector 证明了系统综述检查流程可以被自动化，但它仍然更像一个宽泛的 LLM pipeline。这个阶段我把问题收窄成一个可验证的研究问题：综述里报告的每一个数据，能否在源论文中找到，并且能否给出可复现、可追溯的审计结论？ReAct-Review 就是围绕这个问题重新设计的。

结束：

> 当前 MVP 已经跑通了从综述 PDF、源论文证据、确定性比对到人审报告的闭环。下一阶段的关键不再是继续调 prompt，而是让 DKB、Parser 和 benchmark 跨出 EAT/T1DM 单领域。我希望和您确认的是：是否先证明跨领域泛化，再进入森林图和结论审计。
