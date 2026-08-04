# ReAct-Review — 已知局限 & 通用化路线

> 当前 MVP 有两个冻结检查点:EAT/T1DM 主基准与 melanoma 跨领域检查点。
> EAT 确定性 replay 为 **89.47%** 标签准确率、严格差异召回 **80%**、静默放行 **0**(Phase 7 后逐项不变)。
> melanoma 在 Phase 7 的新契约下为 **80.0%** 标签准确率、精确率/召回率/F1 **100%**、静默放行 **0**(n=15,`docs/baselines/melanoma_phase7_metrics.json`)。
> **跨领域准确率门槛仍未通过,也不以其通过为验收目标**:15 行里一行就值 6.7 个百分点,这个百分数不构成跨领域准确性的证明。
> 本文列出**已知边界**(非 bug,是 MVP 的范围选择)与**通用化路线**,供后续(尤其 DKB 阶段)一并处理。

最后更新:2026-08-04 · Phase 6E 验收基于 commit `4742517`;Phase 7 处置见 `docs/deferred/phase6b-melanoma-audit.md` 的 "Phase 7 disposition" 一节。

---

## 一、已知局限

### A. 输入层(PDF → 文本)

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L1** | **图片 / 森林图 / 扫描页读不到** —— `get_text()` 只读文字层,无 OCR / 无多模态 | Meta 分析的**效应量常在森林图=图片里 → 全丢**;某研究若只把数据画成图也会漏 | `retrieval/local_pdf.py` `_pdf_text`、`parser/review_parser.py` `_pdf_text` | 接 OCR / 多模态视觉模型读图与扫描页 |
| **L2** | **表格结构被拍平** —— `get_text()` 把表格线性化成一维文字流,列对齐丢失 | 模型分不清"哪个值属于哪一列/队列" → **group 串列**(拿糖尿病组的值当对照组) | 同上 | 版面感知的结构化表格抽取(PyMuPDF table detection / layout parser) |
| **L3** | **`[:50000]` 截断** | 超 5 万字符的后半部分被切;Table 1 通常在前面所以目前没踩到 | `ReviewParser.max_chars` | 定位相关表格区域后再喂,或按需调大 |
| **L11** | **Stage-2 单次吐全表 → 输出 token 上限** —— 9 篇 ≈12k 字符 JSON;`max_tokens<8192` 直接截断→JSON 失败→**0 项**;综述再大(几十篇)连 8192 也会爆 | parser 整段失败 | `parser/review_parser.py` `_STAGE2` | **运行时 `max_tokens≥8192` 必需**;根治=Stage-2 按研究/分节**分块抽取**(输出有界,不受综述规模限制) |
| **L12** | **子组样本量 `subgroup_n` 未抽** —— parser 把总 N 复读进两队列,没读每队列的 50/50 分组数 | 15 个 subgroup_n 全漏(占 parser 漏项一半) | `parser/review_parser.py` Stage-2 | 提示 Stage-2 显式抽每队列 N;或和 sample_size 一起做"研究级 vs 队列级"scope 建模 |

### B. 相关性 / 上下文

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L4** | **`research_context` 手填** —— 走 CLI `--context`,默认空串 → 兜底 `"a systematic review"`,**不从 PDF 自动提取** | 上下文泛化时,Tier-2 消歧(如裸 "Patient" → t1dm)会变弱 | `cli.py` `_run_main`、`tools/normalize.py` | 开头加轻量 LLM,从标题/摘要抽研究主题/PICO 当 context |
| **L5** | **无相关性过滤** —— Stage-2 抽**主表所有测量列**(体重/身高/血压…),只挡标识列(A1),不按研究问题过滤 | 无关字段也进审计 | `parser/review_parser.py` `_STAGE2` + `_postprocess` | **需产品决策:审全部 vs 审聚焦**;若聚焦则用 `research_context` 在 Stage-2 限定字段 |

### C. 领域绑定(硬编码,不通用)

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L6** | ~~队列跨文献别名与多臂锚定不充分~~ → **Phase 7 已大部分关闭**:请求显式携带 arm/comparison pair,模型枚举、确定性代码做全局一对一指派,打平即拒绝(`wrong_target_accepted` 3→1) | 剩余:一条臂标签虽指派正确,但语义上仍不被接受为等价 | `normalize/cohorts.py`、`tools/target_assignment.py` | 剩余部分与 L9 合并;跨文献别名仍待 DKB |
| **L7** | **DKB 的 bootstrap ontology 仍以现有基准概念为主** | 新领域虽可生成 provisional concept,但覆盖与稳定性尚未由更多领域证明 | `configs/knowledge.seed.json`、`dkb/` | 扩充人工批准的领域包；保持 provisional→验证→批准的治理链 |
| **L8** | **单位拼写靠 Tier-1 硬编码穷举**(cm³/cc/mL、yrs/years…) | 模型每轮可能吐新拼写 → 打不完的地鼠 | `normalize/units.py` | 单位归一下沉到 DKB 语义层,而非无限扩表 |

### D. 抽取可靠性(模型能力)

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L9** | **近值与多臂 target drift** —— Phase 7 后模型仍会选错,但**选错不再被接受**:枚举项必须各自带原文引文,指派唯一才采纳 | 代价转为能力损失:melanoma 有 2 行因模型改写引文被证据守卫拒绝(`missing_source`),而非错值入库 | `tools/target_assignment.py`、`tools/extract_source.py` | 双模型交叉校验仍是治本方向;引文改写可考虑要求模型给出字符区间 |
| **L16** | **模型会改写自己的引文** —— 把论文缩写的 `95% CI` 拼成 `95% confidence interval [CI]`,引文因此不再是原文连续子串 | 守卫正确拒绝(安全),但正确的值也一并丢失 | `tools/target_assignment.py`、`normalize/anchors.py` | 值的**数字序列**已放宽为可接受措辞规整;引文本身仍要求逐字,不打算放宽 |

### E. 评测

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L10** | **仍没有跨领域准确率证明** —— Phase 7 修完归档的多臂/CI/confidence-level 问题后,melanoma 为 80.0%(n=15),但 15 行的 Wilson 区间过宽,**门槛既未通过、也不以通过为目标** | 可以证明路径可达、缺陷可修、失败可见;不能宣称系统在肿瘤领域准确 | `eval/benchmarks/melanoma_checkpoint_2017/`、`docs/baselines/melanoma_phase7_metrics.json` | 增加第 3 个独立领域基准,行数足以支撑区间;必须由专家确认答案键 |
| **L13** | **实时 LLM 抽取不能充当确定性代码的回归基线** —— 同一评估代码与输入在两次 live 运行中曾把 Iacobellis 总人数分别抽成 30 和 15；Phase 6-0e 的独立 live 又比冻结 replay 多漏 2 行 | 把 live 波动混入代码回归会误判修复或回归，也会诱导“重跑到绿色” | `tools/extraction_cache.py`、`eval/run_full_accuracy.py` | 确定性回归使用版本化 raw-response replay；live 运行独立报告方差，禁止覆盖旧缓存或用重复运行挑选最好结果 |

### F. 语义等价(Phase 4B 新增)

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L14** | **模型自报的 confidence 无信息量** —— 2026-08-03 实测 glm-4.5-flash 在 5 个案例(含判断方向相反的对抗例)上**一律返回 1.0** | DKB 字段解析已在 Phase 5A-3 停止用 confidence 放行，改为跨 seed 稳定性 + 确定性自我契约；语义比较仍保留 `min_confidence=0.70`，承重的仍是数值不漂移 / 极性 / 引文锚定 | `dkb/resolver.py`、`dkb/verify.py`、`audit/semantic_control.py` | DKB 路径已缓解；语义路径仍需继续打印阈值敏感度并评估是否彻底移除 confidence 闸门 |
| **L15** | **关系方向:自相矛盾已能拦,自洽却错的拦不住** —— Phase 7 让 verdict 同时给出 `more_specific_side`,与 relation 互斥即判 `NOT_COMPARABLE + relation_direction`(melanoma 少了最后一个假阳性);但 MA003 给出的是**自洽而与事实相反**的方向,确定性检查无从反驳,4 条 semantic 行仍有 3 条与 overlay 的 relation 不一致 | 已证明矛盾可见,仍未证明 semantic 判定准确 | `audit/semantic_control.py`、`schemas/semantic.py` | 跨 seed 结构一致性(Phase 5A 方法)是下一个可用手段,需多次采样,属框架级选择 |
| **L17** | **Phase 6B 的 relation 期望与 prompt 定义相反** —— 答案键把"review 更具体"记成 `review_broader`,而 prompt 定义 `review_broader` 为 review 更不具体;同形的 MA003 又记成 `same` | 只影响诊断字段,不影响标签(两个 broader 方向同出口) | `eval/benchmarks/melanoma_checkpoint_2017/phase7_semantic_overlay.csv` | 已用 Phase 7 overlay 在**不改冻结答案键**的前提下重述四条 semantic 期望 |

---

## 二、通用化路线(按倡议分组,对应关闭哪些局限)

| 倡议 | 关闭 | 说明 | 依赖 |
|---|---|---|---|
| **① DKB(领域知识即数据)** | L6 L7 L8 | field_type + **group 分类** + 单位等价,做成动态 RAG 知识库 + provisional 写回。**group 通用化在这里一并做** | 已决定的下一支柱 |
| **② 多模态 / OCR 输入** | L1 L2 | 读森林图、图表、扫描页、结构化表格 | 视觉模型 / OCR |
| **③ 双模型抽取** | L9 | 两模型交叉校验源值,治近值队列混淆 + 抽取漂移 | 第二个 LLM key |
| **④ 自动 research_context** | L4 | 从综述标题/摘要抽主题/PICO | 轻量 LLM 一次 |
| **⑤ 相关性过滤(需决策)** | L5 | 审全部 vs 审聚焦;聚焦则 context 驱动字段过滤 | 产品决策 |
| **⑥ 多领域评测集** | L10 L15 | melanoma 已作为第 2 个冻结检查点；修复其延期问题后再加第 3 个领域 | 人工标注 + 专家确认 |
| **⑦ 结论轴(P4)** | L1(部分) | 效应量 / 森林图 / 量词声明的审计;需临床输入(Chester) | 多模态 + 临床阈值 |

**优先级建议**:① DKB(已排下一步,顺带 group 通用化)→ ③ 双模型(治 L9)→ ⑥ 多领域集(证明通用)。②多模态 & ⑦结论轴是更大的独立工程。

---

## 三、这一轮已解决(对照,别重复踩)

控制字符 `\x01`→`±` · 单位逐分量归一 · study_id 连字符匹配 · 标识列泄漏 · 循环 import · 检索结果双标签(`source_access_failed` / `missing_source`) · 研究级字段去重 · **综述原词驱动的 cohort registry** · **复合数值逐分量消费** · **semantic 事后控制与缓存** · **DKB/checklist 人审治理** · **source extraction live/record/replay** · **Evidence Package 先保存、HTML 后渲染**。Phase 6B 未通过的准确率问题不列为“已解决”,统一保存在延期档案。

## 四、评测能力(现有脚本)
现有脚本均可输出 JSON + 英文 HTML 报告(打印即 PDF):
- **审计核心**:`eval/run_benchmark.py`
- **EAT 全流程冻结 replay**:`eval/run_full_accuracy.py --extraction replay` → label 89.47% / strict precision-recall-F1 80% / silent release 0 / review visibility 100%
- **melanoma 跨领域冻结 replay**:15 行覆盖 4 semantic / 4 numeric / 7 structured；label 60%,安全可见性 100%,准确率问题延期
- **Parser 准确率**:`eval/run_parser_accuracy.py --html --out`
- **单篇审计报告**:`react-review run` 先保存 package 再渲染 HTML；`react-review report` 可仅凭已保存 package 重建报告
- **确定性回归**使用 extraction/semantic replay；live LLM 运行只报告方差,不作为代码门禁
