# ReAct-Review — 已知局限 & 通用化路线

> 当前 MVP **刻意绑定在单一基准**:Sooragonda 2025 的 EAT/T1DM 系统综述。
> 核心链路(3 agent + 确定性审计 + eval)已跑通,当前最佳端到端标签准确率 **84.2%**,召回 **100%**(无漏报)。
> 本文列出**已知边界**(非 bug,是 MVP 的范围选择)与**通用化路线**,供后续(尤其 DKB 阶段)一并处理。

最后更新:2026-07-28 · 对应 commit `4650baa`。

---

## 一、已知局限

### A. 输入层(PDF → 文本)

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L1** | **图片 / 森林图 / 扫描页读不到** —— `get_text()` 只读文字层,无 OCR / 无多模态 | Meta 分析的**效应量常在森林图=图片里 → 全丢**;某研究若只把数据画成图也会漏 | `retrieval/local_pdf.py` `_pdf_text`、`parser/review_parser.py` `_pdf_text` | 接 OCR / 多模态视觉模型读图与扫描页 |
| **L2** | **表格结构被拍平** —— `get_text()` 把表格线性化成一维文字流,列对齐丢失 | 模型分不清"哪个值属于哪一列/队列" → **group 串列**(拿糖尿病组的值当对照组) | 同上 | 版面感知的结构化表格抽取(PyMuPDF table detection / layout parser) |
| **L3** | **`[:50000]` 截断** | 超 5 万字符的后半部分被切;Table 1 通常在前面所以目前没踩到 | `ReviewParser.max_chars` | 定位相关表格区域后再喂,或按需调大 |

### B. 相关性 / 上下文

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L4** | **`research_context` 手填** —— 走 CLI `--context`,默认空串 → 兜底 `"a systematic review"`,**不从 PDF 自动提取** | 上下文泛化时,Tier-2 消歧(如裸 "Patient" → t1dm)会变弱 | `cli.py` `_run_main`、`tools/normalize.py` | 开头加轻量 LLM,从标题/摘要抽研究主题/PICO 当 context |
| **L5** | **无相关性过滤** —— Stage-2 抽**主表所有测量列**(体重/身高/血压…),只挡标识列(A1),不按研究问题过滤 | 无关字段也进审计 | `parser/review_parser.py` `_STAGE2` + `_postprocess` | **需产品决策:审全部 vs 审聚焦**;若聚焦则用 `research_context` 在 Stage-2 限定字段 |

### C. 领域绑定(硬编码,不通用)

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L6** | **Stage-2 + `normalize_group` 写死 `T1DM / Control / all`** | 换领域直接失效:肿瘤 `treatment/placebo`、病例对照 `case/control`、多臂 `arm A/B/C` 都套不进 | `_STAGE2` 提示、`normalize/groups.py` | **队列/分组也是领域知识** → 从 Stage-1 的 `group_handling` / context 动态推;交给 DKB(field_type 之外再管 group) |
| **L7** | **field_type 词表是 EAT/T1DM 种子** | 换领域要重建词表 | `configs/vocabulary.seed.json` | DKB(动态 RAG 知识库替换静态词表) |
| **L8** | **单位拼写靠 Tier-1 硬编码穷举**(cm³/cc/mL、yrs/years…) | 模型每轮可能吐新拼写 → 打不完的地鼠 | `normalize/units.py` | 单位归一下沉到 DKB 语义层,而非无限扩表 |

### D. 抽取可靠性(模型能力)

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L9** | **近值队列 group-confusion = 模型上限** —— glm-4.5-flash 会读叙述句("组间无显著差异")把一组的值复制给另一组;多队列整行抽取重设计**已试过、更差、回退** | 目前**安全降级**为 `missing_source` 交人审(不误告) | `tools/extract_source.py`(守卫) | 更强模型 / **双模型交叉校验**(需第二个 key) |

### E. 评测

| 编号 | 现象 | 影响 | 位置 | 改进方向 |
|---|---|---|---|---|
| **L10** | **只在单一基准验证** —— 全部指标基于一篇综述(EAT/T1DM) | 跨领域/跨综述表现未知 | `eval/benchmark/` | 加第 2、3 个不同领域的标注基准 |

---

## 二、通用化路线(按倡议分组,对应关闭哪些局限)

| 倡议 | 关闭 | 说明 | 依赖 |
|---|---|---|---|
| **① DKB(领域知识即数据)** | L6 L7 L8 | field_type + **group 分类** + 单位等价,做成动态 RAG 知识库 + provisional 写回。**group 通用化在这里一并做** | 已决定的下一支柱 |
| **② 多模态 / OCR 输入** | L1 L2 | 读森林图、图表、扫描页、结构化表格 | 视觉模型 / OCR |
| **③ 双模型抽取** | L9 | 两模型交叉校验源值,治近值队列混淆 + 抽取漂移 | 第二个 LLM key |
| **④ 自动 research_context** | L4 | 从综述标题/摘要抽主题/PICO | 轻量 LLM 一次 |
| **⑤ 相关性过滤(需决策)** | L5 | 审全部 vs 审聚焦;聚焦则 context 驱动字段过滤 | 产品决策 |
| **⑥ 多领域评测集** | L10 | 第 2/3 个标注基准 | 人工标注 |
| **⑦ 结论轴(P4)** | L1(部分) | 效应量 / 森林图 / 量词声明的审计;需临床输入(Chester) | 多模态 + 临床阈值 |

**优先级建议**:① DKB(已排下一步,顺带 group 通用化)→ ③ 双模型(治 L9)→ ⑥ 多领域集(证明通用)。②多模态 & ⑦结论轴是更大的独立工程。

---

## 三、这一轮已解决(对照,别重复踩)

控制字符 `\x01`→`±`(L2 相关的 JSON 崩溃根因)· 单位逐分量归一(部分 L8)· study_id 连字符匹配 · 标识列泄漏(部分 L5)· 循环 import · 检索结果双标签(`source_access_failed` / `missing_source`)。详见 commit `f932e1d`→`4650baa`。
