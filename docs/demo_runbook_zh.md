# 导师演示 · 命令手册

配合 `docs/supervisor_presentation_outline_zh.md`。这份只管**现场怎么跑**。

演示要证明的三件事,按顺序:

1. 管道**会停**,停下来时给的是文件名和理由,不是一个结论;
2. 判决由**确定性代码**做出,模型只负责读和提议;
3. 跨领域**仍然失败**,而且失败是可见、可复现、可归因的 —— 这是导师明确要看的部分。

---

## 0 · 开场前(不占演示时间)

在另一个终端先跑,证明后面所有数字都不是现场调出来的。

```bash
python -m pytest -q
```

预期 `716 passed`。

```bash
python eval/run_full_accuracy.py --extraction replay --extraction-cache output/baselines/phase6_0d_final_extraction_replay.json
```

预期 `71 reused / 0 cache misses`、`label accuracy 89.5%`、`silent releases 0`。
**这一步不调用任何模型**,可以当场断网证明。

---

## 1 · 停在表格捕获门(约 2 分钟等待,先启动再讲话)

```bash
python -m react_review run --pdf "eval/benchmark/raw/EAT_T1DM_SRMA.pdf" --studies eval/benchmark/included_studies.csv --limit 3 --checkpoints key --allow-skip --out output/demo
```

启动后它会依次经过 `review_pdf_loaded` → **`review_table_capture`**。

到达表格捕获门时,屏幕上是**逐字转录的表格**加上 `difficulties`(哪一格读不准、为什么),提示行是:

```
  [C]ontinue  [S]top  [X] drop one  [D]etail  [O]pen artifact  [A]ll (skip remaining checkpoints) >
```

现场按键顺序建议:

| 键 | 目的 | 说给导师听 |
|---|---|---|
| `O` | 打印这一步的 artifact 路径 | "每一步在**问你之前**就已经写盘了,Ctrl-C 也不会丢" |
| `D` | 打印完整 payload | "给你看的不是摘要,是它接下来真正要用的全部内容" |
| `X` | 进入可丢弃列表,选一个表格丢掉 | "你可以在这里把一张读错的表**踢出去**,后面的流程不会再碰它" |
| `S` | 停止 | "这就是它会停的证明" |

按 `S` 之后:

```bash
echo $?            # bash:2      PowerShell 用 $LASTEXITCODE
ls output/demo/*/steps/
ls output/demo/*/package.partial.json
```

要点:**退出码 2 = 人工叫停**(Ctrl-C 是 130),并且已经落了 `package.partial.json` 和逐步 journal。
"跑到一半被叫停"和"跑完"在产物上是分得清的。

> 如果现场网络或 key 出问题:直接 `ls output/runs/` 找一次历史运行,打开它的 `steps/` 目录讲同样的事。门是代码结构,不是这一次运行的运气。

---

## 2 · 让它跑完,打开自动报告

同一条命令重跑,这次在每个门按 `C`(或第一个门按 `A` 一路放行):

```bash
python -m react_review run --pdf "eval/benchmark/raw/EAT_T1DM_SRMA.pdf" --studies eval/benchmark/included_studies.csv --limit 3 --checkpoints key --allow-skip --out output/demo
```

结束后:

```bash
ls output/demo/                      # 找到 run_id
start output/demo/<run_id>/report.html      # macOS/Linux 用 open
```

HTML 里要指的三处:每一行的 **source quote**、**为什么是这个标签**、以及 **Human Review Flag**。
一句话:"报告里没有一个判断是没有出处的。"

---

## 3 · 跨领域:展示失败(零调用,秒出)

这是导师要的核心。两条命令,同一个冻结基准、同一份未被改动的答案键,对比两个契约。

**3.1 Phase 6 的结果(去年那次失败)**

```bash
python eval/run_full_accuracy.py --benchmark eval/benchmarks/melanoma_checkpoint_2017 --extraction replay --extraction-cache output/releases/phase6e_2026-08-04/melanoma_extraction_cache.json --semantic cache-only --semantic-cache output/releases/phase6e_2026-08-04/melanoma_semantic_cache.json
```

预期:`label accuracy 66.7%`、`silent releases 0`。

**3.2 Phase 7 的结果(修完四项归档缺陷之后)**

```bash
python eval/run_full_accuracy.py --benchmark eval/benchmarks/melanoma_checkpoint_2017 --benchmark-profile phase7_profile.json --extraction replay --extraction-cache output/baselines/melanoma_checkpoint_2017/phase7_extraction_cache.json --semantic cache-only --semantic-cache output/baselines/melanoma_checkpoint_2017/phase7_semantic_cache.json --html output/demo/melanoma_phase7.html
```

预期输出里要念出来的几行:

```
audit label accuracy   :  80.0%
  precision            : 100.0%   recall : 100.0%
  silent releases      : 0
  review visibility    : 100.0%
  wrong target accepted: 1   (must be 0)
observed modes         : {'semantic': 4, 'numeric': 6, 'structured': 5}
status                 : fail_unexpected_differences
```

**必须自己说出口的话**(不要等导师问):

> 这里仍然是 **fail**。15 行的基准,一行就值 6.7 个百分点,80% 不构成跨领域准确性的证明,我们也没有把它当作验收目标。变化的是:去年那四个缺陷里,两个已关、一个基本关、一个只关了一半,而且每一处修复都是**确定性代码在判模型的枚举**,不是把 prompt 改顺。

三行仍然失败,直接把原因说清楚:

| 行 | 结果 | 原因 |
|---|---|---|
| MA005 | `not_comparable` + 需复核 | 模型的语义判断**自相矛盾**(说"不同"却又指认某一侧更具体),被拒绝而不是被当成差异记下来 |
| MA009 / MA014 | `missing_source` | 模型把自己的引文**改写**了,不再是原文连续子串,证据守卫拒收 —— 宁可丢值,不要错值 |

---

## 4 · 如果被追问"你怎么知道这不是调出来的"

```bash
git log --oneline -3
python -c "import json;d=json.load(open('docs/baselines/melanoma_phase7_metrics.json',encoding='utf-8'));print(json.dumps(d['private_artifact_sha256'],indent=1))"
```

三句话:

- 原始模型响应**录制一次**、哈希公开,后面所有数字都是同一份录制的 replay;
- 答案键、manifest、PDF 的 SHA-256 由 profile 文件钉住,改了就加载失败;
- 不传 profile 时,每一条 prompt 与 Phase 6 逐字节相同 —— 旧结果照样重现(3.1 就是当场证明)。

---

## 5 · 时间与风险

| 步骤 | 耗时 | 需要 key | 断网可跑 |
|---|---|---|---|
| 0 预检 | ~10 秒 | 否 | 是 |
| 1 停在门上 | ~2 分钟到达 | 是 | 否 |
| 2 跑完 + 报告 | ~2-4 分钟(`--limit 3`) | 是 | 否 |
| 3 跨领域失败 | 秒级 | **否** | **是** |
| 4 溯源 | 秒级 | 否 | 是 |

风险只有一个:步骤 1-2 依赖 GLM 在线。**演示前一天完整彩排一次**,把那次的 `output/demo/<run_id>/` 留着 —— 现场万一连不上,直接讲彩排产物,步骤 3 和 4 完全不受影响。
