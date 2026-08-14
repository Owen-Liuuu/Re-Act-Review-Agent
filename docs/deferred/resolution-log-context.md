# 字段解析日志缺少位置上下文

- 记录日期：2026-08-14
- 状态：待调整
- 类型：CLI 日志可读性 / 可观测性
- 相关位置：`src/react_review/parser/review_parser.py::_render_resolutions`

## 问题

字段解析汇总会逐条打印 `raw_field_name -> field_type`，但没有显示该解析问题来自哪一页、哪张表、哪一列。不同位置中常见的同名表头（例如 `Country`、`N`、`Measurement tool`）因此看起来像同一条日志被重复输出。

实际的唯一解析问题包含更多上下文；当前日志只展示了名称和受影响单元格数量，隐藏了用于区分这些问题的位置信息，容易造成“程序重复运行或重复解析”的误解。

## 调整建议

每条解析日志至少增加以下定位信息：

- `page`：PDF 页码；
- `table`：表格 ID；
- `column`：列标题或列坐标。

`ResolutionCellRef` 已包含 `table_id`、`cell_ref` 和 `column_header`，可优先用于日志展示；`page` 当前未包含在该结构中，需要确认上游表格捕获阶段是否可以传递页码。

若一条解析决定影响多个位置，应显示简短的代表位置和总位置数，详细位置可放在 artifact 中，避免终端日志过长。

示例：

```text
'Country' -> country [page=12; table=table_3; column=Country; 1 cell]
'Country' -> country [page=18; table=table_7; column=Country; 1 cell]
```

## 验收标准

- 同名字段来自不同位置时，用户可直接从终端日志区分；
- 日志至少展示 `table` 和 `column`，可获得页码时同时展示 `page`；
- 多位置解析不会把所有单元格坐标全部铺开；
- 现有解析结果、解析键和 evidence package 内容不因日志展示调整而改变。
