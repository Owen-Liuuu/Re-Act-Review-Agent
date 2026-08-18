# TableCapture v1/v2 A/B gate

This gate compares the frozen production baseline `table_capture_v1` with the
domain-neutral candidate `table_capture_v2`. It covers only the first
characteristics-of-included-studies table in each review. It does not support a
claim about outcome tables, risk-of-bias tables, or all PDFs.

## Copyright and secret boundary

The source PDFs, full cell-level gold, extracted review text, rendered prompts
that contain review text, raw model replies, and scored live artifacts are
private evaluation material. They belong only under the Git-ignored `output/`
tree and must not be staged or committed. A raw reply may reproduce copyrighted
table text. Local configuration and API keys must not be copied into artifacts.
The live test records only the config filename and a service URL with all query
parameters removed.

Public Git contains only:

- the cell schema and annotation rules;
- SHA-256 hashes and aggregate table/row/cell counts;
- the scorer and mutation tests;
- prompt contracts and this procedure.

## Annotation rules

Each line of private gold is one JSON object conforming to
`eval/table_capture_gold_schema.json`.

- `row_id` is positional: `header_01`, then `data_01`, `data_02`, and so on.
- `column_id` is positional (`c01`, `c02`, ...), so duplicate printed header
  names remain distinguishable.
- `raw_value` preserves printed text, including `NR`, `NA`, dashes, units,
  punctuation, line breaks, and line-end hyphenation.
- `normalized_value` may remove only PDF layout artifacts: Unicode
  normalization, whitespace runs, soft hyphens, and a hyphen immediately
  followed by a line break inside one word. It must not standardize units,
  numeric formats, terminology, spelling, or case.
- `value` includes `NR`, `NA`, `not reported`, and printed dashes. These are
  values, not missing cells.
- `true_blank` is a visibly empty independent cell.
- `merged` is the continuation area of a visually spanning cell. For example,
  a study-level total `N` printed once across T1DM and Control rows is recorded
  once as `value`; the continuation row is `merged`. The total must not be
  copied into both cohort rows.
- `not_applicable` is used only when the table explicitly makes that coordinate
  structurally inapplicable.
- A study split across printed rows remains split across data row IDs. Do not
  unpivot it or fill its merged cells.
- Footnotes remain outside the cell gold. A marker printed inside a data cell
  stays in that cell, while the footnote body is not scored as a table cell.
- Every annotation has a PDF page/table/row/column source locator.

The public manifest `eval/table_capture_ab_v1.json` pins two gold files without
publishing them: 2 tables, 25 rows including headers, and 201 cells.

## Metrics

`python -m eval.table_capture_score --gold GOLD.jsonl --capture CAPTURE.json`
reports:

- JSON and rectangular-schema success rate;
- table and row recall;
- exact and normalized cell accuracy;
- value-cell precision and recall;
- unanchored cells and hallucinated cells.

An observed non-empty value in `true_blank`, `merged`, `not_applicable`, or
outside annotated geometry is a hallucinated cell. A wrong value at a real
value coordinate is unanchored but is not counted as inventing a new cell.

## Preflight and mandatory stop

Run this before any real model call:

```powershell
python -m eval.table_capture_preflight --require-private --config configs/config.local.yaml
```

The command must report `ready: true`, two documents, two prompts, four planned
calls, seed 42, verified PDF/gold hashes, model settings, and the ignored raw
artifact path. Pricing is reported as unavailable when no versioned provider
pricing table exists; do not invent a cost estimate.

Stop here and obtain explicit approval for the four paid/network calls.

## Four live calls after approval

Each command selects both the PDF and prompt explicitly. Do not run this block
without approval.

```powershell
$env:RUN_LIVE_LLM='1'
$env:TABLE_CAPTURE_CONFIG='configs/config.local.yaml'
$env:TABLE_CAPTURE_PDF='SRMA.pdf'
$env:TABLE_CAPTURE_PROMPT_PROFILE='table_capture_v1'
python -m pytest tests/parser/test_table_capture_live.py::test_live_llm_uses_the_explicit_table_capture_prompt -q -s -p no:cacheprovider --basetemp=.tmp/pytest-table-capture-live-eat-v1

$env:TABLE_CAPTURE_PROMPT_PROFILE='table_capture_v2'
python -m pytest tests/parser/test_table_capture_live.py::test_live_llm_uses_the_explicit_table_capture_prompt -q -s -p no:cacheprovider --basetemp=.tmp/pytest-table-capture-live-eat-v2

$env:TABLE_CAPTURE_PDF='eval/benchmark_2/raw/review_karlsson_saleh_2017.pdf'
$env:TABLE_CAPTURE_PROMPT_PROFILE='table_capture_v1'
python -m pytest tests/parser/test_table_capture_live.py::test_live_llm_uses_the_explicit_table_capture_prompt -q -s -p no:cacheprovider --basetemp=.tmp/pytest-table-capture-live-melanoma-v1

$env:TABLE_CAPTURE_PROMPT_PROFILE='table_capture_v2'
python -m pytest tests/parser/test_table_capture_live.py::test_live_llm_uses_the_explicit_table_capture_prompt -q -s -p no:cacheprovider --basetemp=.tmp/pytest-table-capture-live-melanoma-v2
```

Every call writes `review_text.txt`, `prompt.txt`, `raw_response.txt`,
`capture.json`, rendered tables, diagnostics, and non-secret metadata below
`output/live_tests/table_capture/<prompt-profile>/`. Score each `capture.json`
against the matching ignored gold JSONL. Preserve all four raw replies even
when parsing or schema validation fails.

## Recorded paired diagnostic

The four approved calls were executed on 2026-08-15 with GLM-4.5-Flash,
temperature 0.1, seed 42, and an 8,192-token output limit per call. The public
result is `eval/table_capture_ab_v1_result.json`; its checker recomputes the
decision from the per-document metrics and refuses a claimed promotion that
contradicts them.

| Document | Prompt | Table / row recall | Exact / normalized cell accuracy | Precision / recall | Unanchored / hallucinated | JSON / schema |
|---|---|---:|---:|---:|---:|---:|
| EAT/T1DM | v1 | 1.000 / 1.000 | 0.9085 / 0.9739 | 0.9825 / 0.9655 | 2 / 2 | 1 / 1 |
| EAT/T1DM | v2 | 1.000 / 1.000 | 0.6863 / 0.7712 | 0.7682 / 1.0000 | 35 / 35 | 1 / 1 |
| Melanoma | v1 | 1.000 / 1.000 | 0.4583 / 1.0000 | 1.0000 / 1.0000 | 0 / 0 | 1 / 1 |
| Melanoma | v2 | 1.000 / 1.000 | 0.4583 / 1.0000 | 1.0000 / 1.0000 | 0 / 0 | 1 / 0 |

The EAT v2 response filled all 35 visually merged continuation cells. This
duplicates study-level Author, Country, total N, Measurement tool, and Overall
quality into the second cohort row. Although v2 recovered four values v1 had
missed, it therefore regressed both cell accuracy and hallucination controls.
For melanoma, both prompts transcribed the scored first table perfectly after
layout normalization, but v2 produced a ragged row in another captured table,
so its whole-response schema check failed. Neither candidate response introduced
the frozen EAT/T1DM example terms into the melanoma review.

The paired decision is `regressed`; `table_capture_v2` is not promoted and the
production default remains `table_capture_v1`. This does not claim v1 is ideal:
its EAT output added two extra header coordinates and missed four values. It
only says the tested v2 cannot safely replace it under the preregistered rules.

All four raw replies remain under the ignored artifact directory. The provider
token usage was not retained and this repository has no versioned pricing
table, so an actual currency cost is not reported or inferred from character
counts.
