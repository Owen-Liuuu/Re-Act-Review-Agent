"""Old eval folder names still resolve after the numbered rename."""
from __future__ import annotations

from react_review.eval_layout import (
    BENCHMARK_1,
    BENCHMARK_2,
    BENCHMARK_3,
    resolve_eval_relpath,
)


def test_frozen_table_capture_pdf_path_still_resolves():
    assert resolve_eval_relpath(
        "eval/benchmarks/melanoma_checkpoint_2017/raw/review_karlsson_saleh_2017.pdf"
    ) == f"{BENCHMARK_2}/raw/review_karlsson_saleh_2017.pdf"


def test_eat_prefix_does_not_swallow_benchmarks():
    assert resolve_eval_relpath("eval/benchmark") == BENCHMARK_1
    assert resolve_eval_relpath(
        "eval/benchmark/raw/EAT_T1DM_SRMA.pdf"
    ) == f"{BENCHMARK_1}/raw/EAT_T1DM_SRMA.pdf"
    assert resolve_eval_relpath(
        "eval/benchmarks/melanoma_checkpoint_2017"
    ) == BENCHMARK_2


def test_escc_worksheet_alias():
    assert resolve_eval_relpath(
        "eval/benchmarks/benchmarks_1/gold_worksheet.csv"
    ) == f"{BENCHMARK_3}/gold_worksheet.csv"
