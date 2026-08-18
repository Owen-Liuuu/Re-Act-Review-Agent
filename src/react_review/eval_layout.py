"""Where the three eval datasets live on disk.

Logical ids inside frozen JSON (for example ``melanoma_checkpoint_2017``)
do not change. The folders are numbered:

* ``eval/benchmark_1`` — EAT/T1DM
* ``eval/benchmark_2`` — melanoma checkpoint inhibitors
* ``eval/benchmark_3`` — ESCC
"""
from __future__ import annotations

from pathlib import Path

BENCHMARK_1 = "eval/benchmark_1"
BENCHMARK_2 = "eval/benchmark_2"
BENCHMARK_3 = "eval/benchmark_3"

LOGICAL_DIRS = {
    "melanoma_checkpoint_2017": BENCHMARK_2,
}

# Longest prefix first so ``eval/benchmark`` does not swallow ``eval/benchmarks``.
_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("eval/benchmarks/melanoma_checkpoint_2017", BENCHMARK_2),
    ("eval/benchmarks/benchmarks_1", BENCHMARK_3),
    ("eval/benchmark", BENCHMARK_1),
)


def resolve_eval_relpath(rel: str | Path) -> str:
    posix = str(rel).replace("\\", "/")
    for old, new in _PREFIX_ALIASES:
        if posix == old:
            return new
        if posix.startswith(old + "/"):
            return new + posix[len(old):]
    return posix


def resolve_eval_path(root: Path, rel: str | Path) -> Path:
    path = Path(rel)
    if path.is_absolute():
        return path
    return Path(root) / resolve_eval_relpath(path.as_posix())


def benchmark_dir(root: Path, logical_id: str) -> Path:
    rel = LOGICAL_DIRS.get(str(logical_id or ""))
    if rel is None:
        mapped = resolve_eval_relpath(f"eval/benchmarks/{logical_id}")
        rel = mapped if mapped != f"eval/benchmarks/{logical_id}" else f"eval/{logical_id}"
    return Path(root) / rel
