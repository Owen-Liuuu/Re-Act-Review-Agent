"""Per-stage time ranges for the desk UI. Never inverted to Remaining.

Bands are historical min–max from ``output/runs`` journals (nonzero
``elapsed_ms``), rounded to a displayable pair. Stages that never recorded a
clock use a residual inferred from complete runs (table + forest + local
parser together finished in about 85–137s). The progress bar uses the high
end so it does not sit at 100% while the step is still running.
"""
from __future__ import annotations

import math
import time
from typing import Any

# stage, title, lo_s, hi_s
PIPELINE: list[tuple[str, str, int, int]] = [
    ("review_pdf_loaded", "Review PDF loaded", 5, 15),
    ("review_lens", "Review lens", 15, 90),
    ("evidence_localize", "Evidence localized", 15, 150),
    ("review_table_capture", "Displays captured", 30, 90),
    ("forest_ocr", "Forest OCR", 60, 180),
    ("claim_origin", "Claim origin", 15, 600),
    ("cohort_registry", "Cohorts", 5, 15),
    ("field_resolution", "Field resolution", 5, 20),
    ("checklist_review", "Checklist", 5, 15),
    ("long_format_rows", "Long format", 5, 15),
    ("reference_coverage", "References", 5, 10),
    ("checklist_study_coverage", "Study coverage", 5, 15),
    ("collect_study", "Collect papers", 10, 4500),
    ("collection_review", "Collection review", 5, 15),
    ("audit_summary", "Audit", 30, 90),
    ("judge_flags", "Flags", 5, 15),
]

_TITLE = {stage: title for stage, title, _lo, _hi in PIPELINE}
_LO = {stage: lo for stage, _title, lo, _hi in PIPELINE}
_HI = {stage: hi for stage, _title, _lo, hi in PIPELINE}
_ORDER = [stage for stage, _title, _lo, _hi in PIPELINE]


def title_for(stage: str, fallback: str = "") -> str:
    return _TITLE.get(stage, fallback or stage.replace("_", " "))


def estimate_s(stage: str) -> int:
    """High end of the band. Used by the elapsed bar."""
    return int(_HI.get(stage, 60))


def estimate_range(stage: str) -> tuple[int, int]:
    return int(_LO.get(stage, 15)), int(_HI.get(stage, 60))


def format_range(lo: int, hi: int) -> str:
    """``15–90s``, ``1–3 min``, or ``10s–75 min``."""
    lo_s = max(0, int(lo))
    hi_s = max(lo_s, int(hi))
    if lo_s == hi_s:
        return _one(lo_s)
    if hi_s <= 90:
        return f"{lo_s}–{hi_s}s"
    hi_m = max(1, math.ceil(hi_s / 60))
    if lo_s < 60:
        return f"{lo_s}s–{hi_m} min"
    lo_m = max(1, lo_s // 60)
    if lo_m == hi_m:
        return f"{hi_m} min"
    return f"{lo_m}–{hi_m} min"


def _one(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds}s"
    return f"{max(1, int(round(seconds / 60)))} min"


def _band(stage: str) -> dict[str, Any]:
    lo, hi = estimate_range(stage)
    return {
        "estimate_s": hi,
        "estimate_min_s": lo,
        "estimate_max_s": hi,
        "estimate_label": format_range(lo, hi),
    }


def upcoming(done_stages: list[str], *, live_stage: str = "") -> list[dict[str, Any]]:
    """Stages after the furthest known checkpoint, collapsed if the tail is long."""
    cursor = -1
    for name in [*done_stages, live_stage]:
        if name in _ORDER:
            cursor = max(cursor, _ORDER.index(name))
    rest: list[dict[str, Any]] = []
    for i, (stage, title, lo, hi) in enumerate(PIPELINE):
        if i <= cursor:
            continue
        rest.append({
            "n": i + 1, "stage": stage, "title": title,
            "estimate_s": hi, "estimate_min_s": lo, "estimate_max_s": hi,
            "estimate_label": format_range(lo, hi),
        })
    if len(rest) <= 3:
        return rest
    head, tail = rest[:2], rest[2:]
    lo = sum(int(r["estimate_min_s"]) for r in tail)
    hi = sum(int(r["estimate_s"]) for r in tail)
    head.append({
        "n": f"{head[-1]['n'] + 1}+" if head else "07+",
        "stage": tail[0]["stage"],
        "title": "Collect · audit" if any(
            r["stage"] in {"collect_study", "audit_summary"} for r in tail
        ) else "Later",
        "estimate_s": hi,
        "estimate_min_s": lo,
        "estimate_max_s": hi,
        "estimate_label": format_range(lo, hi),
    })
    return head


def decorate_live(
    raw: dict[str, Any] | None,
    steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not raw or raw.get("state") == "idle":
        return None
    if str(raw.get("state") or "") == "failed":
        started = float(raw.get("started_unix") or 0)
        if started:
            elapsed = max(0, int(time.time() - started))
        else:
            elapsed = int(raw.get("elapsed_s") or 0)
        out = dict(raw)
        out["stage"] = str(raw.get("stage") or "failed")
        out["title"] = "Failed"
        out["elapsed_s"] = elapsed
        out["estimate_s"] = 0
        out["estimate_min_s"] = 0
        out["estimate_max_s"] = 0
        out["estimate_label"] = ""
        out["upcoming"] = []
        return out
    stage = str(raw.get("stage") or raw.get("label") or "")
    started = float(raw.get("started_unix") or 0)
    if started:
        elapsed = max(0, int(time.time() - started))
    else:
        elapsed = int(raw.get("elapsed_s") or 0)
    done = [str(s.get("stage") or "") for s in steps]
    out = dict(raw)
    out["stage"] = stage
    out["title"] = title_for(stage, str(raw.get("title") or raw.get("label") or stage))
    out["elapsed_s"] = elapsed
    out.update(_band(stage))
    out["upcoming"] = upcoming(done, live_stage=stage)
    return out
