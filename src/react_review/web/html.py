"""One-shell desk UI. Four states: upload, running, waiting, complete."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from react_review.core.config import BACKEND_STEPS
from react_review.llm.catalog import (
    DEFAULTS, SIMPLE_STEPS, TEXT_MODELS, VENDORS, VISION_MODELS, catalog_payload,
)
from react_review.web.eta import format_range

_DIR = Path(__file__).resolve().parent
_NOTE = (
    "PDFs and artifacts stay on this host (Hong Kong volume). "
    "Only the text a step reads is sent, to that step's model vendor."
)

# Plain names for the 13 routable model tasks, as the homepage shows them.
_TASK_LABELS = {
    "review_lens": "Review lens",
    "evidence_localize": "Evidence located",
    "table_capture": "Tables captured",
    "forest_ocr_vision": "Forest plot images",
    "forest_ocr_text": "Forest plot text",
    "claim_origin": "Claim origin",
    "unpivot": "Long format",
    "references": "Source papers matched",
    "field_resolution": "Field concepts",
    "extract_locate": "Extract: locate",
    "extract_transcribe": "Extract: transcribe",
    "source_row_map": "Source row map",
    "semantic_compare": "Semantic compare",
}


def gear_tasks(routing: dict[str, str] | None) -> dict[str, list[str]]:
    """Which model tasks each homepage gear will serve in a web run.

    Mirrors ``apply_run_gears``: the host routing, with SIMPLE_STEPS always on
    the transcribe gear. The labels used to be fixed text and could name a task
    under Complex that the host config had already moved to transcribe. A task
    routed to some other host profile is listed as ``other`` instead of being
    shown under a gear it does not use.
    """
    effective = dict(routing or {})
    for step in SIMPLE_STEPS:
        effective[step] = "transcribe"
    tasks: dict[str, list[str]] = {"complex": [], "simple": [], "visual": [], "other": []}
    for step in BACKEND_STEPS:
        profile = effective.get(step)
        if step == "forest_ocr_vision" and not profile:
            tasks["visual"].append(step)
        elif profile == "transcribe":
            tasks["simple"].append(step)
        elif profile:
            tasks["other"].append(step)
        else:
            tasks["complex"].append(step)
    return tasks


def _task_list(steps: list[str]) -> str:
    return ", ".join(_TASK_LABELS.get(step, step) for step in steps) or "No tasks"


def _css() -> str:
    path = _DIR / "desk.css"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _js() -> str:
    path = _DIR / "desk.js"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _fmt_secs(seconds: float, *, approx: bool = False) -> str:
    sec = max(0, int(round(float(seconds))))
    if sec < 90:
        body = f"{sec}s"
    else:
        body = f"{max(1, int(round(sec / 60)))} min"
    return f"~{body}" if approx else body


def _fmt_estimate(obj: dict[str, Any] | None) -> str:
    """Range label when present; otherwise a single ~N s / ~N min."""
    if not obj:
        return ""
    label = str(obj.get("estimate_label") or "").strip()
    if label:
        return label
    lo = obj.get("estimate_min_s")
    hi = obj.get("estimate_max_s") or obj.get("estimate_s")
    try:
        if lo is not None and hi is not None and int(hi) > 0:
            return format_range(int(lo), int(hi))
        if hi is not None and float(hi) > 0:
            return _fmt_secs(float(hi), approx=True)
    except (TypeError, ValueError):
        return ""
    return ""


def _fmt_ms(elapsed_ms: Any, *, approx: bool = False) -> str:
    try:
        ms = int(elapsed_ms or 0)
    except (TypeError, ValueError):
        ms = 0
    return _fmt_secs(ms / 1000, approx=approx)


def _pad_n(index: Any) -> str:
    try:
        return f"{int(index):02d}"
    except (TypeError, ValueError):
        return str(index or "")


def _tree_group(title: str, steps: list[str], *, first: bool = False) -> str:
    pad = "" if first else ' style="padding-top:14px"'
    items = "".join(
        f'<a href="#"><span class="n"></span><span class="label">'
        f'{escape(_TASK_LABELS.get(step, step))}</span></a>'
        for step in steps)
    return f'<div class="group"{pad}>{escape(title)}</div>{items}'


def home_page(run_ids: list[str], *, busy: str | None = None,
              error: str = "", require_keys: bool = False,
              routing: dict[str, str] | None = None) -> str:
    if busy:
        chip = (
            f'<span class="chip blue"><span class="pulse"></span> Running · '
            f'<a href="/runs/{escape(busy)}">{escape(busy)}</a></span>'
        )
    else:
        chip = '<span class="chip muted"><span class="dot"></span> Idle · pick models</span>'
    tasks = gear_tasks(routing)
    other = (_tree_group("Other host profiles", tasks["other"])
             if tasks["other"] else "")
    tree = (
        '<div class="k" style="padding:6px 21px 12px">This run will use</div>'
        + _tree_group("Complex · llm", tasks["complex"], first=True)
        + _tree_group("Simple · transcribe", tasks["simple"])
        + _tree_group("Visual · vision", tasks["visual"])
        + other +
        '<p class="hint" style="margin-top:18px">Start a run. Steps land here one by one.</p>'
        f"{_recent_runs(run_ids)}"
        f'<p class="desk-note">{escape(_NOTE)}</p>'
    )
    err = (
        f'<div id="missing" class="alert bad"><strong>{escape(error)}</strong></div>'
        if error else
        '<div id="missing" class="alert bad" hidden></div>'
    )
    editor = f"""
<div class="k">New run</div>
<h1 style="margin:8px 0 10px">Attach the review</h1>
<p style="color:var(--muted);margin:0 0 8px">Same desk. Pick three models on the right. Keys live in Account — not on this page.</p>
<p class="alert" id="key-hint">No keys in Account yet. Start still uses this host’s config. Open Account (top right) to paste native vendor keys — not an OpenRouter key.</p>
{err}
<form id="start" action="/runs" method="post" enctype="multipart/form-data">
  <label class="file" id="review-row">
    <input id="review" type="file" name="review" accept="application/pdf">
    <div class="ico">PDF</div>
    <div>
      <strong>Review</strong>
      <div id="review-hint" style="color:var(--muted);font-size:0.85em">Required · published systematic review</div>
    </div>
    <span class="chip blue" id="review-chip">Choose</span>
  </label>
  <label class="file" id="src-row">
    <input id="sources" type="file" name="sources" accept="application/pdf" multiple>
    <div class="ico">SRC</div>
    <div>
      <strong>Sources</strong>
      <div id="src-hint" style="color:var(--muted);font-size:0.85em">Optional · local papers skip retrieval</div>
    </div>
    <span class="chip muted" id="src-chip">Choose</span>
  </label>
  <button class="btn" type="submit" style="margin-top:10px">Start run</button>
</form>
"""
    problems = _home_gears(tasks)
    return _shell(
        "ReAct-Review",
        status="idle",
        chip=chip,
        icon_on="n",
        tree=tree,
        editor=editor,
        problems=problems,
        require_keys=require_keys,
        account=True,
        catalog=True,
    )


def _select_options(values: tuple[str, ...] | list[str], selected: str) -> str:
    return "".join(
        f'<option value="{escape(v)}"'
        f'{" selected" if v == selected else ""}>{escape(v)}</option>'
        for v in values)


def _home_gears(tasks: dict[str, list[str]] | None = None) -> str:
    tasks = tasks if tasks is not None else gear_tasks(None)
    cv, cm = DEFAULTS["complex"]
    sv, sm = DEFAULTS["simple"]
    vv, vm = DEFAULTS["visual"]
    vendors = _select_options(VENDORS, cv)
    vendors_s = _select_options(VENDORS, sv)
    vendors_v = _select_options(VENDORS, vv)
    return f"""
<div class="k">Models</div>
<h2 style="margin-top:10px">Three gears</h2>
<p style="color:var(--muted);font-size:0.9em;margin:0 0 14px">Catalog IDs follow OpenRouter. Each gear uses that vendor’s own API key from Account.</p>
<div class="gear">
  <header>Complex <span class="chip muted">llm</span></header>
  <div class="pick">
    <select id="c-vendor" form="start" name="complex_vendor" aria-label="Complex vendor">{vendors}</select>
    <select id="c-model" form="start" name="complex_model" aria-label="Complex model">{_select_options(TEXT_MODELS[cv], cm)}</select>
  </div>
  <p class="slots">{escape(_task_list(tasks["complex"]))}</p>
</div>
<div class="gear">
  <header>Simple <span class="chip muted">transcribe</span></header>
  <div class="pick">
    <select id="s-vendor" form="start" name="simple_vendor" aria-label="Simple vendor">{vendors_s}</select>
    <select id="s-model" form="start" name="simple_model" aria-label="Simple model">{_select_options(TEXT_MODELS[sv], sm)}</select>
  </div>
  <p class="slots">{escape(_task_list(tasks["simple"]))}</p>
</div>
<div class="gear">
  <header>Visual <span class="chip blue">vision</span></header>
  <div class="pick">
    <select id="v-vendor" form="start" name="visual_vendor" aria-label="Visual vendor">{vendors_v}</select>
    <select id="v-model" form="start" name="visual_model" aria-label="Visual model">{_select_options(VISION_MODELS[vv], vm)}</select>
  </div>
  <p class="slots">{escape(_task_list(tasks["visual"]))}</p>
</div>
"""


def _account_sheet() -> str:
    return """
<div class="veil" id="veil" hidden></div>
<aside class="sheet" id="sheet" hidden>
  <div class="k">User center</div>
  <h1>Account</h1>
  <p style="color:var(--muted);font-size:0.9em;margin:0 0 16px">Name and email for the run log. Three native keys — one per gear. Not an OpenRouter key.</p>
  <form id="account-form">
    <div class="row">
      <label for="who">Display name</label>
      <input id="who" form="start" name="name" type="text" autocomplete="name" placeholder="Li Wei">
    </div>
    <div class="row">
      <label for="mail">Email</label>
      <input id="mail" form="start" name="email" type="email" autocomplete="email" placeholder="li@lab.org">
    </div>
    <hr class="rule">
    <div class="k" style="margin-bottom:10px">API keys</div>
    <div class="key-field" id="wrap-c">
      <label id="lab-c" for="key-c">Complex</label>
      <input id="key-c" form="start" name="complex_key" type="password" autocomplete="off">
      <p class="hint-k" id="hint-c"></p>
    </div>
    <div class="key-field" id="wrap-s">
      <label id="lab-s" for="key-s">Simple</label>
      <input id="key-s" form="start" name="simple_key" type="password" autocomplete="off">
      <p class="hint-k" id="hint-s"></p>
    </div>
    <div class="key-field" id="wrap-v">
      <label id="lab-v" for="key-v">Visual</label>
      <input id="key-v" form="start" name="visual_key" type="password" autocomplete="off">
      <p class="hint-k" id="hint-v"></p>
    </div>
    <div class="actions">
      <button class="btn" type="submit">Save</button>
      <button class="btn ghost" id="sheet-close" type="button">Close</button>
    </div>
  </form>
</aside>
"""


def run_page(
    run_id: str,
    *,
    status: str,
    steps: list[dict[str, Any]],
    blocked: dict[str, Any] | None,
    has_report: bool,
    has_log: bool,
    live: dict[str, Any] | None = None,
    has_package: bool = False,
    poll: bool = True,
) -> str:
    waiting = status == "waiting"
    running = status in {"running", "queued"}
    closed = status in {"complete", "stopped"}
    # Waiting must not auto-refresh: a reload would wipe checkbox state.
    refresh = 5 if (not poll and status in {"running", "queued"}) else None
    blocked_idx = blocked.get("index") if isinstance(blocked, dict) else None
    current = _current_step(steps, blocked=blocked, live=live, status=status)
    chip = _run_chip(status, current=current, live=live)
    icon_on = "r" if (closed and has_report) else "g"
    tree = _tree(
        steps, status=status, blocked_idx=blocked_idx, live=live,
        has_report=has_report, run_id=run_id)
    if running:
        editor = _editor_running(steps, live=live, current=current, status=status)
        problems = None
    elif waiting:
        editor = _editor_waiting(steps, blocked=blocked, run_id=run_id)
        problems = _problems_waiting(blocked or current or {}, run_id=run_id)
    elif status == "failed":
        editor = _editor_failed(live=live)
        problems = _problems_failed(run_id, has_log=has_log)
    else:
        editor = _editor_closed(
            run_id, status=status, steps=steps, has_report=has_report)
        problems = _problems_closed(
            run_id, status=status, has_report=has_report,
            has_log=has_log, has_package=has_package)
    return _shell(
        f"run {run_id}",
        status=status,
        run_id=run_id,
        chip=chip,
        icon_on=icon_on,
        tree=tree,
        editor=editor,
        problems=problems,
        running=running,
        refresh=refresh,
        has_report=has_report,
        poll=poll,
        top_id=run_id,
        top_links=_top_links(run_id, has_report=has_report, has_log=has_log,
                             has_package=has_package),
        live_stage=str((live or {}).get("stage") or ""),
        estimate_s=(live or {}).get("estimate_s"),
        account=True,
    )


def _shell(
    title: str,
    *,
    status: str,
    chip: str,
    icon_on: str,
    tree: str,
    editor: str,
    problems: str | None,
    run_id: str = "",
    running: bool = False,
    refresh: int | None = None,
    has_report: bool = False,
    poll: bool = False,
    top_id: str = "",
    top_links: str = "",
    live_stage: str = "",
    estimate_s: Any = None,
    require_keys: bool = False,
    account: bool = False,
    catalog: bool = False,
) -> str:
    rid = escape(run_id)
    report_href = f"/runs/{rid}/report" if run_id and has_report else "#"
    report_cls = "on" if icon_on == "r" else ("dim" if not has_report else "")
    gate_cls = "on" if icon_on == "g" else ""
    new_cls = "on" if icon_on == "n" else ""
    ide_cls = "ide running" if running else "ide"
    editor_cls = "editor rpt-frame" if "report-frame" in editor else "editor"
    head = (
        f'<meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{_css()}</style>"
    )
    if refresh:
        head += f'\n<meta http-equiv="refresh" content="{int(refresh)}">'
    script_tag = ""
    js = _js()
    extra = ""
    if catalog:
        extra = (
            '<script type="application/json" id="desk-catalog">'
            f"{json.dumps(catalog_payload(), ensure_ascii=False)}"
            "</script>"
        )
    if js:
        script_tag = f"{extra}<script>{js}</script>"
    id_span = (
        f'<span class="mono" style="color:var(--muted)">{escape(top_id)}</span>'
        if top_id else "")
    problems_html = (
        f'<aside class="problems" id="desk-problems">{problems}</aside>'
        if problems is not None else "")
    body_attrs = f' data-status="{escape(status)}"'
    if run_id:
        body_attrs += f' data-run-id="{rid}"'
    if live_stage:
        body_attrs += f' data-live-stage="{escape(live_stage)}"'
    if estimate_s:
        body_attrs += f' data-estimate-s="{escape(str(estimate_s))}"'
    if require_keys:
        body_attrs += ' data-require-keys="1"'
    account_btn = (
        '<button class="account" id="account-btn" type="button">Account</button>'
        if account else "")
    sheet = _account_sheet() if account else ""
    return (
        "<!DOCTYPE html><html lang=\"en\"><head>"
        f"{head}</head>"
        f"<body{body_attrs}>"
        '<div class="app">'
        '<nav class="icons">'
        f'<a class="{new_cls}" href="/" title="New run">N</a>'
        f'<a class="{gate_cls}" href="{f"/runs/{rid}" if run_id else "/"}" title="This run">G</a>'
        f'<a class="{report_cls}" href="{report_href}" title="Report">R</a>'
        "</nav>"
        "<div>"
        '<header class="top">'
        '<a class="wordmark" href="/">ReAct-Review</a>'
        f"{id_span}"
        f'<span id="desk-chip">{chip}</span>'
        '<span class="sp"></span>'
        f"{top_links}"
        f"{account_btn}"
        "</header>"
        f'<div class="{ide_cls}" id="desk-ide">'
        f'<aside class="tree" id="desk-tree">{tree}</aside>'
        f'<section class="{editor_cls}" id="desk-editor">{editor}</section>'
        f"{problems_html}"
        "</div></div></div>"
        f"{sheet}"
        f"{script_tag}"
        "</body></html>"
    )


def _top_links(run_id: str, *, has_report: bool, has_log: bool,
               has_package: bool) -> str:
    rid = escape(run_id)
    links = []
    if has_package:
        links.append(f'<a href="/runs/{rid}/package">package.json</a>')
    if has_log:
        links.append(f'<a href="/runs/{rid}/log">log</a>')
    if has_report:
        links.append(f'<a href="/runs/{rid}/report">report.html</a>')
    if not links:
        return ""
    return "<nav>" + " · ".join(links) + "</nav>"


def _recent_runs(run_ids: list[str]) -> str:
    if not run_ids:
        return ""
    rows = "".join(
        f'<a href="/runs/{escape(rid)}"><span class="n">·</span>'
        f'<span class="label">{escape(rid)}</span></a>'
        for rid in run_ids)
    return (
        '<div class="k" style="padding:16px 14px 8px">Recent</div>' + rows
    )


def _current_step(
    steps: list[dict[str, Any]],
    *,
    blocked: dict[str, Any] | None,
    live: dict[str, Any] | None,
    status: str,
) -> dict[str, Any] | None:
    if status == "waiting" and isinstance(blocked, dict):
        for step in steps:
            if step.get("index") == blocked.get("index"):
                return step
        return blocked
    if live:
        return live
    if steps:
        return steps[-1]
    return None


def _run_chip(
    status: str,
    *,
    current: dict[str, Any] | None,
    live: dict[str, Any] | None,
) -> str:
    title = ""
    if current:
        title = str(current.get("title") or current.get("label")
                    or current.get("stage") or "")
    if status == "queued":
        return '<span class="chip muted"><span class="dot"></span> Queued</span>'
    if status == "running":
        bits = ["Running"]
        if title:
            bits.append(escape(title))
        elapsed = ""
        estimate = ""
        if live:
            if live.get("elapsed_s") is not None:
                elapsed = _fmt_secs(float(live["elapsed_s"]))
            estimate = _fmt_estimate(live)
        extra = " · ".join(bits)
        times = ""
        if elapsed:
            times = f" · {elapsed}"
            if estimate:
                times += f" / {estimate}"
        return (
            f'<span class="chip blue"><span class="pulse"></span> {extra}{times}</span>'
        )
    if status == "waiting":
        n = ""
        if current and current.get("index") is not None:
            n = f" · step {escape(str(current.get('index')))}"
        return (
            f'<span class="chip warn"><span class="dot"></span> Waiting · your turn{n}</span>'
        )
    if status == "stopped":
        return '<span class="chip bad"><span class="dot"></span> Stopped</span>'
    if status == "failed":
        return '<span class="chip bad"><span class="dot"></span> Failed</span>'
    if status == "complete":
        return '<span class="chip good"><span class="dot"></span> Complete</span>'
    return f'<span class="chip muted"><span class="dot"></span> {escape(status)}</span>'


def _tree(
    steps: list[dict[str, Any]],
    *,
    status: str,
    blocked_idx: Any,
    live: dict[str, Any] | None,
    has_report: bool,
    run_id: str,
) -> str:
    rows: list[str] = [
        '<div class="k" style="padding:4px 14px 10px">This run</div>'
    ]
    live_stage = str((live or {}).get("stage") or "")
    last_done = steps[-1]["index"] if steps else None
    for step in steps:
        idx = step.get("index")
        title = str(step.get("title") or step.get("stage") or "")
        on = False
        cls = ""
        if status == "waiting" and idx == blocked_idx:
            on = True
        elif status in {"running", "queued"} and live_stage:
            on = str(step.get("stage") or "") == live_stage
        elif status in {"running", "queued"} and not live_stage and idx == last_done:
            on = True
        eta = ""
        if step.get("elapsed_ms"):
            eta = _fmt_ms(step.get("elapsed_ms"))
        elif status in {"running", "queued"}:
            eta = _fmt_estimate(step)
        rows.append(_tree_row(
            n=_pad_n(idx), label=title, eta=eta, on=on, running=bool(on and status == "running")))
    if live and str(live.get("stage") or "") not in {
            str(s.get("stage") or "") for s in steps}:
        eta = _fmt_estimate(live)
        if not eta and live.get("elapsed_s") is not None:
            eta = _fmt_secs(float(live["elapsed_s"]))
        title = str(live.get("title") or live.get("label") or live.get("stage") or "Working")
        n = _pad_n(live.get("index") or (int(last_done or 0) + 1))
        rows.append(_tree_row(
            n=n, label=title, eta=eta, on=True,
            running=status in {"running", "queued"}))
    future = list(live.get("upcoming") or []) if live else []
    for item in future:
        rows.append(_tree_row(
            n=_pad_n(item.get("n") or item.get("index") or ""),
            label=str(item.get("title") or item.get("stage") or ""),
            eta=_fmt_estimate(item),
            future=True,
        ))
    if status in {"complete", "stopped"} and has_report:
        rows.append(_tree_row(
            n="R", label="Report", on=True,
            href=f"/runs/{escape(run_id)}/report"))
    if len(rows) == 1:
        if status == "failed":
            rows.append('<p class="hint">(the run failed before the first step)</p>')
        else:
            rows.append('<p class="hint">(no steps yet — waiting for the parser)</p>')
    rows.append(f'<p class="desk-note">{escape(_NOTE)}</p>')
    return "".join(rows)


def _tree_row(
    *,
    n: str,
    label: str,
    eta: str = "",
    on: bool = False,
    running: bool = False,
    future: bool = False,
    href: str = "#",
) -> str:
    classes = []
    if on:
        classes.append("on")
    if running:
        classes.append("run")
    cls = f' class="{" ".join(classes)}"' if classes else ""
    style = ' style="opacity:.45"' if future else ""
    eta_html = f'<span class="eta">{escape(eta)}</span>' if eta else ""
    return (
        f'<a href="{escape(href)}"{cls}{style}>'
        f'<span class="n">{escape(n)}</span>'
        f'<span class="label">{escape(label)}</span>'
        f"{eta_html}</a>"
    )


def _editor_running(
    steps: list[dict[str, Any]],
    *,
    live: dict[str, Any] | None,
    current: dict[str, Any] | None,
    status: str,
) -> str:
    title = "Queued" if status == "queued" else "Working"
    caption = "The worker has not written a live step yet."
    if current:
        title = str(current.get("title") or current.get("label")
                    or current.get("stage") or title)
    if live:
        cap = str(live.get("caption") or "")
        label = str(live.get("label") or "")
        if cap:
            caption = cap if not live.get("index") else (
                f"{label} · {cap}" if label else cap)
        elif label:
            caption = label
        else:
            caption = title
    elapsed_html = "—"
    estimate_html = "—"
    bar = 0
    started = ""
    if live:
        if live.get("elapsed_s") is not None:
            elapsed_html = _fmt_secs(float(live["elapsed_s"]))
        estimate_html = _fmt_estimate(live) or "—"
        cap = live.get("estimate_max_s") or live.get("estimate_s")
        if cap:
            try:
                bar = min(100, int(
                    100 * float(live.get("elapsed_s") or 0) / float(cap)))
            except (TypeError, ValueError, ZeroDivisionError):
                bar = 0
        if live.get("started_unix"):
            started = str(live["started_unix"])
    log = _running_log(steps, live=live)
    past = _past_details(steps, skip_index=(current or {}).get("index") if not live else None)
    return f"""
<div class="k">Live · watching the step</div>
<h1 style="margin:6px 0 8px" id="desk-live-title">{escape(title)}</h1>
<div class="live" id="desk-live" data-started-unix="{escape(started)}">
  <div style="display:flex;gap:10px;align-items:center">
    <span class="pulse"></span>
    <strong id="desk-live-caption">{escape(caption)}</strong>
  </div>
  <div class="eta-row">
    <span>Elapsed <b id="desk-elapsed">{elapsed_html}</b></span>
    <span>Estimate <b id="desk-estimate">{estimate_html}</b></span>
  </div>
  <div class="eta-bar" id="desk-eta-bar"><i style="width:{bar}%"></i></div>
  <div class="log" id="desk-log">{log}</div>
</div>
{past}
"""


def _running_log(
    steps: list[dict[str, Any]],
    *,
    live: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    for step in steps[-6:]:
        idx = _pad_n(step.get("index"))
        title = escape(str(step.get("title") or step.get("stage") or ""))
        took = _fmt_ms(step.get("elapsed_ms")) if step.get("elapsed_ms") else "done"
        lines.append(f"{idx} {title} · {escape(took)}")
    if live:
        cap = escape(str(live.get("caption") or live.get("label") or live.get("stage") or "working"))
        lines.append(f'<span class="now">live {cap}</span>')
    elif not lines:
        lines.append('<span class="now">waiting for the parser …</span>')
    return "<br>\n".join(lines)


def _past_details(
    steps: list[dict[str, Any]],
    *,
    skip_index: Any = None,
    limit: int = 8,
) -> str:
    out: list[str] = []
    for step in reversed(steps):
        if skip_index is not None and step.get("index") == skip_index:
            continue
        if len(out) >= limit:
            break
        idx = _pad_n(step.get("index"))
        title = escape(str(step.get("title") or step.get("stage") or ""))
        decision = escape(str(step.get("decision") or "done"))
        took = _fmt_ms(step.get("elapsed_ms")) if step.get("elapsed_ms") else ""
        mark = f"{took} · {decision}" if took else decision
        body = "".join(
            f"<pre>{escape(block)}</pre>"
            for block in (step.get("render_blocks") or []) if block)
        out.append(
            f'<details class="past"><summary>{idx} {title} '
            f'<span class="ok">{mark}</span></summary>'
            f'<div style="padding:0 12px 10px">{body}</div></details>'
        )
    return "".join(out)


def _editor_waiting(
    steps: list[dict[str, Any]],
    *,
    blocked: dict[str, Any] | None,
    run_id: str,
) -> str:
    step = blocked or {}
    for row in steps:
        if row.get("index") == (blocked or {}).get("index"):
            step = row
            break
    title = escape(str(step.get("title") or step.get("stage") or "Checkpoint"))
    warns = "".join(
        f'<div class="alert">{escape(str(w))}</div>'
        for w in (step.get("warnings") or []))
    items = _selectable_editor(step)
    pres = "".join(
        f"<pre>{escape(block)}</pre>"
        for block in (step.get("render_blocks") or []) if block)
    past = _past_details(steps, skip_index=step.get("index"))
    return f"""
<div class="k">Step finished · thrown to you</div>
<h1 style="margin:6px 0 8px">{title}</h1>
<div class="throw">The extractor paused here on purpose. Check the artefact, drop anything unfaithful, then Continue.</div>
{warns}{items}{pres}{past}
"""


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("table_id") or row.get("display_id") or "")


def _row_on(row: dict[str, Any]) -> bool:
    label = str(row.get("label") or "")
    return not label.lower().lstrip().startswith("[off]")


def _selectable_rows(step: dict[str, Any]) -> list[dict[str, Any]]:
    payload = step.get("payload") or {}
    key = step.get("selectable") or ""
    rows = payload.get(key) if key else None
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def _selectable_editor(step: dict[str, Any]) -> str:
    rows = _selectable_rows(step)
    if not rows:
        return ""
    bits: list[str] = []
    for row in rows:
        rid = _row_id(row)
        if not rid:
            continue
        label = escape(str(row.get("label") or rid))
        checked = " checked" if _row_on(row) else ""
        bits.append(
            f'<div class="item"><label>'
            f'<input type="checkbox" form="desk-decision" name="keep" '
            f'value="{escape(rid)}"{checked}> {label}</label></div>'
        )
    return "".join(bits)


def _problems_waiting(step: dict[str, Any], *, run_id: str) -> str:
    rows = _selectable_rows(step)
    kept = sum(1 for r in rows if _row_on(r))
    summary = (
        f"<h2 style=\"margin-top:8px\">Keep {kept} of {len(rows)}</h2>"
        '<p style="color:var(--muted);font-size:13px">Unchecked items are dropped. The run stays on this step until you post.</p>'
        if rows else
        '<h2 style="margin-top:8px">Your turn</h2>'
        '<p style="color:var(--muted);font-size:13px">The run stays on this step until you post.</p>'
    )
    return (
        '<div class="k">Decision</div>'
        f"{summary}"
        f"{_decision_form(step, run_id=run_id)}"
    )


def _editor_failed(*, live: dict[str, Any] | None) -> str:
    caption = ""
    if live:
        caption = str(live.get("caption") or live.get("label") or "")
    caption = caption or (
        "The provider refused the request. See the serve terminal.")
    return (
        '<div class="k">Run failed</div>'
        '<h1 style="margin:6px 0 8px">Failed</h1>'
        f'<div class="alert bad"><strong>The run stopped.</strong> '
        f"{escape(caption)}</div>"
        '<p style="color:var(--muted);font-size:13px">'
        "HTTP 402 means the vendor key has no remaining credit. "
        "Open Account, paste a funded native key, and start a new run.</p>"
    )


def _problems_failed(run_id: str, *, has_log: bool) -> str:
    rid = escape(run_id)
    log = (
        f'<a class="btn ghost" href="/runs/{rid}/log">Open log</a>'
        if has_log else "")
    return f"""
<div class="k">Decision</div>
<h2 style="margin-top:8px">Run failed</h2>
<p style="color:var(--muted);font-size:13px">No gate is waiting. Fix the key or credit, then start another run.</p>
<div class="actions">
  {log}
  <a class="btn" href="/">New run</a>
</div>
"""


def _editor_closed(
    run_id: str,
    *,
    status: str,
    steps: list[dict[str, Any]],
    has_report: bool,
) -> str:
    if has_report:
        return (
            f'<iframe class="report-frame" title="Report" '
            f'src="/runs/{escape(run_id)}/report"></iframe>'
        )
    kicker = "Run stopped" if status == "stopped" else "Run closed"
    body = _past_details(steps, limit=20) or "<p>(no steps)</p>"
    return (
        f'<div class="k">{escape(kicker)}</div>'
        f'<h1 style="margin:6px 0 8px">{escape(run_id)}</h1>'
        f"{body}"
    )


def _problems_closed(
    run_id: str,
    *,
    status: str,
    has_report: bool,
    has_log: bool,
    has_package: bool,
) -> str:
    rid = escape(run_id)
    heading = "Run closed" if status == "complete" else "Run stopped"
    links = []
    if has_package:
        links.append(
            f'<a class="btn ghost" href="/runs/{rid}/package">Open package.json</a>')
    if has_report:
        links.append(
            f'<a class="btn ghost" href="/runs/{rid}/report">Open report.html</a>')
    if has_log:
        links.append(f'<a class="btn ghost" href="/runs/{rid}/log">Open log</a>')
    return f"""
<div class="k">Decision</div>
<h2 style="margin-top:8px">{heading}</h2>
<p style="color:var(--muted);font-size:13px">No gate is waiting. Export the evidence package or start another run.</p>
<div class="actions">
  {''.join(links)}
  <a class="btn" href="/">New run</a>
</div>
"""


def _decision_form(step: dict[str, Any], *, run_id: str) -> str:
    idx = int(step.get("index") or 0)
    offers = set(step.get("offers") or [])
    buttons = [
        '<button class="btn" name="decision" value="continue">Continue</button>',
        '<button class="btn danger" name="decision" value="stop">Stop</button>',
    ]
    if "retry" in offers:
        buttons.insert(
            1, '<button class="btn ghost" name="decision" value="retry">Retry</button>')
    if "retry_alt" in offers:
        buttons.insert(
            2,
            '<button class="btn ghost" name="decision" value="retry_alt">'
            "Retry with model 2</button>")
    hidden_ids = ""
    for row in _selectable_rows(step):
        rid = _row_id(row)
        if rid:
            hidden_ids += (
                f'<input type="hidden" name="all" value="{escape(rid)}">')
    return (
        f'<form class="actions" id="desk-decision" method="post" '
        f'action="/runs/{escape(run_id)}/decision">'
        f'<input type="hidden" name="index" value="{idx}">'
        f"{hidden_ids}"
        f"{''.join(buttons)}"
        "</form>"
    )
