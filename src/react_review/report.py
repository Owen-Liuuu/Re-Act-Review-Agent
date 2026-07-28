"""Render audit + benchmark results into standalone, self-contained HTML reports.

Deterministic (no LLM). Two renderers:
  - render_html_report(pkg)      : one audit run → evidence grouped BY SOURCE PAPER
  - render_eval_report(metrics, rows) : a benchmark accuracy run → the test report

Both emit a single .html that opens in any browser; Print → Save as PDF gives a
formal deliverable with no PDF dependency.
"""
from __future__ import annotations

from html import escape
from typing import Any

from react_review.schemas.package import EvidencePackage

_VERDICT = {
    "PASS": ("PASS", "good"), "PARTIAL": ("PARTIAL", "warn"),
    "FAIL": ("FAIL", "bad"), "INCOMPLETE": ("INCOMPLETE", "muted"),
}
_LABEL = {
    "match": ("Match", "good"), "mismatch": ("Mismatch", "bad"),
    "unit_mismatch": ("Unit mismatch", "warn"), "not_comparable": ("Not comparable", "muted"),
    "missing_source": ("Missing in source", "bad"),
    "source_access_failed": ("Source unavailable", "muted"),
    "unmatched": ("No source evidence", "muted"),
}


def _chip(label: str) -> str:
    text, cls = _LABEL.get(label, (label, "muted"))
    return f'<span class="chip {cls}">{escape(text)}</span>'


def _err(pct: float | None) -> str:
    return "—" if pct is None else f"{pct:.2f}%"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Audit report — evidence grouped by source paper
# --------------------------------------------------------------------------- #

def render_html_report(pkg: EvidencePackage) -> str:
    rep = pkg.report
    fv = pkg.final_verification
    v_text, v_cls = _VERDICT.get(rep.verdict.value, (rep.verdict.value, "muted"))
    src = {(s.study_id, s.group, s.field_type): s for s in pkg.source_items}

    tiles = [("Match", rep.n_match, "good"), ("Mismatch", rep.n_mismatch, "bad"),
             ("Unit mismatch", rep.n_unit_mismatch, "warn"),
             ("Not comparable", rep.n_not_comparable, "muted"),
             ("Flagged", len(fv.human_review_flags), "accent")]
    tiles_html = "".join(
        f'<div class="tile {c}"><div class="tv">{n}</div><div class="tk">{escape(k)}</div></div>'
        for k, n, c in tiles)

    # human-review flags, grouped by label
    grouped: dict[str, list] = {}
    for f in fv.human_review_flags:
        grouped.setdefault(f.label, []).append(f)
    flags_html = ""
    for label, items in grouped.items():
        rows = "".join(
            f'<li><code>{escape(f.study_id)}/{escape(f.group)}/{escape(f.field_type)}</code>'
            f' — {escape(f.reason)}</li>' for f in items)
        flags_html += (f'<div class="fg">{_chip(label)}'
                       f'<span class="fgn">{len(items)}</span><ul>{rows}</ul></div>')
    if not flags_html:
        flags_html = '<p class="empty">Nothing flagged for human review.</p>'

    # evidence grouped BY SOURCE PAPER (study_id), in first-seen order
    by_study: dict[str, list] = {}
    for r in rep.results:
        by_study.setdefault(r.study_id, []).append(r)

    studies_html = ""
    for study_id, results in by_study.items():
        flagged = sum(1 for r in results if r.label.value != "match")
        body_rows = ""
        for r in results:
            s = src.get((r.study_id, r.group, r.field_type))
            quote = escape(s.source_quote) if s and s.source_quote else ""
            loc = escape(s.source_location_in_paper) if s and s.source_location_in_paper else ""
            ev = (f'<div class="q">“{quote}”</div>' if quote else "") + \
                 (f'<div class="loc">{loc}</div>' if loc else "")
            body_rows += (
                f'<tr class="r-{_LABEL.get(r.label.value, ("", "muted"))[1]}">'
                f'<td>{escape(r.group)}</td>'
                f'<td>{escape(r.field_type)}</td>'
                f'<td class="num">{escape(str(r.review_value))} <span class="u">{escape(r.review_unit)}</span></td>'
                f'<td class="num">{escape(str(r.source_value))} <span class="u">{escape(r.source_unit)}</span></td>'
                f'<td>{_chip(r.label.value)}</td>'
                f'<td class="num">{_err(r.rel_error_pct)}</td>'
                f'<td class="ev">{ev or "—"}</td></tr>')
        cnt = (f'<span class="cnt">{len(results)} claims'
               + (f' · <b class="flag">{flagged} flagged</b>' if flagged else ' · all clear') + '</span>')
        studies_html += (
            f'<section class="study"><div class="stitle"><h3>{escape(study_id)}</h3>{cnt}</div>'
            '<div class="scroll"><table><thead><tr>'
            '<th>Group</th><th>Field</th><th>Review value</th><th>Source value</th>'
            '<th>Verdict</th><th>Error</th><th>Source evidence (quote · location)</th>'
            f'</tr></thead><tbody>{body_rows}</tbody></table></div></section>')

    body = (
        f'<div class="banner {v_cls}"><div class="vb">{escape(v_text)}</div>'
        f'<div class="sm">{escape(fv.summary)}</div></div>'
        f'<div class="tiles">{tiles_html}</div>'
        f'<h2>Flagged for human review</h2>{flags_html}'
        f'<h2>Evidence by source paper</h2>{studies_html}')
    return _shell(
        title=f"Audit report · {escape(pkg.run_id)}",
        eyebrow="ReAct-Review · Source Data Verification",
        h1="Systematic Review — Source Data Verification",
        rid=f"run: {escape(pkg.run_id)} · {len(rep.results)} claims · {len(by_study)} source papers",
        body=body,
        foot="Deterministic render from EvidencePackage · verdict = tolerance (mean 1% / SD 3%) + unit axis · recall-first (flag rather than miss)")


# --------------------------------------------------------------------------- #
# Benchmark accuracy report (the "test report")
# --------------------------------------------------------------------------- #

def render_eval_report(metrics: dict[str, Any], rows: list) -> str:
    if metrics.get("n", 0) == 0:
        return _shell("Benchmark report", "ReAct-Review · Benchmark",
                      "Benchmark Accuracy Report", "no rows scored",
                      "<p class='empty'>No rows scored.</p>", "")

    d, e = metrics["discrepancy"], metrics["extraction"]
    tiles = [("Label accuracy", _pct(metrics["label_accuracy"]), "accent"),
             ("Precision", _pct(d["precision"]), "good"),
             ("Recall", _pct(d["recall"]), "good"),
             ("F1", _pct(d["f1"]), "good"),
             ("Found rate", _pct(e["found_rate"]), "muted"),
             ("Value match", _pct(e["value_match_rate"]), "muted")]
    tiles_html = "".join(
        f'<div class="tile {c}"><div class="tv">{v}</div><div class="tk">{escape(k)}</div></div>'
        for k, v, c in tiles)

    dd = (f'<p class="sub">Discrepancy detection (flag = mismatch / unit mismatch): '
          f'TP {d["tp"]} · FP {d["fp"]} · FN {d["fn"]} · TN {d["tn"]} · '
          f'outcomes {escape(str(metrics["outcomes"]))}</p>')

    body_rows = ""
    for r in rows:
        ok = r.predicted_label == r.expected_label
        body_rows += (
            f'<tr class="r-{"" if ok else "bad"}">'
            f'<td><b>{escape(r.study_id)}</b><br><span class="sub">{escape(r.group)}</span></td>'
            f'<td>{escape(r.field_type)}</td>'
            f'<td>{_chip(r.expected_label)}</td>'
            f'<td>{_chip(r.predicted_label)}</td>'
            f'<td class="num">{"✓" if ok else "✗"}</td>'
            f'<td class="num">{escape(r.extracted_source)} <span class="u">vs {escape(r.expected_source)}</span></td>'
            f'<td>{escape(r.outcome)}</td></tr>')

    body = (f'<div class="tiles">{tiles_html}</div>{dd}'
            '<h2>Per-row results</h2><div class="scroll"><table><thead><tr>'
            '<th>Study / Group</th><th>Field</th><th>Expected</th><th>Predicted</th>'
            '<th>OK</th><th>Extracted vs truth</th><th>Outcome</th>'
            f'</tr></thead><tbody>{body_rows}</tbody></table></div>')
    return _shell(
        title="Benchmark Accuracy Report",
        eyebrow="ReAct-Review · Benchmark test",
        h1="Benchmark Accuracy Report",
        rid=f"{metrics['n']} rows · answer-key driven (Collector + audit)",
        body=body,
        foot="Deterministic score vs the hand-labelled benchmark answer key · recall-first")


# --------------------------------------------------------------------------- #

def _shell(title: str, eyebrow: str, h1: str, rid: str, body: str, foot: str) -> str:
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>{_CSS}</style></head><body><div class="wrap">'
            f'<div class="eyebrow">{eyebrow}</div><h1>{h1}</h1><div class="rid">{rid}</div>'
            f'{body}<footer>{escape(foot)}</footer></div></body></html>')


_CSS = """
:root{--paper:#F4F6F8;--panel:#fff;--ink:#1A2026;--muted:#5E6B77;--faint:#8A96A2;--line:#E3E7EB;
--good:#218a57;--good-bg:rgba(33,138,87,.10);--bad:#C0392B;--bad-bg:rgba(192,57,43,.09);
--warn:#B5732C;--warn-bg:rgba(181,115,44,.11);--accent:#4a54d6;--mutedc:#6b7883;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif;
--mono:ui-monospace,"Cascadia Code",Menlo,Consolas,monospace;}
@media(prefers-color-scheme:dark){:root{--paper:#0F1319;--panel:#161B21;--ink:#E7ECF1;--muted:#98A4B0;
--faint:#697580;--line:#242C34;--good:#4fc487;--good-bg:rgba(79,196,135,.12);--bad:#e8705f;--bad-bg:rgba(232,112,95,.13);
--warn:#e0975a;--warn-bg:rgba(224,151,90,.13);--accent:#8b92ff;--mutedc:#8b96a1;}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
line-height:1.55;font-size:15px;padding:clamp(18px,4vw,44px)}
.wrap{max-width:1000px;margin:0 auto}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:600}
h1{font-size:clamp(1.5rem,3vw,2rem);margin:.3em 0 .1em;letter-spacing:-.02em;text-wrap:balance}
.rid{font-family:var(--mono);font-size:12.5px;color:var(--faint)}
.banner{margin:22px 0 8px;padding:18px 22px;border-radius:12px;display:flex;align-items:center;gap:16px;
border:1px solid var(--line);background:var(--panel);flex-wrap:wrap}
.banner .vb{font-size:1.4rem;font-weight:750;padding:6px 16px;border-radius:8px;letter-spacing:.02em}
.banner.good .vb{color:var(--good);background:var(--good-bg)}.banner.bad .vb{color:var(--bad);background:var(--bad-bg)}
.banner.warn .vb{color:var(--warn);background:var(--warn-bg)}.banner.muted .vb{color:var(--mutedc);background:rgba(120,130,140,.12)}
.banner .sm{color:var(--muted);font-size:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:16px 0 8px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;text-align:center}
.tile .tv{font-size:1.7rem;font-weight:740;font-variant-numeric:tabular-nums}
.tile .tk{font-size:12px;color:var(--muted);margin-top:2px}
.tile.good .tv{color:var(--good)}.tile.bad .tv{color:var(--bad)}.tile.warn .tv{color:var(--warn)}.tile.accent .tv{color:var(--accent)}
h2{font-size:1.15rem;margin:34px 0 12px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.chip{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:99px;white-space:nowrap}
.chip.good{color:var(--good);background:var(--good-bg)}.chip.bad{color:var(--bad);background:var(--bad-bg)}
.chip.warn{color:var(--warn);background:var(--warn-bg)}.chip.muted{color:var(--mutedc);background:rgba(120,130,140,.12)}
.chip.accent{color:var(--accent);background:rgba(74,84,214,.1)}
.fg{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:10px}
.fg .fgn{font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-left:8px}
.fg ul{margin:8px 0 0;padding-left:18px}.fg li{font-size:13.5px;color:var(--muted);margin:3px 0}
.fg code{font-family:var(--mono);font-size:.86em;color:var(--ink)}
.empty{color:var(--faint)}
.study{margin-bottom:22px}
.stitle{display:flex;align-items:baseline;gap:12px;margin-bottom:8px;flex-wrap:wrap}
.stitle h3{margin:0;font-family:var(--mono);font-size:15px;letter-spacing:-.01em}
.cnt{font-size:12.5px;color:var(--faint)}.cnt .flag{color:var(--bad)}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;min-width:720px;font-size:13.5px;background:var(--panel)}
thead th{text-align:left;font-family:var(--mono);font-weight:600;font-size:11px;letter-spacing:.03em;color:var(--muted);
padding:11px 12px;border-bottom:1px solid var(--line);text-transform:uppercase}
tbody td{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}.u{color:var(--faint);font-size:.85em}
tr.r-bad td{background:var(--bad-bg)}tr.r-warn td{background:var(--warn-bg)}
.ev{max-width:280px}.ev .q{color:var(--muted);font-size:12.5px;font-style:italic}
.ev .loc{color:var(--faint);font-family:var(--mono);font-size:11px;margin-top:3px}
footer{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);color:var(--faint);
font-family:var(--mono);font-size:11px}
"""
