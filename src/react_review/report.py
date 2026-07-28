"""Render an EvidencePackage into a standalone, self-contained HTML report.

Deterministic (no LLM): everything is already in the package. Produces a single
.html file that opens in any browser and can be "Print → Save as PDF" for a
formal deliverable. Used by the ``react-review report`` CLI command.
"""
from __future__ import annotations

from html import escape

from react_review.schemas.package import EvidencePackage

_VERDICT = {
    "PASS": ("通过", "good"), "PARTIAL": ("部分通过", "warn"),
    "FAIL": ("未通过", "bad"), "INCOMPLETE": ("证据不足", "muted"),
}
_LABEL = {
    "match": ("一致", "good"), "mismatch": ("不一致", "bad"),
    "unit_mismatch": ("单位不符", "warn"), "not_comparable": ("无法比对", "muted"),
    "missing_source": ("源缺失·疑造假", "bad"), "source_access_failed": ("源未取到", "muted"),
    "unmatched": ("无源证据", "muted"),
}


def _chip(label: str) -> str:
    text, cls = _LABEL.get(label, (label, "muted"))
    return f'<span class="chip {cls}">{escape(text)}</span>'


def _err(pct: float | None) -> str:
    return "—" if pct is None else f"{pct:.2f}%"


def render_html_report(pkg: EvidencePackage) -> str:
    rep = pkg.report
    fv = pkg.final_verification
    v_text, v_cls = _VERDICT.get(rep.verdict.value, (rep.verdict.value, "muted"))
    src = {(s.study_id, s.group, s.field_type): s for s in pkg.source_items}

    tiles = [("一致", rep.n_match, "good"), ("不一致", rep.n_mismatch, "bad"),
             ("单位不符", rep.n_unit_mismatch, "warn"),
             ("无法比对", rep.n_not_comparable, "muted"),
             ("待人审", len(fv.human_review_flags), "accent")]
    tile_html = "".join(
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
                       f'<span class="fgn">{len(items)} 项</span><ul>{rows}</ul></div>')
    if not flags_html:
        flags_html = '<p class="empty">没有需要人工复核的项。</p>'

    # per-item evidence rows
    body_rows = ""
    for r in rep.results:
        s = src.get((r.study_id, r.group, r.field_type))
        quote = escape(s.source_quote) if s and s.source_quote else ""
        loc = escape(s.source_location_in_paper) if s and s.source_location_in_paper else ""
        ev = (f'<div class="q">“{quote}”</div>' if quote else "") + \
             (f'<div class="loc">{loc}</div>' if loc else "")
        body_rows += (
            f'<tr class="r-{_LABEL.get(r.label.value, ("","muted"))[1]}">'
            f'<td><b>{escape(r.study_id)}</b><br><span class="sub">{escape(r.group)}</span></td>'
            f'<td>{escape(r.field_type)}</td>'
            f'<td class="num">{escape(str(r.review_value))} <span class="u">{escape(r.review_unit)}</span></td>'
            f'<td class="num">{escape(str(r.source_value))} <span class="u">{escape(r.source_unit)}</span></td>'
            f'<td>{_chip(r.label.value)}</td>'
            f'<td class="num">{_err(r.rel_error_pct)}</td>'
            f'<td class="ev">{ev or "—"}</td></tr>')

    return _TEMPLATE.format(
        run_id=escape(pkg.run_id), v_text=escape(v_text), v_cls=v_cls,
        summary=escape(fv.summary), tiles=tile_html, flags=flags_html,
        rows=body_rows, n=len(rep.results),
    )


_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>证据审计报告 · {run_id}</title>
<style>
:root{{--paper:#F4F6F8;--panel:#fff;--ink:#1A2026;--muted:#5E6B77;--faint:#8A96A2;--line:#E3E7EB;
--good:#218a57;--good-bg:rgba(33,138,87,.10);--bad:#C0392B;--bad-bg:rgba(192,57,43,.09);
--warn:#B5732C;--warn-bg:rgba(181,115,44,.11);--accent:#4a54d6;--mutedc:#6b7883;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Microsoft YaHei",sans-serif;
--mono:ui-monospace,"Cascadia Code",Menlo,Consolas,monospace;}}
@media(prefers-color-scheme:dark){{:root{{--paper:#0F1319;--panel:#161B21;--ink:#E7ECF1;--muted:#98A4B0;
--faint:#697580;--line:#242C34;--good:#4fc487;--good-bg:rgba(79,196,135,.12);--bad:#e8705f;--bad-bg:rgba(232,112,95,.13);
--warn:#e0975a;--warn-bg:rgba(224,151,90,.13);--accent:#8b92ff;--mutedc:#8b96a1;}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
line-height:1.55;font-size:15px;padding:clamp(18px,4vw,44px)}}
.wrap{{max-width:1000px;margin:0 auto}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);font-weight:600}}
h1{{font-size:clamp(1.5rem,3vw,2rem);margin:.3em 0 .1em;letter-spacing:-.02em}}
.rid{{font-family:var(--mono);font-size:12.5px;color:var(--faint)}}
.banner{{margin:22px 0 8px;padding:18px 22px;border-radius:12px;display:flex;align-items:center;gap:16px;
border:1px solid var(--line);background:var(--panel)}}
.banner .vb{{font-size:1.5rem;font-weight:750;padding:6px 16px;border-radius:8px}}
.banner.good .vb{{color:var(--good);background:var(--good-bg)}}.banner.bad .vb{{color:var(--bad);background:var(--bad-bg)}}
.banner.warn .vb{{color:var(--warn);background:var(--warn-bg)}}.banner.muted .vb{{color:var(--mutedc);background:rgba(120,130,140,.12)}}
.banner .sm{{color:var(--muted);font-size:14px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:12px;margin:16px 0 8px}}
.tile{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;text-align:center}}
.tile .tv{{font-size:1.9rem;font-weight:740;font-variant-numeric:tabular-nums}}
.tile .tk{{font-size:12px;color:var(--muted);margin-top:2px}}
.tile.good .tv{{color:var(--good)}}.tile.bad .tv{{color:var(--bad)}}.tile.warn .tv{{color:var(--warn)}}.tile.accent .tv{{color:var(--accent)}}
h2{{font-size:1.15rem;margin:34px 0 12px;letter-spacing:-.01em}}
.chip{{display:inline-block;font-size:12px;font-weight:600;padding:2px 9px;border-radius:99px;white-space:nowrap}}
.chip.good{{color:var(--good);background:var(--good-bg)}}.chip.bad{{color:var(--bad);background:var(--bad-bg)}}
.chip.warn{{color:var(--warn);background:var(--warn-bg)}}.chip.muted{{color:var(--mutedc);background:rgba(120,130,140,.12)}}
.chip.accent{{color:var(--accent);background:rgba(74,84,214,.1)}}
.fg{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin-bottom:10px}}
.fg .fgn{{font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-left:8px}}
.fg ul{{margin:8px 0 0;padding-left:18px}}.fg li{{font-size:13.5px;color:var(--muted);margin:3px 0}}
.fg code{{font-family:var(--mono);font-size:.86em;color:var(--ink)}}
.empty{{color:var(--faint)}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:12px}}
table{{border-collapse:collapse;width:100%;min-width:760px;font-size:13.5px;background:var(--panel)}}
thead th{{text-align:left;font-family:var(--mono);font-weight:600;font-size:11px;letter-spacing:.03em;color:var(--muted);
padding:11px 12px;border-bottom:1px solid var(--line);text-transform:uppercase}}
tbody td{{padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
tbody tr:last-child td{{border-bottom:0}}
td.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}.u{{color:var(--faint);font-size:.85em}}
.sub{{color:var(--faint);font-size:.85em}}
tr.r-bad td{{background:var(--bad-bg)}}tr.r-warn td{{background:var(--warn-bg)}}
.ev{{max-width:280px}}.ev .q{{color:var(--muted);font-size:12.5px;font-style:italic}}
.ev .loc{{color:var(--faint);font-family:var(--mono);font-size:11px;margin-top:3px}}
footer{{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);color:var(--faint);
font-family:var(--mono);font-size:11px}}
</style></head><body><div class="wrap">
<div class="eyebrow">ReAct-Review · 证据审计</div>
<h1>系统综述数据核验报告</h1>
<div class="rid">run: {run_id} · 共 {n} 条声明</div>
<div class="banner {v_cls}"><div class="vb">{v_text}</div><div class="sm">{summary}</div></div>
<div class="tiles">{tiles}</div>
<h2>待人工复核</h2>{flags}
<h2>逐条证据</h2>
<div class="scroll"><table>
<thead><tr><th>研究 / 组</th><th>字段</th><th>综述值</th><th>源论文值</th><th>判定</th><th>误差</th><th>源证据(原文引用 · 位置)</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<footer>ReAct-Review · 确定性渲染自 EvidencePackage · 判定=容差(均值1%/SD3%)+单位轴 · 召回优先(宁标勿漏)</footer>
</div></body></html>"""
