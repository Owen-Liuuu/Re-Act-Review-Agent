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
    "provisional_concept": ("Provisional concept", "accent"),
    "concept_contradicted": ("Concept contradicted by source", "bad"),
    "needs_review": ("Unresolved — needs review", "warn"),
    "unresolved_source": ("Source paper not identified", "muted"),
    "unknown_cohort": ("Cohort not identified", "warn"),
    "cohort_ambiguous": ("Cohort unconfirmed", "warn"),
    "ambiguous_match_key": ("Ambiguous — not paired", "warn"),
    "checklist_gap": ("Required checklist gap", "warn"),
}


def _locator(item) -> tuple:
    """The full identity of one audited cell (see Judge._locator)."""
    return (
        item.study_id, item.group, getattr(item, "timepoint", "single"),
        item.field_type, getattr(item, "table_id", "") or "",
        getattr(item, "cell_ref", None),
        getattr(item, "checklist_id", "") or "",
    )


def _provenance_html(item) -> str:
    """WHICH document this evidence was read from — the point of recording it."""
    if item is None:
        return ""
    where = item.source_file or item.source_uri or item.source_doi
    if not where:
        return ""
    kind = f" · {escape(item.retriever_kind)}" if item.retriever_kind else ""
    return f'<div class="src">read from: <code>{escape(where)}</code>{kind}</div>'


def _reasons_html(item) -> str:
    """Why this outcome — including whatever the model said about its difficulty."""
    reasons = getattr(item, "reasons", None) or []
    if not reasons:
        return ""
    rows = "".join(f'<li><b>{escape(r.code)}</b> ({escape(r.source)}): '
                   f'{escape(r.message)}</li>' for r in reasons)
    return f'<ul class="why">{rows}</ul>'


def _semantic_html(result) -> str:
    """Show what the model claimed AND which controls let the claim stand.

    A model-reached verdict is only auditable if the reader sees what it argued,
    what it cited, and which deterministic check would have stopped it. A
    semantic MATCH rendered as a plain MATCH is precisely the opaque output this
    report exists to prevent.
    """
    v = getattr(result, "semantic", None)
    if v is None:
        return ""
    checks = getattr(result, "semantic_controls", None) or {}
    marks = "".join(
        f'<span class="ck {"ok" if passed else "bad"}">{escape(name)} '
        f'{"&#10003;" if passed else "&#10007;"}</span>'
        for name, passed in checks.items())
    span = (f'<div class="q">cited: “{escape(v.evidence_span)}”</div>'
            if v.evidence_span else "")
    return (f'<div class="sem"><b>semantic · {escape(v.relation)}</b> '
            f'(confidence {v.confidence:.2f})'
            f'<div class="loc">{escape(v.rationale)}</div>{span}'
            f'<div class="cks">{marks}</div></div>')


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
    from react_review.schemas.report import AuditReport, FinalVerification

    rep = pkg.report or AuditReport(run_id=pkg.run_id)
    fv = pkg.final_verification or FinalVerification(
        run_id=pkg.run_id, verdict=rep.verdict, summary=rep.summary)
    v_text, v_cls = _VERDICT.get(fv.verdict.value, (fv.verdict.value, "muted"))
    # Keyed on the full locator: two rows of the same study/cohort/field would
    # otherwise collapse here, showing one row's quote next to the other's number.
    src = {_locator(s): s for s in pkg.source_items}

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
            s = src.get(_locator(r))
            quote = escape(s.source_quote) if s and s.source_quote else ""
            loc = escape(s.source_location_in_paper) if s and s.source_location_in_paper else ""
            ev = (f'<div class="q">“{quote}”</div>' if quote else "") + \
                 (f'<div class="loc">{loc}</div>' if loc else "") + \
                 _provenance_html(s) + _reasons_html(s) + _semantic_html(r)
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
    safety = metrics.get("safety", {})
    tiles = [("Label accuracy", _pct(metrics["label_accuracy"]), "accent"),
             ("Precision", _pct(d["precision"]), "good"),
             ("Recall", _pct(d["recall"]), "good"),
             ("F1", _pct(d["f1"]), "good"),
             ("Silent releases", str(safety.get("silent_release_count", 0)), "good"),
             ("Review visibility", _pct(safety.get("review_visibility_rate")), "good"),
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
        audit_detail = " · ".join(x for x in (
            getattr(r, "match_mode", ""), getattr(r, "match_reason", "")) if x)
        source_detail = " · ".join(x for x in (
            getattr(r, "source_unit", ""), getattr(r, "value_origin", ""),
            getattr(r, "source_file", "")) if x)
        quote = getattr(r, "source_quote", "")
        body_rows += (
            f'<tr class="r-{"" if ok else "bad"}">'
            f'<td><b>{escape(r.study_id)}</b><br><span class="sub">{escape(r.group)}</span></td>'
            f'<td>{escape(r.field_type)}</td>'
            f'<td>{_chip(r.expected_label)}</td>'
            f'<td>{_chip(r.predicted_label)}<br><span class="sub">{escape(audit_detail)}</span></td>'
            f'<td class="num">{"✓" if ok else "✗"}</td>'
            f'<td class="num">{escape("" if r.extracted_source is None else str(r.extracted_source))} '
            f'<span class="u">vs {escape("" if r.expected_source is None else str(r.expected_source))}</span>'
            f'<br><span class="sub">{escape(source_detail)}</span>'
            f'<br><span class="sub">{escape(quote)}</span></td>'
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
# Parser accuracy report (parser stage vs review_ground_truth)
# --------------------------------------------------------------------------- #

def render_parser_report(stats: dict[str, Any]) -> str:
    tiles = [("Field coverage", _pct(stats["recall"]), "accent"),
             ("Value match · aligned", _pct(stats["value_match"]), "good"),
             ("Field precision", _pct(stats["precision"]), "muted")]
    tiles_html = "".join(
        f'<div class="tile {c}"><div class="tv">{v}</div><div class="tk">{escape(k)}</div></div>'
        for k, v, c in tiles)
    note = (f'<p class="sub">Parser produced <b>{stats["n_parser"]}</b> rows vs '
            f'<b>{stats["n_gt"]}</b> ground-truth rows; <b>{stats["n_matched"]}</b> keys aligned. '
            'Value match is measured only on aligned keys — i.e. once the parser and the '
            'ground truth agree on WHAT to extract, how often the extracted VALUE is correct.</p>')

    def _kv(title: str, d: dict) -> str:
        if not d:
            return f'<div class="fg"><b>{escape(title)}</b><span class="empty"> — none</span></div>'
        items = "".join(f'<li><code>{escape(k)}</code> <span class="fgn">×{v}</span></li>'
                        for k, v in sorted(d.items(), key=lambda x: -x[1]))
        return f'<div class="fg"><b>{escape(title)}</b><ul>{items}</ul></div>'

    missed = _kv("Missed (in ground truth, no matching parser key)", stats.get("missed", {}))
    spurious = _kv("Extra (parser produced, no matching ground-truth key)", stats.get("spurious", {}))

    mm = stats.get("mismatched_values", [])
    mm_html = ""
    if mm:
        rows = "".join(
            f'<tr class="r-warn"><td><b>{escape(m["study"])}</b><br>'
            f'<span class="sub">{escape(m["group"])}</span></td><td>{escape(m["field"])}</td>'
            f'<td class="num">{escape(str(m["parser_value"]))}</td>'
            f'<td class="num">{escape(str(m["gt_value"]))}</td></tr>' for m in mm)
        mm_html = ('<h2>Value mismatches (aligned key, value differs)</h2>'
                   '<div class="scroll"><table><thead><tr><th>Study / Group</th><th>Field</th>'
                   '<th>Parser value</th><th>Ground truth</th></tr></thead>'
                   f'<tbody>{rows}</tbody></table></div>')

    body = (f'<div class="tiles">{tiles_html}</div>{note}'
            f'<h2>Coverage gaps</h2>{missed}{spurious}{mm_html}')
    return _shell(
        title="Parser Accuracy Report", eyebrow="ReAct-Review · Parser stage",
        h1="Parser Accuracy Report",
        rid=f"review PDF → long table vs review_ground_truth ({stats['n_gt']} rows)",
        body=body,
        foot="Deterministic score of the real parser output vs the hand-labelled review ground truth")


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
.ev .src{color:var(--faint);font-size:11px;margin-top:4px;word-break:break-all}
.ev .src code{font-family:var(--mono);font-size:10.5px}
.ev .why{margin:4px 0 0;padding-left:14px;color:var(--muted);font-size:11px}
.ev .why b{font-family:var(--mono);font-weight:600}
.ev .sem{margin-top:6px;padding:5px 7px;border-left:2px solid var(--line);font-size:11px}
.ev .sem>b{font-family:var(--mono);font-weight:600}
.ev .cks{margin-top:4px;display:flex;flex-wrap:wrap;gap:4px}
.ev .ck{font-family:var(--mono);font-size:10px;padding:1px 5px;border-radius:3px;
border:1px solid var(--line);color:var(--muted)}
.ev .ck.bad{border-color:var(--bad);color:var(--bad)}
footer{margin-top:36px;padding-top:16px;border-top:1px solid var(--line);color:var(--faint);
font-family:var(--mono);font-size:11px}
"""
