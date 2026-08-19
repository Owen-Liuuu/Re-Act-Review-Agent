"""Terminal rendering: ASCII fallback so a GBK console never breaks a run."""
from __future__ import annotations

from react_review.hitl import (
    StepEvent,
    StepStage,
    box_chars,
    render_event,
    render_progress,
    render_prompt,
    rule,
    supports_unicode,
)


class _Stream:
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding


def test_unicode_support_probe():
    assert supports_unicode(_Stream("utf-8")) is True
    # GBK (the usual Windows Chinese console) DOES carry box glyphs and ±; its
    # actual limitation is emoji, which safe_print handles. Probe the real chars.
    assert supports_unicode(_Stream("gbk")) is True
    assert supports_unicode(_Stream("ascii")) is False
    assert supports_unicode(_Stream("latin-1")) is False
    assert supports_unicode(_Stream("no-such-codec")) is False   # LookupError path


def test_box_chars_fall_back_to_ascii():
    assert box_chars(_Stream("utf-8"))["h"] == "─"
    assert box_chars(_Stream("ascii"))["h"] == "-"


def test_rule_uses_the_stream_charset():
    assert rule("Step", width=20, stream=_Stream("ascii")).startswith("-- Step ")
    assert len(rule(width=20, stream=_Stream("ascii"))) == 20


def _event(**kw) -> StepEvent:
    base = dict(run_id="r", index=3, stage=StepStage.COLLECT_STUDY,
                title="Source evidence", subject="C:/pdf/Ahmad 2022.pdf")
    base.update(kw)
    return StepEvent(**base)


def test_render_event_shows_the_file_and_the_content():
    out = render_event(_event(render_blocks=["  eat_thickness: 6.60 ± 0.71"],
                              warnings=["control/age: missing_source"]),
                       stream=_Stream("utf-8"))
    assert "C:/pdf/Ahmad 2022.pdf" in out          # requirement #1: which file
    assert "eat_thickness: 6.60" in out            # requirement #2: full content
    assert "! control/age: missing_source" in out
    assert "[3] Source evidence" in out
    assert "\n\n  file: C:/pdf/Ahmad 2022.pdf" in out


def test_render_event_without_a_subject_omits_the_file_line():
    assert "file:" not in render_event(_event(subject=""), stream=_Stream("utf-8"))


def test_prompt_keys_use_letter_outside_the_verb():
    text = render_prompt(_event())
    assert "[C]Continue" in text and "[S]Stop" in text
    assert "[D]Detail" in text and "[O]Open artifact" in text
    assert "[C]ontinue" not in text
    assert "[T]Toggle one" not in render_prompt(_event(
        selectable="tables", payload={"tables": [{"id": "t1", "label": "t1"}]}))
    assert "[N]On <n>" in render_prompt(_event(
        selectable="tables", payload={"tables": [{"id": "t1", "label": "t1"}]}))
    assert "[F]Off <n>" in render_prompt(_event(
        selectable="tables", payload={"tables": [{"id": "t1", "label": "t1"}]}))
    assert "[M]Retry with Model 2" not in render_prompt(_event(offers=["retry"]))
    assert "[R]Retry" not in render_prompt(_event())
    assert "[R]Retry" in render_prompt(_event(offers=["retry"]))
    assert "[M]Retry with Model 2" in render_prompt(_event(offers=["retry", "retry_alt"]))


def test_skip_is_hidden_unless_explicitly_allowed():
    # Findings are shown unconditionally until the pipeline has earned trust.
    assert "skip remaining" not in render_prompt(_event())
    assert "skip remaining" in render_prompt(_event(), allow_skip=True)


def test_progress_line_is_discrete_with_fraction_and_elapsed():
    line = render_progress(
        "table", 1, 1,
        caption="Table 1. Characteristics of included studies.",
        elapsed_s=11,
    )
    assert "1/1" in line
    assert "11s" in line
    assert "\r" not in line
    elapsed_only = render_progress("review_lens", elapsed_s=9.2)
    assert "9s" in elapsed_only
    assert "/" not in elapsed_only.split("review_lens", 1)[1]
