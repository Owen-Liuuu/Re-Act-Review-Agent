"""Terminal rendering: ASCII fallback so a GBK console never breaks a run."""
from __future__ import annotations

from react_review.hitl import (
    StepEvent,
    StepStage,
    box_chars,
    render_event,
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


def test_render_event_without_a_subject_omits_the_file_line():
    assert "file:" not in render_event(_event(subject=""), stream=_Stream("utf-8"))


def test_prompt_offers_retry_only_when_the_stage_does():
    assert "[R]etry" not in render_prompt(_event())
    assert "[R]etry" in render_prompt(_event(offers=["retry"]))
    assert "retry with [M]odel2" in render_prompt(_event(offers=["retry", "retry_alt"]))


def test_skip_is_hidden_unless_explicitly_allowed():
    # Findings are shown unconditionally until the pipeline has earned trust.
    assert "skip remaining" not in render_prompt(_event())
    assert "skip remaining" in render_prompt(_event(), allow_skip=True)
