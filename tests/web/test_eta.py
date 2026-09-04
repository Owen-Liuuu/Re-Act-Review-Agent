"""Desk ETA bands: historical min–max, formatted as a range."""
from react_review.web.eta import (
    decorate_live,
    estimate_range,
    estimate_s,
    format_range,
    upcoming,
)


def test_format_range_seconds_minutes_and_mixed():
    assert format_range(15, 90) == "15–90s"
    assert format_range(15, 150) == "15s–3 min"
    assert format_range(60, 180) == "1–3 min"
    assert format_range(10, 4500) == "10s–75 min"
    assert format_range(15, 600) == "15s–10 min"
    assert format_range(30, 30) == "30s"


def test_lens_and_localize_match_historical_bands():
    assert estimate_range("review_lens") == (15, 90)
    assert estimate_range("evidence_localize") == (15, 150)
    assert estimate_s("forest_ocr") == 180
    assert estimate_s("collect_study") == 4500


def test_upcoming_tail_is_a_range_not_a_single_sum():
    rest = upcoming(["review_pdf_loaded", "review_lens"])
    assert rest[0]["stage"] == "evidence_localize"
    assert rest[0]["estimate_label"] == "15s–3 min"
    collapsed = rest[-1]
    assert "Collect" in collapsed["title"]
    assert collapsed["estimate_min_s"] < collapsed["estimate_max_s"]
    assert "–" in collapsed["estimate_label"]


def test_decorate_live_exposes_min_max_and_label():
    live = decorate_live(
        {"stage": "review_lens", "state": "running", "elapsed_s": 12},
        [],
    )
    assert live["estimate_min_s"] == 15
    assert live["estimate_max_s"] == 90
    assert live["estimate_s"] == 90
    assert live["estimate_label"] == "15–90s"
