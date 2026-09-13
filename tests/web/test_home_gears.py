"""The homepage says which tasks each gear serves, from the routing it will use."""
from __future__ import annotations

from pathlib import Path

from react_review.core.config import BACKEND_STEPS, load_config
from react_review.llm.catalog import SIMPLE_STEPS
from react_review.web.html import gear_tasks, home_page

ROOT = Path(__file__).resolve().parents[2]


def test_unrouted_host_puts_only_the_forced_copying_tasks_on_transcribe():
    tasks = gear_tasks({})
    assert set(tasks["simple"]) == set(SIMPLE_STEPS)
    assert tasks["visual"] == ["forest_ocr_vision"]
    assert "review_lens" in tasks["complex"]
    assert "evidence_localize" in tasks["complex"]
    assert tasks["other"] == []


def test_example_config_routing_is_what_the_gears_list():
    """The old fixed text named Lens under Complex after the config moved it."""
    routing = load_config(ROOT / "configs" / "config.example.yaml").routing
    tasks = gear_tasks(routing)
    assert tasks["complex"] == ["evidence_localize", "semantic_compare"]
    assert "review_lens" in tasks["simple"]
    assert "extract_transcribe" in tasks["simple"]
    assert tasks["visual"] == ["forest_ocr_vision"]


def test_every_task_is_listed_exactly_once():
    tasks = gear_tasks({"semantic_compare": "transcribe", "review_lens": "judge"})
    listed = [step for group in tasks.values() for step in group]
    assert sorted(listed) == sorted(BACKEND_STEPS)
    assert tasks["other"] == ["review_lens"]


def test_home_page_renders_the_routed_lists_not_fixed_text():
    page = home_page([], routing={"review_lens": "transcribe"})
    assert "Lens, localize, field map, extract, semantic compare" not in page
    complex_slot = page.split('name="complex_model"', 1)[1].split('class="slots">', 1)[1]
    assert complex_slot.startswith("Evidence located")
    assert "Review lens" not in complex_slot.split("</p>", 1)[0]
