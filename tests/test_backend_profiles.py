"""Backend gears: profiles, routing, reasoning injection, and traces."""
from __future__ import annotations

import json

import pytest
import yaml

from react_review.core.config import (
    BACKEND_STEPS,
    AppConfig,
    apply_profile_all,
    load_config,
)
from react_review.core.exceptions import ConfigError
from react_review.hitl import StepReporter, StepStage
from react_review.hitl.journal import RunJournal
from react_review.hitl.render import render_event
from react_review.llm.base import LLMBackend
from react_review.llm.metered import MeteredBackend
from react_review.llm.reasoning import reasoning_extra_body
from react_review.production import ProductionBackends, ProductionStages
from react_review.schemas.telemetry import RunTelemetry


def _yaml(tmp_path, body: dict):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(body), encoding="utf-8")
    return path


def test_unconfigured_profiles_are_empty():
    config = AppConfig()
    assert config.backend_profiles == {}
    assert config.routing == {}


def test_routing_unknown_step_is_a_hard_error(tmp_path):
    with pytest.raises(ConfigError, match="unknown step"):
        load_config(_yaml(tmp_path, {
            "backend_profiles": {
                "judge": {"provider": "openai", "model": "x", "reasoning": "on"},
            },
            "routing": {"not_a_step": "judge"},
        }))


def test_routing_unknown_profile_is_a_hard_error(tmp_path):
    with pytest.raises(ConfigError, match="not in backend_profiles"):
        load_config(_yaml(tmp_path, {
            "backend_profiles": {
                "judge": {"provider": "openai", "model": "x", "reasoning": "on"},
            },
            "routing": {"table_capture": "missing"},
        }))


def test_reasoning_on_unsupported_provider_is_a_hard_error(tmp_path):
    with pytest.raises(ConfigError, match="does not support the reasoning"):
        load_config(_yaml(tmp_path, {
            "backend_profiles": {
                "judge": {"provider": "mock", "model": "x", "reasoning": "on"},
            },
        }))


def test_yaml_on_is_accepted_as_reasoning_on(tmp_path):
    # YAML 1.1 parses bare `on` as boolean True; that must still mean reasoning on.
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "backend_profiles:\n  judge:\n    provider: openai\n"
        "    model: deepseek-v4-pro\n    reasoning: on\n"
        "routing:\n  table_capture: judge\n",
        encoding="utf-8")
    config = load_config(path)
    assert config.backend_profiles["judge"].reasoning == "on"
    assert config.routing["table_capture"] == "judge"


def test_profile_all_requires_the_named_gear():
    with pytest.raises(ConfigError, match="--profile-all"):
        apply_profile_all(AppConfig(), "judge")


def test_profile_all_routes_every_known_step(tmp_path):
    config = load_config(_yaml(tmp_path, {
        "backend_profiles": {
            "judge": {"provider": "openai", "model": "x", "reasoning": "on"},
        },
    }))
    pinned = apply_profile_all(config, "judge")
    assert set(pinned.routing) == set(BACKEND_STEPS)
    assert set(pinned.routing.values()) == {"judge"}


class _Named(LLMBackend):
    def __init__(self, name: str):
        super().__init__()
        self._name = name

    @property
    def model_id(self) -> str:
        return self._name

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        return "{}"


def test_unconfigured_production_backends_keep_four_telemetry_views():
    telemetry = RunTelemetry()
    stages = ProductionStages(parsing="parsing", single="single",
                              batch="batch", semantic="semantic")
    raw = _Named("llm")
    backends = ProductionBackends(raw, telemetry, stages)
    assert backends.parsing._backend is raw
    assert backends.single._backend is raw
    assert backends.batch._backend is raw
    assert backends.semantic._backend is raw
    assert backends.parsing is not backends.single
    assert backends.review_lens._backend is raw
    assert backends.table_capture._backend is raw
    assert backends.extract_transcribe is backends.single
    assert backends.semantic_compare is backends.semantic
    assert backends.forest_ocr_vision is None


def test_deepseek_and_glm_reasoning_patches_do_not_mix():
    glm_off = reasoning_extra_body("glm", reasoning="off")
    ds_off = reasoning_extra_body(
        "openai", reasoning="off", model="deepseek-v4-pro",
        base_url="https://api.deepseek.com")
    oai_off = reasoning_extra_body("openai", reasoning="off", model="gpt-4o")
    assert glm_off == {"thinking": {"type": "disabled"}}
    assert ds_off == {"thinking": {"type": "disabled"}}
    assert oai_off == {"reasoning_effort": "none"}
    assert "reasoning_effort" not in glm_off
    assert "thinking" not in oai_off


class _Probe(LLMBackend):
    def __init__(self, name: str, tokens: int | None = None):
        super().__init__()
        self._name = name
        self.patches: list[dict] = []
        self.last_usage = (
            {"completion_tokens_details": {"reasoning_tokens": tokens}}
            if tokens is not None else {"prompt_tokens": 1, "completion_tokens": 1})
        self.last_reasoning_tokens = tokens
        self._settings = type("S", (), {
            "provider": "openai", "model": name, "base_url": "",
        })()

    @property
    def model_id(self) -> str:
        return self._name

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        from react_review.llm.reasoning import current_reasoning_patch
        self.patches.append(current_reasoning_patch())
        return "{}"


@pytest.mark.asyncio
async def test_transcribe_gear_injects_off_and_records_no_reasoning_tokens():
    probe = _Probe("glm-flash")
    probe._settings = type("S", (), {
        "provider": "glm", "model": "glm-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    })()
    telemetry = RunTelemetry()
    backend = MeteredBackend(
        probe, telemetry, "", profile="transcribe",
        reasoning="off", provider="glm")
    await backend.complete("hi")
    assert probe.patches == [{"thinking": {"type": "disabled"}}]
    reporter = StepReporter("r")
    await reporter.step(StepStage.TABLE_CAPTURE, title="Displays captured")
    event = reporter.last_event
    assert event is not None
    assert event.backend_profile == "transcribe"
    assert event.backend_model_id == "glm-flash"
    assert event.backend_reasoning == "off"
    assert event.backend_reasoning_tokens is None
    assert "reasoning_tokens: None" in render_event(event)


@pytest.mark.asyncio
async def test_judge_gear_records_reasoning_tokens():
    probe = _Probe("deepseek-v4-pro", tokens=27)
    probe._settings = type("S", (), {
        "provider": "openai", "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
    })()
    backend = MeteredBackend(
        probe, RunTelemetry(), "", profile="judge",
        reasoning="on", provider="openai")
    await backend.complete("hi")
    assert probe.patches == [{"thinking": {"type": "enabled"}}]
    reporter = StepReporter("r")
    await reporter.step(StepStage.REVIEW_LENS)
    event = reporter.last_event
    assert event is not None
    assert event.backend_profile == "judge"
    assert event.backend_reasoning == "on"
    assert event.backend_reasoning_tokens == 27
    assert "reasoning_tokens: 27" in render_event(event)


@pytest.mark.asyncio
async def test_unconfigured_journal_omits_backend_trace_keys(tmp_path):
    journal = RunJournal(tmp_path)
    reporter = StepReporter("r", journal=journal)
    await reporter.step(StepStage.REVIEW_PDF_LOADED, title="Review PDF loaded")
    data = json.loads(
        (tmp_path / "steps" / "001_review_pdf_loaded.json").read_text(encoding="utf-8"))
    assert "backend_profile" not in data
    assert "backend_reasoning_tokens" not in data


class _Counter(LLMBackend):
    def __init__(self, name: str):
        super().__init__()
        self._name = name
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._name

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.calls += 1
        return '{"rows": [], "studies": []}'


@pytest.mark.asyncio
async def test_unpivot_and_references_do_not_share_a_backend():
    from react_review.dkb import FieldResolver, KnowledgeBase
    from react_review.parser.review_parser import ReviewParser

    class _Slots:
        def __init__(self):
            self.unpivot = _Counter("unpivot")
            self.references = _Counter("refs")
            self.table_capture = _Counter("capture")

    slots = _Slots()
    default = _Counter("default")
    parser = ReviewParser(
        default, FieldResolver(KnowledgeBase()), step_backends=slots)
    await parser._call("{}", slot="unpivot")
    await parser._call("{}", slot="references")
    assert slots.unpivot.calls == 1 and slots.references.calls == 1
    assert default.calls == 0


def test_profile_all_flag_is_on_the_run_parser():
    from react_review.cli import run_parser
    args = run_parser().parse_args(["--pdf", "review.pdf", "--profile-all", "judge"])
    assert args.profile_all == "judge"
