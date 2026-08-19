"""The two TableCapture questions are explicit, immutable prompt contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.contracts import ContractError
from react_review.hitl import StepReporter
from react_review.llm.base import LLMBackend
from react_review.parser.table_capture import TableCapturer
from react_review.parser.table_capture_contract import (
    DEFAULT_TABLE_CAPTURE_PROFILE,
    TABLE_CAPTURE_V1,
    TABLE_CAPTURE_V2,
    TABLE_CAPTURE_V3,
    TABLE_CAPTURE_V4,
    TableCapturePromptContract,
    render_table_capture_prompt,
    sha256_rendered_prompt,
)
from react_review.run_profile import ExecutionMode, RunManifest, load_run_contract


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("table_capture_v1", "table_capture_v2", "table_capture_v3", "table_capture_v4")


class RecordingBackend(LLMBackend):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "contract-test"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.payload)


@pytest.mark.parametrize("profile", PROFILES)
def test_each_table_capture_profile_pins_the_rendered_question(profile):
    contract = TableCapturePromptContract.load(profile)
    rendered = render_table_capture_prompt(profile, **contract.fixture_inputs)

    assert contract.prompt_id == profile
    assert contract.rendered_prompt_sha256 == sha256_rendered_prompt(rendered)
    assert contract.drifts() == []


def test_contract_is_a_rendered_prompt_pin_not_a_python_source_pin():
    """Comments/refactors are outside the boundary; only model-visible bytes count."""
    contract = TableCapturePromptContract.load("table_capture_v1")
    assert contract.hash_algorithm == "sha256-rendered-utf8-v1"
    assert set(contract.fixture_inputs) == {"text"}
    assert "source_sha256" not in json.loads(contract.path.read_text(encoding="utf-8"))


def test_a_v2_prompt_edit_does_not_move_the_frozen_v1(monkeypatch):
    from react_review.parser import table_capture_contract as prompts

    v1 = TableCapturePromptContract.load("table_capture_v1")
    v2 = TableCapturePromptContract.load("table_capture_v2")
    monkeypatch.setitem(
        prompts.PROMPT_TEMPLATES,
        "table_capture_v2",
        prompts.PROMPT_TEMPLATES["table_capture_v2"] + " ",
    )

    assert v1.drifts() == []
    assert v2.drifts()


def test_unknown_table_capture_profile_is_refused():
    with pytest.raises(ContractError, match="table capture prompt profile"):
        TableCapturePromptContract.load("table_capture_v9")


@pytest.mark.asyncio
async def test_table_capturer_defaults_to_v3_and_keeps_frozen_v1():
    payload = {"research_context": "", "tables": []}
    v3_backend = RecordingBackend(payload)
    v1_backend = RecordingBackend(payload)
    v2_backend = RecordingBackend(payload)
    selected = [{"display_id": "table_1", "caption": "Table 1"}]

    await TableCapturer(v3_backend).capture(
        "fixture", reporter=StepReporter(), selected=selected)
    await TableCapturer(v1_backend, prompt_profile="table_capture_v1").capture(
        "fixture", reporter=StepReporter())
    await TableCapturer(v2_backend, prompt_profile="table_capture_v2").capture(
        "fixture", reporter=StepReporter())

    assert DEFAULT_TABLE_CAPTURE_PROFILE == "table_capture_v3"
    assert v3_backend.prompts == [render_table_capture_prompt(
        "table_capture_v3", text="fixture",
        selected='[{"display_id": "table_1", "caption": "Table 1"}]')]
    assert v1_backend.prompts == [render_table_capture_prompt("table_capture_v1", text="fixture")]
    assert v2_backend.prompts == [render_table_capture_prompt("table_capture_v2", text="fixture")]
    assert v1_backend.prompts != v2_backend.prompts
    assert "SELECTED DISPLAYS" in v3_backend.prompts[0]
    assert "SELECTED DISPLAYS" not in v1_backend.prompts[0]


def test_new_run_profile_records_table_capture_contract_in_the_manifest():
    contract = load_run_contract(ROOT / "configs" / "run_profiles" / "phase8_batch_v6.json")
    manifest = RunManifest.of(contract, ExecutionMode())
    pinned = TableCapturePromptContract.load("table_capture_v1")

    assert contract.table_capture_prompt_profile == "table_capture_v1"
    assert manifest.contract["table_capture_prompt_id"] == pinned.prompt_id
    assert manifest.contract["table_capture_prompt_hash"] == pinned.rendered_prompt_sha256


def test_old_profiles_do_not_gain_empty_table_capture_identity_keys():
    contract = load_run_contract(ROOT / "configs" / "run_profiles" / "legacy.json")
    identity = contract.identity()

    assert TABLE_CAPTURE_V1.name == "table_capture_v1.json"
    assert TABLE_CAPTURE_V2.name == "table_capture_v2.json"
    assert TABLE_CAPTURE_V3.name == "table_capture_v3.json"
    assert TABLE_CAPTURE_V4.name == "table_capture_v4.json"
    assert "table_capture_prompt_id" not in identity
    assert "table_capture_prompt_hash" not in identity
    assert "table_capture_prompt_profile" not in identity


def test_v4_differs_from_v3_only_in_the_example_cells():
    v3 = render_table_capture_prompt(
        "table_capture_v3", text="X", selected="[]").splitlines()
    v4 = render_table_capture_prompt(
        "table_capture_v4", text="X", selected="[]").splitlines()
    changed = [(a, b) for a, b in zip(v3, v4) if a != b]
    assert len(v3) == len(v4)
    assert len(changed) == 3
    joined_old = "\n".join(old for old, _new in changed)
    joined_new = "\n".join(new for _old, new in changed)
    assert "Ahmad 2022" in joined_old and "T1DM" in joined_old
    assert "Ahmad 2022" not in joined_new and "T1DM" not in joined_new
    assert "<exact header cell>" in joined_new
    assert "<exact cell text>" in joined_new
    assert "<exact printed cohort label>" in joined_new
    assert TableCapturePromptContract.load("table_capture_v4").drifts() == []

