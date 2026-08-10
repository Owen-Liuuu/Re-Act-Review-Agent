"""The production path runs under a stated contract — and says which one.

Phase 7 built prompt contracts and then left production on the old ones: an
audit run still used the legacy extraction prompt, semantic v1, live extraction
with no recording, and threw away the research context the parser had just read
out of the review. These tests hold the wiring in place, and hold the DEFAULT
where it is: switching production onto the Phase 8 contract is a later,
explicit decision, not a side effect of building the mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from react_review.agents.collector import Collector
from react_review.contracts import ContractError, repo_root
from react_review.orchestrator.audit_pipeline import AuditPipeline
from react_review.run_profile import (
    ExecutionMode,
    RunManifest,
    guard_contract_overrides,
    load_run_contract,
)
from react_review.schemas.package import EvidencePackage
from react_review.schemas.run_manifest import RunManifest as SchemaRunManifest
from react_review.tools.extract_source import (
    ExtractSourceValueInput,
    ExtractSourceValueTool,
)
from react_review.tools.extraction_profile import prompt_profile
from react_review.tools.registry import ToolRegistry

PROFILES = repo_root() / "configs" / "run_profiles"


def _cli_flags(argv: list[str]):
    """Parse a `run` command line without executing it."""
    from react_review.cli import run_parser

    return run_parser().parse_args(["--pdf", "review.pdf", *argv])


# --- the default stays where it is ---------------------------------------

def test_run_defaults_to_the_legacy_contract():
    args = _cli_flags([])
    assert args.profile is None            # resolved to configs/run_profiles/legacy.json
    contract = load_run_contract(PROFILES / "legacy.json")
    assert contract.extraction_profile == "legacy_v3"
    assert contract.semantic_prompt_profile == "semantic_v1"
    assert contract.context_policy == "cli_only"


def test_extraction_defaults_to_live_and_can_be_recorded():
    assert _cli_flags([]).extraction == "live"
    assert _cli_flags(["--extraction", "record"]).extraction == "record"


# --- the override boundary, as the CLI applies it ------------------------

def test_a_tolerance_flag_is_refused_under_a_contract_that_fixes_tolerances():
    contract = load_run_contract(PROFILES / "phase8.json")
    with pytest.raises(ContractError, match="already fixes"):
        guard_contract_overrides(contract, {"--tolerances": Path("configs/x.yaml")})


def test_the_same_flag_is_allowed_where_the_contract_is_silent():
    contract = load_run_contract(PROFILES / "legacy.json")
    guard_contract_overrides(contract, {"--tolerances": Path("configs/x.yaml")})


def test_execution_flags_are_never_overrides():
    """record now, replay later — under one unchanged contract."""
    contract = load_run_contract(PROFILES / "phase8.json")
    for mode in ("live", "record", "replay"):
        guard_contract_overrides(contract, {"--extraction": mode,
                                            "--semantic": "cache-only"})


# --- the contract reaches the tools --------------------------------------

class _Backend:
    model_id = "stub"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        self.prompts.append(prompt)
        return json.dumps({"found": False, "not_found_reason": "stub"})


@pytest.mark.asyncio
async def test_the_collector_asks_under_the_contracts_extraction_profile():
    from react_review.schemas.evidence import ReviewDataItem
    from react_review.steps.data_extraction.schemas import PaperDocument
    from react_review.steps.paper_verification.schemas import ReferenceEntry
    from react_review.steps.paper_verification.interfaces import PaperRetriever
    from react_review.tools.extract import FetchFullTextTool

    class _Retriever(PaperRetriever):
        async def retrieve(self, reference):
            return PaperDocument(paper_id="p", reference=reference,
                                 full_text="Table 1. Age 12.9")

    seen: list[str] = []

    class _Recording(ExtractSourceValueTool):
        async def run(self, payload: ExtractSourceValueInput):
            seen.append(prompt_profile(payload))
            return await super().run(payload)

    registry = ToolRegistry()
    registry.register(FetchFullTextTool(_Retriever()))
    registry.register(_Recording(_Backend()))
    collector = Collector(registry, extraction_profile="targeted_v4")
    await collector.collect(
        ReviewDataItem(study_id="s", group="t1dm", field_type="age", value="12.9"),
        ReferenceEntry(title="paper", doi="10.1/x"))
    assert seen and set(seen) == {"targeted_v4"}


# --- the manifest is written, partial runs included ----------------------

def test_a_package_records_which_rules_produced_it():
    contract = load_run_contract(PROFILES / "phase8.json")
    manifest = RunManifest.of(
        contract, ExecutionMode(extraction_mode="live"), context_source="parsed")
    package = EvidencePackage(run_id="r1", run_manifest=manifest)
    restored = EvidencePackage.model_validate(
        json.loads(package.model_dump_json()))
    assert restored.run_manifest.contract["extraction_profile"] == "targeted_v4"
    assert restored.run_manifest.context_source == "parsed"
    assert restored.run_manifest.complete is False


def test_a_package_written_before_contracts_existed_still_loads():
    """Every Phase 6/7 package on disk has no manifest at all."""
    package = EvidencePackage.model_validate({"run_id": "old"})
    assert package.run_manifest is None


@pytest.mark.asyncio
async def test_the_partial_package_carries_the_manifest_too(tmp_path):
    """A run that stops halfway must still say what rules it was applying."""
    from react_review.store import EvidencePackageStore
    from react_review.orchestrator.pipeline import AuditOrchestrator
    from react_review.orchestrator.judge import Judge
    from react_review.schemas.evidence import ReviewDataItem
    from react_review.steps.paper_verification.schemas import ReferenceEntry

    class _StubCollector:
        async def collect(self, review_item, reference, *, research_context=""):
            from react_review.agents.collector import CollectResult
            from react_review.schemas.agent import AgentRun
            from react_review.schemas.evidence import SourceEvidenceItem
            from react_review.core.enums import ReflectionDecision
            return CollectResult(
                source_item=SourceEvidenceItem(
                    study_id=review_item.study_id, group=review_item.group,
                    field_type=review_item.field_type, source_value="12.9"),
                record=AgentRun(agent="collector"),
                decision=ReflectionDecision.ACCEPT)

    from react_review.audit import ToleranceTable
    from react_review.tools.compare import CompareValuesTool

    registry = ToolRegistry()
    registry.register(CompareValuesTool(ToleranceTable()))
    store = EvidencePackageStore(tmp_path)
    manifest = RunManifest.of(
        load_run_contract(PROFILES / "legacy.json"),
        ExecutionMode(extraction_mode="live"), context_source="cli")
    pipeline = AuditPipeline(_StubCollector(), AuditOrchestrator(registry),
                             Judge(), store=store, run_manifest=manifest)
    await pipeline.run(
        [ReviewDataItem(study_id="s", group="t1dm", field_type="age", value="12.9")],
        lambda study_id: ReferenceEntry(title="paper", doi="10.1/x"),
        run_id="r2")

    partial = json.loads((tmp_path / "r2" / "package.partial.json").read_text(
        encoding="utf-8"))
    assert partial["run_manifest"]["contract"]["profile_id"] == "legacy"
    assert partial["run_manifest"]["complete"] is False


def test_the_cache_hash_is_only_recorded_once_writing_has_stopped(tmp_path):
    cache = tmp_path / "extraction_cache.json"
    cache.write_text('{"entries": {}}', encoding="utf-8")
    mode = ExecutionMode(extraction_mode="record", extraction_cache=cache)
    manifest = SchemaRunManifest.of(load_run_contract(PROFILES / "legacy.json"), mode)
    assert manifest.extraction_cache_sha256 == ""
    assert manifest.finalise(mode).extraction_cache_sha256 != ""
