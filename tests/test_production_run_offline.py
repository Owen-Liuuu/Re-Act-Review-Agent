"""`react-review run`, offline, through the code that actually runs it.

Nothing else in the suite exercises this entry point. It needs a review PDF and
a model, so it was covered by no test at all — and a signature and its one call
site drifted apart twice while more than a thousand tests stayed green, caught
only by a linter.

Two things are substituted: the model, and the supply of source papers. Not the
parser. A stub parser would turn the one test that runs review parsing end to
end into a lifecycle test that never parses anything, so the real `ReviewParser`
reads a real (tiny) PDF here, the real `FieldResolver` maps its columns, and the
real extraction tools read the source. The scripted backend answers by prompt
type, which is also how the test can assert that all three of those stages
actually happened rather than assuming it from a green result.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.conftest import requires_frozen_evaluator

from react_review.cli import _run_main
from react_review.production import ProductionDependencies
from react_review.steps.data_extraction.schemas import PaperDocument
from react_review.steps.paper_verification.interfaces import PaperRetriever
from react_review.store import EvidencePackageStore

REVIEW_TEXT = """Epicardial adipose tissue in type 1 diabetes: a review

Table 1. Characteristics of included studies
Study      Group     EAT thickness (mm)
Ahmad 2022 T1DM      6.60
Ahmad 2022 Control   3.83

References
1. Ahmad S. Epicardial fat in T1DM. J Cardiol 2022. doi:10.1000/ahmad
"""

PAPER_TEXT = ("In this cohort, epicardial adipose tissue thickness was 6.60 mm "
              "in the T1DM group and 3.83 mm in the control group.")

# What the parser must find, before any model is asked to read a source paper.
CAPTURED = {
    "research_question": "EAT thickness in type 1 diabetes",
    "tables": [{
        "table_id": "table_1",
        "caption": "Characteristics of included studies",
        "header_rows": [["Study", "Group", "EAT thickness (mm)"]],
        "rows": [["Ahmad 2022", "T1DM", "6.60"],
                 ["Ahmad 2022", "Control", "3.83"]],
        "row_axis": [0],
        "difficulties": [],
    }],
}

UNPIVOTED = {"rows": [
    {"row": 0, "col": 2, "column_header": "EAT thickness (mm)",
     "cohort_label": "T1DM", "timepoint_label": "", "value": "6.60", "unit": "mm"},
    {"row": 1, "col": 2, "column_header": "EAT thickness (mm)",
     "cohort_label": "Control", "timepoint_label": "", "value": "3.83", "unit": "mm"},
]}

REFERENCES = {"studies": [
    {"citation": "Ahmad S. Epicardial fat in T1DM. J Cardiol 2022.",
     "doi": "10.1000/ahmad"}]}


class ScriptedBackend:
    """One model, answering by what it was ASKED — and remembering that it was.

    Dispatching on the prompt is what lets this test assert that review parsing,
    field resolution and source extraction each really happened. A backend that
    returned one canned answer would let a run that silently skipped a stage
    pass exactly as well as one that did not.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    @property
    def model_id(self) -> str:
        return "scripted"

    def _kind(self, prompt: str) -> str:
        if "compress the FRONT MATTER" in prompt:
            return "review_lens"
        if "RESULTS WINDOW" in prompt:
            return "evidence_localize"
        if "already-captured display" in prompt:
            return "claim_origin"
        if (("transcribing the tables of a review" in prompt
             or "transcribing SELECTED tables" in prompt
             or "STRICT TRANSCRIPTION RULES" in prompt)
                and "REVIEW TEXT" in prompt):
            return "table_capture"
        if "converting one table" in prompt:
            return "unpivot"
        if "REFERENCE LIST" in prompt:
            return "references"
        if "canonical field_type" in prompt:
            return "field_resolution"
        if "reading ONE field out of a source paper" in prompt:
            return "batch_extraction"
        if "extracting ONE specific value" in prompt:
            return "single_extraction"
        return "unknown"

    async def complete(self, prompt: str, *, seed: int = 42) -> str:
        kind = self._kind(prompt)
        self.asked.append(kind)
        if kind == "review_lens":
            return json.dumps({
                "lens_one_line": "EAT thickness in T1DM vs healthy controls",
                "domain": "cardiometabolic imaging",
                "population": "T1DM",
                "comparison": "T1DM vs healthy controls",
                "outcomes": ["EAT thickness"],
            })
        if kind == "evidence_localize":
            return json.dumps({"displays": [{
                "display_id": "table_1",
                "kind": "pdf_table",
                "caption": "Characteristics of included studies",
                "evidence_chain": True,
                "reason": "per-study EAT for included papers",
            }]})
        if kind == "claim_origin":
            return json.dumps({"labels": [{
                "table_id": "table_1",
                "column_path": "EAT thickness (mm)",
                "value_source": "source_paper",
            }]})
        if kind == "table_capture":
            return json.dumps(CAPTURED)
        if kind == "unpivot":
            return json.dumps(UNPIVOTED)
        if kind == "references":
            return json.dumps(REFERENCES)
        if kind == "field_resolution":
            return json.dumps({
                "field_type": "eat_thickness", "concept": "epicardial fat thickness",
                "value_type": "numeric", "default_unit": "mm", "scope": "cohort",
                "is_new": False, "grounded_on": ["eat_thickness"], "confidence": 0.9})
        if kind == "batch_extraction":
            return json.dumps({"readings": [
                {"arm_label": "T1DM group", "value": "6.60", "unit": "mm",
                 "quote": PAPER_TEXT, "population_phrase": "In this cohort"},
                {"arm_label": "control group", "value": "3.83", "unit": "mm",
                 "quote": PAPER_TEXT, "population_phrase": "In this cohort"}]})
        if kind == "single_extraction":
            return json.dumps({"found": True, "value": "6.60", "unit": "mm",
                               "quote": PAPER_TEXT, "source_field_name": "EAT",
                               "location": "results"})
        raise AssertionError(f"the run asked something unscripted:\n{prompt[:400]}")


class _Papers(PaperRetriever):
    async def retrieve(self, reference):
        return PaperDocument(paper_id="p1", reference=reference,
                             full_text=PAPER_TEXT,
                             metadata={"source": "test", "path": "ahmad.pdf"})


@pytest.fixture
def workspace(tmp_path):
    """A review PDF and an included-studies registry, on disk, as production reads them."""
    fitz = pytest.importorskip("fitz")

    pdf = tmp_path / "review.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 60), REVIEW_TEXT, fontsize=9)
    document.save(str(pdf))
    document.close()

    studies = tmp_path / "included_studies.csv"
    with studies.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_id", "doi", "source_pdf"])
        writer.writerow(["ahmad_2022", "10.1000/ahmad", "ahmad.pdf"])
    return pdf, studies


def _run(workspace, tmp_path, backend, *, run_id="offline1", gate=None,
         argv_extra=()):
    pdf, studies = workspace
    out = tmp_path / "runs"
    _run_main([
        "--pdf", str(pdf), "--studies", str(studies), "--out", str(out),
        "--run-id", run_id, "--non-interactive", "--checkpoints", "none",
        "--semantic", "off", "--no-checklist", *argv_extra,
    ], dependencies=ProductionDependencies(backend=backend, retriever=_Papers(),
                                           gate=gate))
    return EvidencePackageStore(out)


# --- the whole entry point, offline ----------------------------------------

def test_a_production_run_parses_resolves_extracts_and_publishes(workspace, tmp_path):
    backend = ScriptedBackend()
    store = _run(workspace, tmp_path, backend)

    # Every stage the entry point owns actually ran. Asserted from what the model
    # was asked, so a run that skipped one cannot pass by producing a package.
    assert "table_capture" in backend.asked
    assert "unpivot" in backend.asked
    assert "field_resolution" in backend.asked
    assert {"batch_extraction", "single_extraction"} & set(backend.asked)

    package = store.load("offline1")
    assert package.status == "complete"
    assert package.review_items and package.source_items
    assert package.final_verification is not None
    assert [item.review_data_id for item in package.review_items] == ["A_01", "A_02"]
    assert [item.review_data_id for item in package.source_items] == ["A_01", "A_02"]
    assert [result.audit_id for result in package.report.results] == ["A_01", "A_02"]


def test_the_published_package_is_the_one_on_disk_and_it_reloads(workspace, tmp_path):
    """The report renders from the file, so the file has to be loadable."""
    store = _run(workspace, tmp_path, ScriptedBackend())
    path = store.package_path("offline1")
    assert path.is_file()
    body = json.loads(path.read_text(encoding="utf-8-sig"))
    assert body["run_id"] == "offline1"
    assert store.load("offline1").run_id == "offline1"


def test_the_run_reports_what_it_spent(workspace, tmp_path):
    """`wall_seconds` was 0.0 on disk for as long as the package was saved from
    inside the clock."""
    store = _run(workspace, tmp_path, ScriptedBackend())
    telemetry = store.load("offline1").telemetry
    assert telemetry is not None
    assert telemetry.backend_requests > 0
    assert telemetry.wall_seconds > 0


def test_the_report_is_rendered_from_the_saved_package(workspace, tmp_path):
    store = _run(workspace, tmp_path, ScriptedBackend())
    report = store.run_dir("offline1") / "report.html"
    assert report.is_file() and report.stat().st_size > 0


def test_the_run_manifest_records_the_contract_that_governed_it(workspace, tmp_path):
    store = _run(workspace, tmp_path, ScriptedBackend())
    manifest = store.load("offline1").run_manifest
    assert manifest is not None
    assert manifest.contract["profile_id"]
    # And the modes it ran under, which are not the contract and must not be
    # recorded as if they were.
    assert manifest.execution["extraction_mode"]


# --- the ways it can end early ----------------------------------------------

def test_a_stop_at_a_checkpoint_ends_the_run_through_the_real_entry_point(
        workspace, tmp_path):
    """The stop path, driven by the thing that actually raises it.

    `RunStopped` comes from the reporter, on a gate's decision, and from nowhere
    else — so a stop test that constructs its own session proves only that the
    session handles an exception a test threw.
    """
    from react_review.hitl.gate import Decision, ScriptedCheckpoint

    out = tmp_path / "runs"
    with pytest.raises(SystemExit) as exit_code:
        _run(workspace, tmp_path, ScriptedBackend(), run_id="stopped1",
             gate=ScriptedCheckpoint([Decision.STOP]))
    assert exit_code.value.code == 2

    partial = out / "stopped1" / "package.partial.json"
    assert partial.is_file(), "a stopped run left nothing behind"
    body = json.loads(partial.read_text(encoding="utf-8-sig"))
    assert body["status"] == "stopped_by_user"
    assert body["stop_reason"]
    assert body["stopped_at_stage"]
    # The contract it was running under, on the artifact of a run that never
    # reached its first paper.
    assert body["run_manifest"]["contract"]["profile_id"]
    assert not (out / "stopped1" / "package.json").is_file()


def test_a_model_that_fails_throughout_ends_the_run_as_an_error(
        workspace, tmp_path):
    """A provider that is down is not a review with no extractable table.

    Both leave zero items, and both used to publish `status: complete` — so a
    run that never received an answer was indistinguishable from one that
    received the answer "there is nothing here". The counters already knew the
    difference; nothing acted on them. Now the run refuses to publish as
    complete, and says where it failed.
    """

    class _Broken(ScriptedBackend):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            self.asked.append(self._kind(prompt))
            raise RuntimeError("the provider is down")

    backend = _Broken()
    out = tmp_path / "runs"
    with pytest.raises(SystemExit) as exit_code:
        _run(workspace, tmp_path, backend, run_id="broken1")
    assert exit_code.value.code == 3
    assert backend.asked, "the run never even tried to reach the model"

    body = json.loads((out / "broken1" / "package.partial.json")
                      .read_text(encoding="utf-8-sig"))
    assert body["status"] == "error"
    assert body["stopped_at_stage"] == "review_parsing"
    assert "ModelUnavailable" in body["stop_reason"]
    # Nothing was published as a finished audit.
    assert not (out / "broken1" / "package.json").is_file()


def test_a_review_with_no_table_still_completes_when_the_model_answered(
        workspace, tmp_path):
    """The other zero-item run, which must NOT become an error.

    A model that answers "there are no tables here" has done its job, and the
    advisor review asked for exactly that to be stated rather than crashed on.
    Refusing to publish it would replace a silent false pass with a false
    failure, so the two zero-item runs are separated by whether an answer
    arrived — not by whether the answer was empty.
    """

    class _NoTables(ScriptedBackend):
        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            if self._kind(prompt) == "table_capture":
                self.asked.append("table_capture")
                return json.dumps({"research_question": "", "tables": []})
            return await super().complete(prompt, seed=seed)

    store = _run(workspace, tmp_path, _NoTables(), run_id="notables1")

    package = store.load("notables1")
    assert package.status == "complete"
    assert package.review_items == []


def test_one_failed_call_among_answered_ones_does_not_fail_the_run(
        workspace, tmp_path):
    """The guard fires on "never answered", not on "answered imperfectly".

    A transient failure that a retry recovers from is ordinary, and a run that
    died on it would trade a false pass for a false failure on every flaky
    provider.
    """

    class _FlakyOnce(ScriptedBackend):
        def __init__(self) -> None:
            super().__init__()
            self._failed = False

        async def complete(self, prompt: str, *, seed: int = 42) -> str:
            if not self._failed:
                self._failed = True
                self.asked.append(self._kind(prompt))
                raise RuntimeError("one transient failure")
            return await super().complete(prompt, seed=seed)

    store = _run(workspace, tmp_path, _FlakyOnce(), run_id="flaky1")

    package = store.load("flaky1")
    assert package.status == "complete"
    assert package.telemetry.backend_failures == 1
    assert package.telemetry.backend_requests > 1


# --- what may be substituted, and what may not ------------------------------

def test_only_the_model_the_papers_and_the_person_may_be_injected():
    """A `parser` field here would make this file a lifecycle test.

    The wiring is the part that keeps being wrong while every component is
    right, so the run has to build it — a test that supplies it proves only
    that the wiring a test writes works. The three that ARE injectable are the
    three a run reaches outside itself for; the gate is the human operator, not
    a stage of the pipeline.
    """
    from dataclasses import fields

    assert {f.name for f in fields(ProductionDependencies)} == {
        "backend", "retriever", "gate"}


# --- the batch contract, through the same entry point -----------------------

def test_the_batch_contract_reads_once_and_records_what_it_sent(workspace, tmp_path):
    """The route Phase 8 exists for, run end to end offline.

    The default contract is `legacy`, so every test above exercises the
    single-target route. Batching has never been run through this entry point at
    all — only through pipelines tests build themselves — which is exactly the
    gap that let the CLI ship without a runtime, without routes, and without
    telemetry on three separate occasions.
    """
    requires_frozen_evaluator()
    from react_review.contracts import repo_root
    from react_review.schemas.telemetry import BATCH_EXTRACTION

    profile = repo_root() / "configs/run_profiles/phase8_batch_v7.json"
    backend = ScriptedBackend()
    store = _run(workspace, tmp_path, backend, run_id="batched1",
                 argv_extra=("--profile", str(profile)))

    package = store.load("batched1")
    assert package.status == "complete"
    assert "batch_extraction" in backend.asked

    # One reading, answering more than one claim, resolvable in both directions.
    assert package.batch_records
    record = package.batch_records[0]
    known = {r.execution_id for r in package.batch_records}
    batched = [i for i in package.source_items if i.batch_provenance]
    assert batched and all(i.batch_provenance.batch_execution_id in known
                           for i in batched)
    assert len(record.claim_ids) == len(batched)
    assert sorted(record.claim_ids) == ["A_01", "A_02"]
    assert sorted(item.review_data_id for item in batched) == ["A_01", "A_02"]
    assert all(item.review_data_id == item.batch_provenance.claim_id
               for item in batched)
    assert sorted(result.audit_id for result in package.report.results) == ["A_01", "A_02"]

    # And it says what it SENT — the paper here is short, so nothing was cut,
    # which is itself the answer rather than a missing measurement.
    provenance = record.excerpt_provenance
    assert provenance is not None
    assert provenance.windowed is False
    assert provenance.source_chars == provenance.excerpt_chars == len(PAPER_TEXT)
    assert provenance.selection_method_id and provenance.selection_version

    # The stage buckets exist only because this contract has two routes to
    # compare, and the batch one did the reading.
    assert package.telemetry.stages[BATCH_EXTRACTION].requests >= 1


def test_the_batched_package_records_which_evaluator_decided(workspace, tmp_path):
    """A run that aggregates must name the code that did it, or its answers are
    attributed to nothing."""
    requires_frozen_evaluator()
    from react_review.contracts import repo_root

    profile = repo_root() / "configs/run_profiles/phase8_batch_v7.json"
    store = _run(workspace, tmp_path, ScriptedBackend(), run_id="batched2",
                 argv_extra=("--profile", str(profile)))

    runtime = store.load("batched2").run_manifest.aggregation_runtime
    assert runtime["policy_id"] == "safe_sum_v5"
    assert runtime["evaluator_version"] == "1.8.2"


def test_schema_v4_production_run_records_and_applies_evidence_gate(
    workspace, tmp_path,
):
    from react_review.contracts import repo_root

    profile = repo_root() / "configs/run_profiles/phase8_batch_v8.json"
    store = _run(workspace, tmp_path, ScriptedBackend(), run_id="adequacy-v8",
                 argv_extra=("--profile", str(profile)))

    package = store.load("adequacy-v8")
    runtime = package.run_manifest.adequacy_runtime
    assert runtime["policy_id"] == "evidence_adequacy_v1"
    assert runtime["evaluator_version"] == "1.0.0"
    assert runtime["release_eligible"] is True
    assert all(item.evidence_adequacy is not None for item in package.source_items)
    assert all(result.evidence_adequacy is not None for result in package.report.results)
