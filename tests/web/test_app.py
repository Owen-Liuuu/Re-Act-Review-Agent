"""Read-only web UI over existing run files (no live model)."""
from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from react_review.hitl import Decision, StepEvent, StepStage, WebCheckpoint
from react_review.web.app import UPLOAD_RETENTION, create_app
from react_review.web.reads import blocked_step, load_steps, run_status
from react_review.web.runner import RunRunner, write_studies_csv


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "web_run"
WAITING = Path(__file__).resolve().parents[1] / "fixtures" / "web_waiting"


def _app(tmp_path: Path, *, runner: RunRunner | None = None):
    runs = tmp_path / "runs"
    runs.mkdir()
    # Copy fixture as a historical run the browser can page through.
    dest = runs / "web_run"
    dest.mkdir()
    (dest / "journal.ndjson").write_text(
        (FIXTURE / "journal.ndjson").read_text(encoding="utf-8"), encoding="utf-8")
    (dest / "checkpoints.log").write_text(
        (FIXTURE / "checkpoints.log").read_text(encoding="utf-8"), encoding="utf-8")
    (dest / "report.html").write_text(
        (FIXTURE / "report.html").read_text(encoding="utf-8"), encoding="utf-8")
    (dest / "package.json").write_text(
        (FIXTURE / "package.json").read_text(encoding="utf-8"), encoding="utf-8")
    steps = dest / "steps"
    steps.mkdir()
    for src in (FIXTURE / "steps").iterdir():
        (steps / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    wait = runs / "web_waiting"
    wait.mkdir()
    (wait / "journal.ndjson").write_text(
        (WAITING / "journal.ndjson").read_text(encoding="utf-8"), encoding="utf-8")
    (wait / "steps").mkdir()
    (wait / "steps" / "004_review_table_capture.json").write_text(
        (WAITING / "steps" / "004_review_table_capture.json").read_text(
            encoding="utf-8"), encoding="utf-8")
    return create_app(
        runs_dir=runs, uploads_dir=tmp_path / "uploads",
        config_path=Path("configs/config.example.yaml"),
        basic_auth=None, runner=runner)


def test_reads_fixture_steps_and_report_without_a_model():
    steps = load_steps(FIXTURE)
    assert [s["index"] for s in steps] == [1, 2, 4]
    assert steps[1]["interaction"] == "web"
    assert "Elderly patients" in steps[1]["render_blocks"][0]
    assert run_status(FIXTURE) == "complete"
    waiting = load_steps(WAITING)
    blocked = blocked_step(waiting)
    assert blocked is not None and blocked["index"] == 4
    assert blocked["decision"] == ""


def test_get_run_renders_blocks_in_pre(tmp_path):
    client = TestClient(_app(tmp_path))
    home = client.get("/")
    assert home.status_code == 200
    assert "web_run" in home.text
    assert 'class="app"' in home.text
    assert 'name="review"' in home.text
    assert "Start run" in home.text
    assert "Account" in home.text
    assert "Three gears" in home.text
    assert "deepseek/deepseek-v4-pro" in home.text
    assert "Idle · pick models" in home.text
    assert 'id="key-hint"' in home.text
    assert "Open Account" in home.text
    page = client.get("/runs/web_run")
    assert page.status_code == 200
    assert 'class="app"' in page.text
    assert 'class="ide"' in page.text
    assert 'class="report-frame"' in page.text
    assert 'src="/runs/web_run/report"' in page.text
    assert "New run" in page.text
    assert "package.json" in page.text
    report = client.get("/runs/web_run/report")
    assert report.status_code == 200
    assert "fixture report" in report.text
    pkg = client.get("/runs/web_run/package")
    assert pkg.status_code == 200
    assert pkg.json()["status"] == "complete"
    log = client.get("/runs/web_run/log")
    assert log.status_code == 200
    assert "interaction: web" in log.text


def test_refresh_rereads_files(tmp_path):
    client = TestClient(_app(tmp_path))
    first = client.get("/runs/web_run")
    second = client.get("/runs/web_run")
    assert first.text == second.text
    assert 'src="/runs/web_run/report"' in second.text


def test_decision_requires_the_waiting_index(tmp_path):
    gate = WebCheckpoint()
    event = StepEvent(
        run_id="live1", index=4, stage=StepStage.TABLE_CAPTURE,
        title="Displays captured",
        payload={"tables": [{"id": "table_1", "label": "table_1"}]},
        selectable="tables", offers=["retry"],
        render_blocks=["  table_1"],
    )

    def _start(argv, g):
        import asyncio
        asyncio.run(g.check(event))

    runner = RunRunner(_start)
    runner.enqueue(("live1", ["--pdf", "x"], gate))
    client = TestClient(_app(tmp_path, runner=runner))
    # Wait until the gate is actually blocked.
    import time
    for _ in range(50):
        if gate.waiting is not None:
            break
        time.sleep(0.02)
    assert gate.waiting is not None
    stale = client.post("/runs/live1/decision", json={"index": 3, "decision": "continue"})
    assert stale.status_code == 409
    ok = client.post("/runs/live1/decision", json={"index": 4, "decision": "continue"})
    assert ok.status_code in {200, 303}
    for _ in range(50):
        if event.interaction == "web":
            break
        time.sleep(0.02)
    assert event.interaction == "web"
    assert event.decision == Decision.CONTINUE.value


def test_basic_auth_is_required_when_configured(tmp_path):
    app = create_app(
        runs_dir=tmp_path / "runs", uploads_dir=tmp_path / "up",
        config_path=Path("configs/config.example.yaml"),
        basic_auth=("reviewer", "secret"))
    (tmp_path / "runs").mkdir()
    client = TestClient(app)
    assert client.get("/").status_code == 401
    ok = client.get("/", auth=("reviewer", "secret"))
    assert ok.status_code == 200


def test_studies_csv_names_uploaded_pdfs(tmp_path):
    pdf = tmp_path / "Capovilla 2023.pdf"
    pdf.write_bytes(b"%PDF")
    csv_path = tmp_path / "included_studies.csv"
    write_studies_csv(csv_path, [pdf])
    text = csv_path.read_text(encoding="utf-8")
    assert "Capovilla 2023.pdf" in text
    assert "study_id" in text


def test_run_json_is_pollable(tmp_path):
    client = TestClient(_app(tmp_path))
    res = client.get("/runs/web_run", headers={"Accept": "application/json"})
    assert res.status_code == 200
    body = res.json()
    assert body["run_id"] == "web_run"
    assert body["status"] == "complete"
    assert [s["index"] for s in body["steps"]] == [1, 2, 4]
    assert body["has_report"] is True
    assert body["live"] is None


def test_running_json_includes_live_eta(tmp_path):
    import json
    import time

    client = TestClient(_app(tmp_path))
    wait = tmp_path / "runs" / "web_waiting"
    (wait / "live.json").write_text(json.dumps({
        "stage": "forest_ocr",
        "label": "figure",
        "caption": "fig_2",
        "index": 1,
        "total": 2,
        "started_unix": time.time() - 41,
        "updated_unix": time.time(),
        "state": "running",
    }), encoding="utf-8")
    res = client.get("/runs/web_waiting", headers={"Accept": "application/json"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "running"
    assert body["live"]["stage"] == "forest_ocr"
    assert body["live"]["estimate_s"] == 180
    assert body["live"]["estimate_min_s"] == 60
    assert body["live"]["estimate_max_s"] == 180
    assert body["live"]["estimate_label"] == "1–3 min"
    assert body["live"]["elapsed_s"] >= 40
    assert "Remaining" not in json.dumps(body["live"])
    page = client.get("/runs/web_waiting")
    assert "Elapsed" in page.text
    assert "Estimate" in page.text
    assert "Remaining" not in page.text
    assert 'name="decision"' not in page.text
    assert "1–3 min" in page.text


def test_upload_requires_a_review_pdf(tmp_path):
    client = TestClient(_app(tmp_path, runner=RunRunner(lambda argv, gate: None)))
    res = client.post("/runs", data={})
    assert res.status_code == 400
    assert "No review PDF detected" in res.text
    assert 'class="app"' in res.text


def test_upload_redirects_to_a_run_page_that_exists(tmp_path):
    client = TestClient(_app(tmp_path, runner=RunRunner(lambda argv, gate: None)))
    res = client.post(
        "/runs",
        files={"review": ("review.pdf", b"%PDF-1.4", "application/pdf")},
        follow_redirects=False,
    )
    assert res.status_code == 303
    loc = res.headers["location"]
    assert loc.startswith("/runs/")
    page = client.get(loc)
    assert page.status_code == 200


def test_waiting_form_posts_to_this_run():
    from react_review.web import html as pages

    body = pages.run_page(
        "abc",
        status="waiting",
        steps=[{
            "index": 4,
            "title": "Displays captured",
            "render_blocks": ["  table_1"],
            "payload": {"tables": [{"id": "table_1", "label": "Table 1"}]},
            "selectable": "tables",
            "offers": ["retry"],
            "decision": "",
            "interaction": "",
        }],
        blocked={"index": 4},
        has_report=False,
        has_log=False,
    )
    assert 'action="/runs/abc/decision"' in body
    assert 'name="index" value="4"' in body
    assert "<pre>" in body
    assert 'name="keep" value="table_1"' in body
    assert 'class="ide"' in body
    assert "ide running" not in body
    assert 'http-equiv="refresh"' not in body
    assert 'data-status="waiting"' in body


def test_running_page_has_no_decision_controls():
    from react_review.web import html as pages

    body = pages.run_page(
        "abc",
        status="running",
        steps=[{
            "index": 2,
            "title": "Review lens compressed",
            "stage": "review_lens",
            "render_blocks": ["  review focus"],
            "elapsed_ms": 18000,
        }],
        blocked=None,
        has_report=False,
        has_log=False,
        live={
            "stage": "forest_ocr",
            "title": "Forest OCR",
            "caption": "fig_2",
            "elapsed_s": 41,
            "estimate_s": 180,
            "estimate_min_s": 60,
            "estimate_max_s": 180,
            "estimate_label": "1–3 min",
            "started_unix": 1,
        },
    )
    assert 'name="decision"' not in body
    assert "Continue" not in body
    assert "ide running" in body
    assert "Elapsed" in body
    assert "Estimate" in body
    assert "Remaining" not in body
    assert 'http-equiv="refresh"' not in body
    assert "application/json" in body
    assert 'data-run-id="abc"' in body
    assert 'data-status="running"' in body
    assert "41s" in body
    assert "1–3 min" in body
    assert "Forest OCR" in body


def test_upload_writes_redacted_gears_and_launch(tmp_path):
    import json
    import time

    caught = {}

    def _start(argv, gate, overlay=None):
        caught["overlay"] = overlay

    client = TestClient(_app(tmp_path, runner=RunRunner(_start)))
    res = client.post(
        "/runs",
        data={
            "name": "Li Wei",
            "email": "li@lab.org",
            "complex_key": "sk-complex",
            "simple_key": "sk-simple",
            "visual_key": "glm-visual",
        },
        files={"review": ("review.pdf", b"%PDF-1.4", "application/pdf")},
        follow_redirects=False,
    )
    assert res.status_code == 303
    rid = res.headers["location"].rsplit("/", 1)[-1]
    gears = json.loads(
        (tmp_path / "uploads" / rid / "gears.json").read_text(encoding="utf-8"))
    blob = json.dumps(gears)
    assert "api_key" not in blob
    assert "sk-complex" not in blob
    assert gears["name"] == "Li Wei"
    assert gears["complex"]["model"] == "deepseek-v4-pro"
    launch = json.loads(
        (tmp_path / "uploads" / rid / "launch.json").read_text(encoding="utf-8"))
    assert "sk-complex" not in json.dumps(launch)
    retention = (tmp_path / "uploads" / rid / "RETENTION.txt").read_text(encoding="utf-8")
    assert retention == UPLOAD_RETENTION
    assert "shareable=false" in retention
    for _ in range(50):
        if "overlay" in caught:
            break
        time.sleep(0.02)
    assert caught["overlay"] is not None
    assert caught["overlay"].llm.api_key == "sk-complex"
    assert caught["overlay"].backend_profiles["transcribe"].api_key == "sk-simple"


def test_openrouter_key_is_rejected_on_upload(tmp_path):
    client = TestClient(_app(tmp_path, runner=RunRunner(lambda argv, gate: None)))
    res = client.post(
        "/runs",
        data={
            "complex_key": "sk-or-v1-nope",
            "simple_key": "sk-or-v1-nope",
            "visual_key": "sk-or-v1-nope",
        },
        files={"review": ("review.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert res.status_code == 400
    assert "OpenRouter" in res.text


def test_live_host_without_keys_requires_account(tmp_path):
    cfg = tmp_path / "live.yaml"
    cfg.write_text(
        "mock_mode: false\nllm:\n  provider: openai\n  api_key: ''\n",
        encoding="utf-8")
    app = create_app(
        runs_dir=tmp_path / "runs", uploads_dir=tmp_path / "up",
        config_path=cfg, basic_auth=None,
        runner=RunRunner(lambda argv, gate: None))
    (tmp_path / "runs").mkdir()
    client = TestClient(app)
    res = client.post(
        "/runs",
        files={"review": ("review.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert res.status_code == 400
    assert "API keys missing" in res.text
    assert 'data-require-keys="1"' in res.text


def test_fail_json_renders_failed_not_queued(tmp_path):
    import json

    client = TestClient(_app(tmp_path))
    dead = tmp_path / "runs" / "dead"
    dead.mkdir()
    (dead / "journal.ndjson").write_text("", encoding="utf-8")
    (dead / "fail.json").write_text(
        json.dumps({"state": "failed", "error": "HTTP 402: no remaining credit"}),
        encoding="utf-8")
    (dead / "live.json").write_text(json.dumps({
        "stage": "failed",
        "label": "failed",
        "caption": "HTTP 402: no remaining credit",
        "state": "failed",
        "started_unix": 1,
        "updated_unix": 1,
    }), encoding="utf-8")
    body = client.get("/runs/dead", headers={"Accept": "application/json"}).json()
    assert body["status"] == "failed"
    assert "402" in (body["live"] or {}).get("caption", "")
    assert body["live"]["upcoming"] == []
    page = client.get("/runs/dead")
    assert page.status_code == 200
    assert "Failed" in page.text
    assert "402" in page.text
    assert "Queued" not in page.text
    assert "New run" in page.text
    assert 'data-status="failed"' in page.text


def test_systemexit_does_not_kill_the_worker(tmp_path):
    import json
    import time

    from react_review.web.runner import make_gate

    seen: list[str] = []

    def _start(argv, gate):
        rid = argv[argv.index("--run-id") + 1]
        seen.append(rid)
        if rid == "a":
            raise SystemExit(3)

    runner = RunRunner(_start)
    out = tmp_path / "out"
    runner.enqueue(("a", ["--out", str(out), "--run-id", "a"], make_gate()))
    runner.enqueue(("b", ["--out", str(out), "--run-id", "b"], make_gate()))
    for _ in range(100):
        if seen == ["a", "b"] and (out / "a" / "fail.json").is_file():
            break
        time.sleep(0.02)
    assert seen == ["a", "b"]
    fail = json.loads((out / "a" / "fail.json").read_text(encoding="utf-8"))
    assert fail["state"] == "failed"
    assert "credit" in fail["error"].lower() or "402" in fail["error"]


def test_head_of_queue_is_running_not_queued():
    import time

    from react_review.web.runner import make_gate

    hold = True

    def _start(argv, gate):
        while hold:
            time.sleep(0.01)

    runner = RunRunner(_start)
    try:
        runner.enqueue(("x", ["--out", "o", "--run-id", "x"], make_gate()))
        for _ in range(50):
            if runner.current == "x" or runner.starting("x"):
                break
            time.sleep(0.02)
        assert runner.starting("x")
        assert not runner.queued("x")
    finally:
        hold = False


def test_in_progress_partial_is_running_while_the_worker_holds_it(tmp_path):
    import json
    import time

    from react_review.web.runner import make_gate

    hold = True

    def _start(argv, gate):
        while hold:
            time.sleep(0.01)

    runner = RunRunner(_start)
    rid = "partial_live"
    client = TestClient(_app(tmp_path, runner=runner))
    dest = tmp_path / "runs" / rid
    dest.mkdir()
    (dest / "journal.ndjson").write_text("", encoding="utf-8")
    (dest / "package.partial.json").write_text(
        json.dumps({"status": "in_progress"}), encoding="utf-8")
    try:
        runner.enqueue(
            (rid, ["--out", str(tmp_path / "out"), "--run-id", rid], make_gate()))
        for _ in range(50):
            if runner.current == rid:
                break
            time.sleep(0.02)
        body = client.get(
            f"/runs/{rid}", headers={"Accept": "application/json"}).json()
        assert body["status"] == "running"
    finally:
        hold = False
